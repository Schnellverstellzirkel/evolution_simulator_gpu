//! One step of evolution for the worker's `Loop`. A search pass runs ring
//! steps on a helper thread while the worker thread keeps reading commands.
//! When a pass ends a generation, the generation boundary handles the world
//! change, the logs, the benchmark, the autosave and the end of a run of one
//! generation.

use super::{Command, EventKind, Loop, log_event, log_world_change, note_generation};
use crate::{config::Config, storage::Experiment};
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
    },
    time::{Duration, Instant},
};

/// What one search pass did. The helper thread fills in `step`, `breeding`
/// and `blocks`. The worker thread adds `pings`.
#[derive(Default)]
struct Pass {
    /// Blocks absorbed and generations ended, added up over the ring steps of
    /// the pass. Its `chain` stays 0.
    step: crate::ring::Step,
    /// Send and read times of the pings read during the pass.
    pings: Vec<(Instant, Instant)>,
    /// Steps that absorbed a block: start, end, and whether it ended a
    /// generation.
    breeding: Vec<(Instant, Instant, bool)>,
    /// One note for each block the steps absorbed, in order.
    blocks: Vec<BlockNote>,
}
/// What one absorbed block tells the ring meter and the stage log.
struct BlockNote {
    /// The generation running when the block was absorbed. A block that ends
    /// a generation counts for the one it ended.
    generation: u32,
    /// The block's host time in seconds (`ring::Step::chain`).
    chain: f64,
    /// Whether the block ended its generation.
    boundary: bool,
    /// The engines' idle seconds so far, added up over all devices.
    idle: f64,
}
/// A search pass ends at the first ring step that finishes after this long.
/// Snapshots are built between passes.
const PASS_LIMIT: Duration = Duration::from_millis(100);
/// Runs ring steps on `helper`, a thread on the pool's CPUs, until a
/// generation ends, a command arrives or `PASS_LIMIT` has passed. Each step
/// absorbs at most one block.
///
/// Absorbing and breeding a block takes a few tenths of a second at 3M. This
/// thread keeps reading commands through it, every millisecond. A ping is
/// read at once and its read time goes into `Pass::pings`. The helper holds
/// the experiment, the scheduler and the ring until the pass ends, so any
/// other command waits in `deferred`, in order, and the pass ends after the
/// step in progress. The next worker pass runs the deferred commands first.
fn search_pass(
    helper: &crate::threads::Helper,
    e: &mut Experiment,
    sched: &mut crate::scheduler::Scheduler,
    ring: &mut crate::ring::Ring,
    rx: &Receiver<Command>,
    deferred: &mut Vec<Command>,
) -> anyhow::Result<Pass> {
    let stop = AtomicBool::new(false);
    let mut pings = Vec::new();
    let mut result = helper.run(
        || -> anyhow::Result<Pass> {
            let started = Instant::now();
            let mut pass = Pass::default();
            loop {
                let step_started = Instant::now();
                let generation = e.generation;
                let step = ring.step(e, sched, Duration::from_millis(4), 1)?;
                if step.absorbed > 0 {
                    pass.blocks.push(BlockNote {
                        generation,
                        chain: step.chain,
                        boundary: step.generations > 0,
                        idle: sched.devices.iter().map(|d| d.idle_seconds).sum(),
                    });
                    pass.breeding
                        .push((step_started, Instant::now(), step.generations > 0));
                }
                pass.step.absorbed += step.absorbed;
                pass.step.generations += step.generations;
                if pass.step.generations > 0
                    || stop.load(Ordering::Relaxed)
                    || started.elapsed() >= PASS_LIMIT
                {
                    return Ok(pass);
                }
            }
        },
        || match rx.recv_timeout(Duration::from_millis(1)) {
            Ok(Command::Ping(sent)) => pings.push((sent, Instant::now())),
            Ok(command) => {
                deferred.push(command);
                stop.store(true, Ordering::Relaxed);
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                // The UI is gone, so end the pass. `recv_timeout` now returns
                // at once, so sleep here instead of spinning.
                stop.store(true, Ordering::Relaxed);
                std::thread::sleep(Duration::from_millis(1));
            }
        },
    );
    if let Ok(pass) = &mut result {
        pass.pings = pings;
    }
    result
}
impl Loop {
    /// One step of evolution, when there is a game and engines and the game
    /// runs or the ring still has work in flight. An error stops the run and
    /// is shown to the player. A run without a game stops.
    pub(super) fn step(&mut self) {
        if self.exp.is_some() && self.gpu.sched.is_some() && (self.running || self.ring.active()) {
            if let Err(err) = self.evolve() {
                self.error = Some(format!("{err:#}"));
                self.running = false;
            }
            self.changed = true;
        } else if self.running && self.exp.is_none() {
            self.running = false;
        }
    }
    /// Moves a developer pause along (`dev_pause`). A pause holds new work
    /// back and closes the engines once they are idle. Its end reopens them.
    /// The start and the end go into the event log.
    pub(super) fn tick_dev_pause(&mut self) {
        if let Some(text) = self.dev.tick(self.gpu.sched.as_mut()) {
            let generation = self.exp.as_ref().map_or(0, |e| e.generation);
            log_event(&mut self.events, generation, EventKind::Gpu, text);
            self.changed = true;
        }
    }
    /// The scheduler's notices go into the event log. They tell the player
    /// that a GPU failed and was reopened, or could not be.
    pub(super) fn log_gpu_notices(&mut self) {
        if let Some(sched) = self.gpu.sched.as_mut() {
            for notice in sched.take_notices() {
                let generation = self.exp.as_ref().map_or(0, |e| e.generation);
                log_event(&mut self.events, generation, EventKind::Gpu, notice);
                self.changed = true;
            }
        }
    }
    /// A finished autosave goes into the event log, so the UI can say
    /// when the experiment was last saved.
    pub(super) fn log_autosave(&mut self) {
        if let Some((path, generation)) = self.autosave.finished() {
            log_event(
                &mut self.events,
                generation,
                EventKind::Saved,
                format!("Autosaved {}.", path.display()),
            );
            self.changed = true;
        }
    }
    /// A paused game only collects what the engines finished. A running game
    /// starts the ring if it is idle, runs a search pass and counts the
    /// seconds of the pass for the benchmark and the stage log. If the pass
    /// ended a generation, `generation_ended` follows.
    fn evolve(&mut self) -> anyhow::Result<()> {
        // `step` found the game and the engines.
        let (Some(e), Some(sched)) = (self.exp.as_mut(), self.gpu.sched.as_mut()) else {
            return Ok(());
        };
        if !self.running {
            // A pause absorbs no block, so no block is bred and no work goes
            // out. Work already on the engines still completes and waits in
            // the ring.
            self.ring.step(e, sched, Duration::from_millis(4), 0)?;
            return Ok(());
        }
        let pass_started = Instant::now();
        let world_before = e.config.clone();
        let kept_before = e.archive.entries.len();
        if !self.ring.active() {
            self.ring.start(e, sched);
        }
        sched.pump()?;
        // Blocks are absorbed in ring order, so the run does not
        // depend on which unit finished first, nor on when commands
        // are read.
        let pass = search_pass(
            &self.helper,
            e,
            sched,
            &mut self.ring,
            &self.rx,
            &mut self.deferred,
        )?;
        let step = pass.step;
        for note in &pass.blocks {
            self.ring_meter
                .add(note.generation, note.chain, note.boundary);
            if let Some(log) = &mut self.stage_log {
                log.block(note.idle);
            }
        }
        self.benchmark.pass_done(&pass.pings, &pass.breeding);
        let seconds = pass_started.elapsed().as_secs_f64();
        e.evaluation_seconds += seconds;
        // The experiment counted the archive and breeding seconds of the
        // pass. The rest of the pass counts as evaluation.
        let [archive, breeding] = std::mem::take(&mut e.stage_seconds);
        let evaluation = (seconds - archive - breeding).max(0.0);
        self.benchmark
            .add_stage_seconds([evaluation, archive, breeding]);
        if let Some(log) = &mut self.stage_log {
            log.add(0, evaluation);
            log.add(1, archive);
            log.add(2, breeding);
        }
        self.status = format!("Evolving · generation {}", e.generation);
        // A finished generation dump (`EVOLUTION_DUMP_GENERATION`, a developer
        // diagnostic) says where it went. The event log records it as `Saved`.
        if let Some(text) = e.dump_notice.take() {
            log_event(&mut self.events, e.generation, EventKind::Saved, text);
        }
        if step.generations == 0 {
            return Ok(());
        }
        self.generation_ended(&world_before, kept_before);
        Ok(())
    }
    /// A search pass ended a generation. `world_before` is the world when the
    /// pass began and `kept_before` is the number of elites in the global
    /// archive then.
    ///
    /// After a world change it retargets the ring and logs the change. Then it
    /// writes the stage log row, marks the end for the end-to-end rate, lets
    /// the benchmark count the generation and starts an autosave if one is
    /// due. It stops the run when the benchmark is over or a run of one
    /// generation is done.
    fn generation_ended(&mut self, world_before: &Config, kept_before: usize) {
        // `evolve` found the game and the engines.
        let (Some(e), Some(sched)) = (self.exp.as_mut(), self.gpu.sched.as_mut()) else {
            return;
        };
        if e.config.physics_differs(world_before) {
            // The boundary changed the world, by Autochange or by settings
            // that waited for it. Blocks not yet on an engine run in the new
            // world, and the kept elites are tested again in it.
            let lost = self.ring.world_changed(e, sched);
            if let Some(log) = &mut self.stage_log {
                log.discarded += lost;
            }
            log_world_change(
                &mut self.events,
                world_before,
                &e.config,
                e.generation,
                kept_before.min(e.config.population),
            );
        }
        if let Some(log) = &mut self.stage_log {
            log.write_generation(e, sched, &self.ring_meter);
        }
        note_generation(&mut self.generation_marks, e.config.population);
        if self.benchmark.generation_done(e, sched, &self.ctx) {
            self.running = false;
        }
        self.autosave.start_if_due(e);
        if self.run_until.is_some_and(|until| e.generation >= until) {
            self.running = false;
            self.status = format!("Paused after generation {}", e.generation - 1);
        }
    }
}
