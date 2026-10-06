//! One step of evolution: ring steps on a helper thread while the worker
//! keeps reading commands, and what a generation boundary asks for.

use super::{Command, EventKind, Loop, log_event, log_world_change, note_generation};
use crate::storage::Experiment;
use std::{
    sync::{
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Receiver},
    },
    time::{Duration, Instant},
};

/// Ring steps of one search pass, run on a helper thread.
#[derive(Default)]
struct Pass {
    step: crate::ring::Step,
    /// Send and read times of the pings read during the pass.
    pings: Vec<(Instant, Instant)>,
    /// Steps that absorbed a block: start, end, and whether it ended a
    /// generation.
    breeding: Vec<(Instant, Instant, bool)>,
    /// Each absorbed block: its generation, its host time (`ring::Step::chain`),
    /// whether it ended the generation, and the engines' idle seconds so far.
    blocks: Vec<BlockNote>,
}
/// What one absorbed block tells the ring meter and the stage log.
struct BlockNote {
    generation: u32,
    chain: f64,
    boundary: bool,
    idle: f64,
}
/// Longest search pass: snapshots are built between passes.
const PASS_LIMIT: Duration = Duration::from_millis(100);
/// Runs ring steps on `helper`, a thread on the pool's CPUs, until a
/// generation ends, a command arrives or `PASS_LIMIT` passes, while this
/// thread reads commands every millisecond. Absorbing and breeding a block
/// takes a few tenths of a second at 3M; the worker keeps reading commands
/// through it. Pings are answered at once. Other commands act on the
/// experiment, so they wait in `deferred`, in order, for the step in
/// progress to end.
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
                // The UI is gone: the loop ends after this pass.
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
    /// runs or the ring still has work in flight.
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
    /// A paused game only collects what the engines finished; a running one
    /// runs a search pass and, when it ended a generation, does what a
    /// generation boundary asks for.
    fn evolve(&mut self) -> anyhow::Result<()> {
        // `step` found the game and the engines.
        let (Some(e), Some(sched)) = (self.exp.as_mut(), self.gpu.sched.as_mut()) else {
            return Ok(());
        };
        if !self.running {
            // A pause stops new submissions and absorption; work on
            // the engines still completes and waits in the ring.
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
        // A developer's generation dump (EVOLUTION_DUMP_GENERATION)
        // says where it went.
        if let Some(text) = e.dump_notice.take() {
            log_event(&mut self.events, e.generation, EventKind::Saved, text);
        }
        if step.generations == 0 {
            return Ok(());
        }
        if e.config.physics_differs(&world_before) {
            // Autochange changed the world at the boundary: blocks
            // not yet on an engine run in the new world, and the
            // kept elites are tested again in it.
            let lost = self.ring.world_changed(e, sched);
            if let Some(log) = &mut self.stage_log {
                log.discarded += lost;
            }
            log_world_change(
                &mut self.events,
                &world_before,
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
        Ok(())
    }
}
