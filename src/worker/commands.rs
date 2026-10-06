//! Commands from the UI: `pass` reads them, `handle` runs each one.

use super::{Command, EventKind, LineageStep, Loop, log_event, log_world_change};
use crate::{
    config::Config,
    storage::{self, Experiment},
};
use std::{
    ops::ControlFlow,
    sync::{Arc, atomic::Ordering, mpsc},
    time::{Duration, Instant},
};

impl Loop {
    /// The next command. While the game is evolving or the engines are busy
    /// it only looks; otherwise it waits up to 100 ms for one. `Break` when
    /// the UI is gone.
    pub(super) fn next_command(&mut self) -> ControlFlow<(), Option<Command>> {
        // A developer pause with the engines closed leaves nothing to
        // collect: wait for commands instead of spinning.
        if (self.running || self.gpu.on_engines() > 0) && !self.dev.idle() {
            return ControlFlow::Continue(self.rx.try_recv().ok());
        }
        match self.rx.recv_timeout(Duration::from_millis(100)) {
            Ok(c) => ControlFlow::Continue(Some(c)),
            Err(mpsc::RecvTimeoutError::Disconnected) => ControlFlow::Break(()),
            Err(_) => ControlFlow::Continue(None),
        }
    }
    /// Handles every queued command now; controls never wait behind a GPU
    /// batch. `first` is the command `next_command` read. `Break` on
    /// `Command::Shutdown`.
    pub(super) fn handle_commands(&mut self, first: Option<Command>) -> ControlFlow<()> {
        // Commands held back during a load run first once it is done.
        let mut commands: Vec<Command> = if self.loading.is_none() {
            std::mem::take(&mut self.deferred)
        } else {
            Vec::new()
        };
        commands.extend(first);
        commands.extend(self.rx.try_iter());
        for command in commands {
            if matches!(command, Command::Shutdown) {
                return ControlFlow::Break(());
            }
            // While a save loads there is no game to act on: hold every other
            // command, in order, until the load is done.
            if self.loading.is_some()
                && !matches!(
                    command,
                    Command::New(_) | Command::Load(_) | Command::Ping(_)
                )
            {
                self.deferred.push(command);
                continue;
            }
            // No command waits for the engines: a save holds only the
            // archives, settings apply to the blocks bred after them, and a
            // new or loaded game drops the ring.
            if matches!(command, Command::New(_) | Command::Load(_)) {
                self.autosave.join();
            }
            self.changed = true;
            self.error = None;
            if let Err(e) = self.handle(command) {
                self.error = Some(format!("{e:#}"));
                self.running = false;
            }
            // Disconnection is handled separately; shutdown closes the receiver after this iteration.
        }
        ControlFlow::Continue(())
    }
    /// Runs one command. An error is shown to the player and stops the run.
    fn handle(&mut self, command: Command) -> anyhow::Result<()> {
        match command {
            Command::Shutdown => {}
            Command::New(cfg) => self.new_game(cfg)?,
            Command::Run { continuous, guided } => self.start_run(continuous, guided),
            Command::Pause => {
                self.running = false;
                self.status = "Paused".into();
            }
            Command::Configure(cfg) => self.configure(cfg)?,
            Command::ConfigureProbe(sent) => self.configure_probe(sent)?,
            Command::Meteor => self.meteor(),
            Command::Extinction => self.extinction(),
            Command::UndoMeteor => self.undo_meteor(),
            Command::Save(path) => self.request_save(path),
            Command::Load(path) => self.start_load(path)?,
            Command::Export(path) => self.export(path)?,
            Command::Ping(sent) => {
                self.benchmark.ping(sent);
                self.changed = false;
            }
            Command::Cards => self.send_cards = true,
            Command::MapTable(on) => self.show_map_table(on),
            Command::Select(id) => self.select(id),
            Command::Lineage(id) => self.trace_lineage(id),
        }
        Ok(())
    }
    fn new_game(&mut self, cfg: Config) -> anyhow::Result<()> {
        self.cancel_load();
        self.running = false;
        if let Some(sched) = self.gpu.sched.as_mut() {
            self.ring.stop(sched);
        }
        self.status = "Creating population…".into();
        // The ring is sized once, for the whole game, from
        // what this session measured.
        let shape = storage::RingShape::size(&self.ring_meter.times(self.gpu.sched.as_ref()));
        let next = Experiment::with_ring(cfg, shape)?;
        self.ring_meter.clear();
        self.preview = Some((next.blocks[0].population.creature(0), next.config.clone()));
        self.events = Arc::new(Vec::new());
        log_event(
            &mut self.events,
            next.generation,
            EventKind::Started,
            format!(
                "New experiment: {} creatures, seed {}",
                next.config.population, next.config.seed
            ),
        );
        self.exp = Some(next);
        self.epoch += 1;
        self.history = Arc::new(Vec::new());
        self.status = "Population ready".into();
        Ok(())
    }
    /// `Command::Run`: `continuous` runs on, a guided run or one that is not
    /// continuous stops after one generation.
    fn start_run(&mut self, continuous: bool, guided: bool) {
        self.benchmark
            .run_started(self.exp.as_ref().map(|e| e.generation));
        // A run of one generation stops at its end.
        self.run_until = self
            .exp
            .as_ref()
            .filter(|_| !continuous || guided)
            .map(|e| e.generation + 1);
        self.pause.store(false, Ordering::Relaxed);
        self.running = true;
        if let Some(log) = &mut self.stage_log {
            log.reset();
        }
    }
    fn configure(&mut self, cfg: Config) -> anyhow::Result<()> {
        if let Some(e) = &mut self.exp {
            let before = e.config.clone();
            // The change applies now. Blocks already run or
            // running in the old world enter no archive;
            // blocks not yet on an engine run in the new one.
            e.update_config_now(cfg)?;
            if before.physics_differs(&e.config)
                && let Some(sched) = self.gpu.sched.as_mut()
            {
                let lost = self.ring.world_changed(e, sched);
                if let Some(log) = &mut self.stage_log {
                    log.discarded += lost;
                }
            }
            log_world_change(
                &mut self.events,
                &before,
                &e.config,
                e.generation,
                e.reseed.len(),
            );
            self.status = if e.pending.is_some() {
                "The world changes when the next generation starts".into()
            } else {
                "Settings applied".into()
            };
        }
        Ok(())
    }
    /// Benchmark probe: applies the current settings again.
    fn configure_probe(&mut self, sent: Instant) -> anyhow::Result<()> {
        if let Some(e) = &mut self.exp {
            let cfg = e.config.clone();
            e.update_config_now(cfg)?;
            self.benchmark.configure_probe(sent);
        }
        Ok(())
    }
    fn meteor(&mut self) {
        if let Some(e) = &mut self.exp {
            let lost = e.meteor(0.5);
            self.status = format!("A meteor wiped out {lost} creatures");
            log_event(
                &mut self.events,
                e.generation,
                EventKind::Catastrophe,
                format!("Meteor strike: {lost} kept creatures wiped out."),
            );
        }
    }
    fn extinction(&mut self) {
        if let Some(e) = &mut self.exp {
            let lost = e.extinction();
            self.status = format!("The slowest group lost all {lost} creatures");
            log_event(
                &mut self.events,
                e.generation,
                EventKind::Catastrophe,
                format!("Extinction: the slowest group lost all {lost} creatures."),
            );
        }
    }
    fn undo_meteor(&mut self) {
        if let Some(e) = &mut self.exp {
            let back = e.undo_meteor();
            self.status = format!("{back} creatures came back");
            log_event(
                &mut self.events,
                e.generation,
                EventKind::Undo,
                format!("Undo: {back} creatures came back."),
            );
        }
    }
    /// `Command::MapTable`: the UI starts or stops asking for the archive
    /// map table.
    fn show_map_table(&mut self, on: bool) {
        self.want_map = on;
        if !on {
            self.map = None;
            self.map_key = (u64::MAX, usize::MAX, 0);
        }
    }
    /// `Command::Select`: the archive creature with this id goes to the next
    /// snapshot.
    fn select(&mut self, id: u64) {
        if let Some(e) = &self.exp
            && let Some(elite) = e
                .archive
                .entries
                .iter()
                .find(|elite| elite.creature.id == id)
        {
            self.selected = Some(elite.replay_of(&e.config));
        }
    }
    /// `Command::Lineage`: the ancestors of the creature with this id go to
    /// the next snapshot.
    fn trace_lineage(&mut self, id: u64) {
        if let Some(e) = &self.exp {
            let chain = e.ancestry(id, storage::ANCESTRY_DEPTH);
            self.lineage = Some((
                id,
                chain
                    .iter()
                    .enumerate()
                    .map(|(k, a)| LineageStep {
                        generation: a.generation,
                        fitness: a.fitness,
                        gain: chain
                            .get(k + 1)
                            .map_or(0.0, |parent| a.fitness - parent.fitness),
                        change: a.change.clone(),
                        creature: a.creature.unpack(),
                    })
                    .collect(),
            ));
        }
    }
}
