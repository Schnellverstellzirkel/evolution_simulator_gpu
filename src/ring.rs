//! The ring of creatures in flight between the experiment and the engines.
//!
//! Every block of the experiment's ring is on its way through the engines
//! at all times. Engines finish blocks in whatever order the hardware
//! produces, but blocks are absorbed strictly in ring order: a block waits
//! until the blocks before it are absorbed, then it is decided against the
//! archives as they stand (`Experiment::verdict`), gets the confirmation
//! trials that decision asks for, is absorbed, and is bred again and queued
//! at the back. Breeding happens only at absorption and every block keeps the
//! trial settings it was bred with, so one seed gives one search whatever the
//! timing.

use crate::{
    config::Config,
    qd::EvaluationMetrics,
    scheduler::{self, Scheduler, Trial},
    storage::Experiment,
};
use anyhow::Result;
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc,
    time::Duration,
};

/// One block on its way.
struct Flight {
    seq: u64,
    /// The experiment's block index.
    block: usize,
    standard: Vec<Option<EvaluationMetrics>>,
    missing: usize,
    /// Confirmation trials by position in the block: `None` while running.
    confirms: HashMap<usize, Option<EvaluationMetrics>>,
}

/// What one `Ring::step` did.
#[derive(Clone, Copy, Debug, Default)]
pub struct Step {
    /// Blocks absorbed.
    pub absorbed: usize,
    /// Generations that ended.
    pub generations: usize,
}

#[derive(Default)]
pub struct Ring {
    flights: VecDeque<Flight>,
    next_seq: u64,
}

impl Ring {
    /// Whether blocks are in flight.
    pub fn active(&self) -> bool {
        !self.flights.is_empty()
    }

    /// Confirmation trials running now.
    pub fn confirming(&self) -> usize {
        self.flights
            .iter()
            .flat_map(|f| f.confirms.values())
            .filter(|c| c.is_none())
            .count()
    }

    /// Queues every block of `e` in ring order from its cursor. Work of an
    /// earlier ring is dropped. Blocks bred for a world that has changed
    /// since take the current settings first: none of them has run.
    pub fn start(&mut self, e: &mut Experiment, sched: &mut Scheduler) {
        self.stop(sched);
        let current = Arc::new(e.config.clone());
        let count = e.blocks.len();
        for i in 0..count {
            let k = (e.cursor + i) % count;
            e.retarget_block(k, &current);
            self.launch(e, sched, k);
        }
    }

    /// Drops every block in flight. Units on an engine finish there and
    /// their results are dropped.
    pub fn stop(&mut self, sched: &mut Scheduler) {
        sched.reset();
        self.flights.clear();
    }

    fn launch(&mut self, e: &Experiment, sched: &mut Scheduler, k: usize) {
        let seq = self.next_seq;
        self.next_seq += 1;
        let block = &e.blocks[k];
        sched.queue(
            seq << 1,
            Trial::Standard,
            Arc::clone(&block.population),
            None,
            Arc::clone(&block.config),
        );
        self.flights.push_back(Flight {
            seq,
            block: k,
            standard: vec![None; block.len()],
            missing: block.len(),
            confirms: HashMap::new(),
        });
    }

    fn flight(&mut self, seq: u64) -> Option<&mut Flight> {
        let front = self.flights.front()?.seq;
        self.flights.get_mut(seq.checked_sub(front)? as usize)
    }

    /// The world changed: blocks whose work has not reached an engine run in
    /// the new world. Blocks already run or running enter no archive when
    /// they are absorbed.
    pub fn world_changed(&mut self, e: &mut Experiment, sched: &mut Scheduler) {
        let current = Arc::new(e.config.clone());
        for tag in sched.retarget(&current) {
            if let Some(flight) = self.flight(tag >> 1) {
                let k = flight.block;
                e.retarget_block(k, &current);
            }
        }
    }

    /// Collects finished work, waiting up to `timeout` for some, then
    /// absorbs up to `absorb` finished blocks in ring order. With `absorb`
    /// zero it only collects, as while the game is paused.
    pub fn step(
        &mut self,
        e: &mut Experiment,
        sched: &mut Scheduler,
        timeout: Duration,
        absorb: usize,
    ) -> Result<Step> {
        let ready = self
            .flights
            .front()
            .is_some_and(|f| f.missing == 0 && f.confirms.values().all(Option::is_some));
        let wait = if ready && absorb > 0 {
            Duration::ZERO
        } else {
            timeout
        };
        for done in sched.collect(wait)? {
            let confirm = done.trial == Trial::Confirm;
            let Some(flight) = self.flight(done.tag >> 1) else {
                continue;
            };
            for (j, metric) in done.members.into_iter().zip(done.metrics) {
                if confirm {
                    flight.confirms.insert(j, Some(metric));
                } else if flight.standard[j].is_none() {
                    flight.standard[j] = Some(metric);
                    flight.missing -= 1;
                }
            }
        }
        let mut step = Step::default();
        while step.absorbed < absorb {
            let Some(front) = self.flights.front_mut() else {
                break;
            };
            if front.missing > 0 {
                break;
            }
            let standard: Vec<EvaluationMetrics> =
                front.standard.iter().map(|m| m.expect("result")).collect();
            let confirmed: HashMap<usize, EvaluationMetrics> = front
                .confirms
                .iter()
                .filter_map(|(&j, m)| m.map(|m| (j, m)))
                .collect();
            let k = front.block;
            match e.verdict(k, &standard, &confirmed) {
                crate::storage::Verdict::Confirm(need) => {
                    let new: Vec<usize> = need
                        .into_iter()
                        .filter(|j| !front.confirms.contains_key(j))
                        .collect();
                    if !new.is_empty() {
                        for &j in &new {
                            front.confirms.insert(j, None);
                        }
                        let block = &e.blocks[k];
                        let config: Arc<Config> =
                            Arc::new(scheduler::confirm_config(&block.config));
                        sched.queue(
                            front.seq << 1 | 1,
                            Trial::Confirm,
                            Arc::clone(&block.population),
                            Some(new),
                            config,
                        );
                    }
                    break;
                }
                crate::storage::Verdict::Final(finals) => {
                    if e.absorb(k, &finals)? {
                        step.generations += 1;
                    }
                    self.flights.pop_front();
                    self.launch(e, sched, k);
                    step.absorbed += 1;
                }
            }
        }
        if absorb > 0 {
            // Confirmations and bred blocks go out at once.
            sched.pump()?;
        }
        Ok(step)
    }
}
