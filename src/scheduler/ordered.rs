//! In-order absorption of evaluation results, so a run is repeatable.
//!
//! GPU units finish in whatever order the hardware and the host threads
//! happen to produce. Everything the search does with a result (deciding who
//! is worth a check trial, sharing one check per archive cell, archiving,
//! breeding, the screen bar) must not depend on that order. The scheduler
//! therefore works in blocks of creatures in a fixed sequence:
//!
//! - A block is released to the engines only after `window` earlier blocks
//!   have been returned, and takes the trial settings current at that moment.
//! - Its standard results are held until every member has one. Then the block
//!   is decided: members that cannot enter an archive are final, the best
//!   contender of each archive cell gets a check trial, and the other
//!   contenders for that cell stay out of every archive.
//! - Finished blocks are returned strictly in sequence, so the caller
//!   archives and breeds them in one fixed order.
//!
//! When `gated`, the archives change between blocks, so a block may only be
//! decided once the blocks up to `lag + 1` places before it are absorbed, and
//! a block is returned only after the block `lag` places behind it is decided.
//! Decisions then see one fixed archive state whatever the timing. The caller
//! must absorb every returned block before it pumps or collects again.

use super::*;
use std::collections::VecDeque;

#[derive(Clone)]
enum Slot {
    /// No standard result yet.
    Pending,
    /// The standard result waits for the block's decision.
    Standard(EvaluationMetrics),
    /// The check trial runs.
    Checking,
    Final(EvaluationMetrics),
}

struct Block {
    seq: u64,
    members: Vec<usize>,
    slots: Vec<Slot>,
    released: bool,
    /// Every member has been decided.
    decided: bool,
    pending: usize,
    standard: usize,
    checking: usize,
}

impl Block {
    fn complete(&self) -> bool {
        self.pending == 0 && self.standard == 0 && self.checking == 0
    }
}

#[derive(Default)]
pub(super) struct Ordered {
    blocks: VecDeque<Block>,
    /// Sequence number of the next block to create and of the next to return.
    created: u64,
    returned: u64,
    gated: bool,
    /// Blocks released ahead of the next one to return.
    window: u64,
    lag: u64,
    draining: bool,
    /// Creature index to its block's sequence number and slot.
    owner: HashMap<usize, (u64, usize)>,
    /// Archive cells whose check is claimed by a block that has not come back
    /// yet: the claiming block and the claimant's standard score. Later
    /// blocks check a contender for such a cell only if it scores higher.
    claimed: HashMap<u64, (u64, f32)>,
}

impl Ordered {
    fn block(&mut self, seq: u64) -> Option<&mut Block> {
        let front = self.blocks.front()?.seq;
        self.blocks.get_mut(seq.checked_sub(front)? as usize)
    }
}

impl Scheduler {
    /// Sets how blocks flow. `gated` is for runs whose archives change while
    /// blocks are in flight; `window` is how many blocks may be released
    /// ahead of the next one to return, and `lag` how many blocks behind a
    /// decision the archives may be (0 or 1, at most `window`).
    pub fn ordered_configure(&mut self, gated: bool, window: usize, lag: usize) {
        let o = &mut self.ordered;
        o.gated = gated;
        o.window = if gated {
            window.max(lag).max(1) as u64
        } else {
            u64::MAX
        };
        o.lag = if gated { lag as u64 } else { 0 };
        o.draining = false;
    }

    /// Drops every block and queued round. Only for an idle scheduler.
    pub fn ordered_reset(&mut self) {
        self.stop();
        self.draining = false;
        self.ordered.blocks.clear();
        self.ordered.owner.clear();
        self.ordered.claimed.clear();
        self.ordered.created = 0;
        self.ordered.returned = 0;
        self.ordered.draining = false;
        self.cfg_of.clear();
    }

    /// Appends one block of creatures, in the order given.
    pub fn add_block(&mut self, members: Vec<usize>) {
        if members.is_empty() {
            return;
        }
        let o = &mut self.ordered;
        let seq = o.created;
        o.created += 1;
        for (pos, &i) in members.iter().enumerate() {
            o.owner.insert(i, (seq, pos));
        }
        let n = members.len();
        o.blocks.push_back(Block {
            seq,
            slots: vec![Slot::Pending; n],
            members,
            released: false,
            decided: false,
            pending: n,
            standard: 0,
            checking: 0,
        });
    }

    /// Appends `indices` as blocks of at most `size`.
    pub fn add_blocks(&mut self, indices: impl IntoIterator<Item = usize>, size: usize) {
        let indices: Vec<usize> = indices.into_iter().collect();
        for chunk in indices.chunks(size.max(1)) {
            self.add_block(chunk.to_vec());
        }
    }

    /// Whether any block is still in flight or waiting to be returned.
    pub fn ordered_in_flight(&self) -> usize {
        self.in_flight() + self.ordered.blocks.len()
    }

    /// Stops handing out new work. Blocks that already hold results are
    /// decided, checked and returned without waiting for their turn.
    pub fn ordered_stop(&mut self) {
        self.stop();
        self.ordered.draining = true;
    }

    /// Releases blocks inside the window to the engines with `cfg`.
    fn release_blocks(&mut self, pop: &Population, cfg: &Config) {
        let limit = self.ordered.returned.saturating_add(self.ordered.window);
        let mut shared: Option<Arc<Config>> = None;
        let mut release = Vec::new();
        for block in &mut self.ordered.blocks {
            if block.seq > limit {
                break;
            }
            if !block.released {
                block.released = true;
                release.push(block.members.clone());
            }
        }
        for members in release {
            let cfg = shared.get_or_insert_with(|| Arc::new(cfg.clone())).clone();
            self.extend_with(pop, members, cfg);
        }
    }

    /// Decides every block whose turn has come.
    fn decide_blocks(&mut self, mut need: impl FnMut(usize, &EvaluationMetrics) -> CheckNeed) {
        let draining = self.ordered.draining;
        let (gated, lag, returned) = (self.ordered.gated, self.ordered.lag, self.ordered.returned);
        let mut sends: Vec<(usize, EvaluationMetrics)> = Vec::new();
        let mut claimed = std::mem::take(&mut self.ordered.claimed);
        let mut dropped = 0u64;
        let mut released = 0u64;
        for block in &mut self.ordered.blocks {
            if block.decided {
                continue;
            }
            let ready = block.released && (block.pending == 0 || draining);
            let turn = !gated || draining || block.seq <= returned + lag;
            if !ready || !turn {
                if !draining {
                    break;
                }
                continue;
            }
            // Members in block order; the best contender of a cell (the
            // first one on a tie) gets its check.
            let mut needs: Vec<(usize, CheckNeed)> = Vec::new();
            let mut champion: HashMap<u64, (f32, usize)> = HashMap::new();
            for (pos, slot) in block.slots.iter().enumerate() {
                let Slot::Standard(metric) = slot else {
                    continue;
                };
                let mut n = need(block.members[pos], metric);
                if let CheckNeed::Check { cell: Some(cell) } = n
                    && !draining
                {
                    // A cell claimed by an earlier block waits for that
                    // check: only a better contender is worth another.
                    if claimed
                        .get(&cell)
                        .is_some_and(|&(_, bar)| metric.fitness <= bar)
                    {
                        n = CheckNeed::Check {
                            cell: Some(u64::MAX),
                        };
                    } else {
                        let entry = champion.entry(cell).or_insert((metric.fitness, pos));
                        if metric.fitness > entry.0 {
                            *entry = (metric.fitness, pos);
                        }
                    }
                }
                needs.push((pos, n));
            }
            for (&cell, &(fitness, _)) in &champion {
                claimed.insert(cell, (block.seq, fitness));
            }
            for (pos, n) in needs {
                let Slot::Standard(mut metric) =
                    std::mem::replace(&mut block.slots[pos], Slot::Pending)
                else {
                    unreachable!("decided slots hold a standard result");
                };
                block.standard -= 1;
                match n {
                    CheckNeed::Release => {
                        released += 1;
                        block.slots[pos] = Slot::Final(metric);
                    }
                    CheckNeed::Check { cell: Some(cell) }
                        if !draining && (cell == u64::MAX || champion[&cell].1 != pos) =>
                    {
                        dropped += 1;
                        metric.unchecked = true;
                        block.slots[pos] = Slot::Final(metric);
                    }
                    CheckNeed::Check { .. } => {
                        block.slots[pos] = Slot::Checking;
                        block.checking += 1;
                        sends.push((block.members[pos], metric));
                    }
                }
            }
            if block.pending == 0 {
                block.decided = true;
            }
        }
        self.ordered.claimed = claimed;
        self.checks_released += released;
        self.checks_dropped += dropped;
        for (i, metric) in sends {
            self.submit_check(i, metric);
        }
    }

    /// Releases and decides blocks, then queues checks and standard work.
    pub fn ordered_pump(
        &mut self,
        pop: &Population,
        cfg: &Config,
        need: impl FnMut(usize, &EvaluationMetrics) -> CheckNeed,
    ) -> Result<()> {
        self.release_blocks(pop, cfg);
        self.decide_blocks(need);
        self.pump(pop, cfg, &[], |_, _| CheckNeed::Check { cell: None })
    }

    /// Queues standard work only, for callers that cannot decide contenders
    /// at the moment (while breeding).
    pub fn ordered_pump_standard(&mut self, pop: &Population, cfg: &Config) -> Result<()> {
        self.release_blocks(pop, cfg);
        self.pump_standard(pop, cfg, &[])
    }

    /// Decides blocks and queues their checks, without new standard work.
    pub fn ordered_pump_checks(
        &mut self,
        pop: &Population,
        cfg: &Config,
        need: impl FnMut(usize, &EvaluationMetrics) -> CheckNeed,
    ) -> Result<()> {
        self.decide_blocks(need);
        self.pump_checks(pop, cfg, |_, _| CheckNeed::Check { cell: None })
    }

    /// Returns finished blocks in sequence as (creatures, final metrics),
    /// waiting up to `timeout` when none is ready. At most `limit` blocks.
    pub fn ordered_collect(
        &mut self,
        pop: &Population,
        cfg: &Config,
        timeout: Duration,
        limit: usize,
    ) -> Result<Vec<(Vec<usize>, Vec<EvaluationMetrics>)>> {
        let mut out = self.take_ordered(limit);
        if !out.is_empty() {
            return Ok(out);
        }
        let deadline = Instant::now() + timeout;
        loop {
            let units = self.collect(pop, cfg, timeout.min(Duration::from_millis(2)), |_, _| {
                false
            })?;
            for (indices, metrics) in units {
                for (i, metric) in indices.into_iter().zip(metrics) {
                    let Some(&(seq, pos)) = self.ordered.owner.get(&i) else {
                        continue;
                    };
                    let Some(block) = self.ordered.block(seq) else {
                        continue;
                    };
                    match block.slots[pos] {
                        Slot::Pending => {
                            block.slots[pos] = Slot::Standard(metric);
                            block.pending -= 1;
                            block.standard += 1;
                        }
                        Slot::Checking => {
                            block.slots[pos] = Slot::Final(metric);
                            block.checking -= 1;
                        }
                        _ => {}
                    }
                }
            }
            out = self.take_ordered(limit);
            if !out.is_empty() || Instant::now() >= deadline || self.in_flight() == 0 {
                return Ok(out);
            }
        }
    }

    /// Finished blocks at the front, in sequence.
    fn take_ordered(&mut self, limit: usize) -> Vec<(Vec<usize>, Vec<EvaluationMetrics>)> {
        let mut out = Vec::new();
        while out.len() < limit {
            let draining = self.ordered.draining;
            let (gated, lag) = (self.ordered.gated, self.ordered.lag);
            let Some(front) = self.ordered.blocks.front() else {
                break;
            };
            let idle = self.in_flight() == 0 && self.round.is_none();
            let flush = draining && idle && self.ordered.blocks.iter().all(|b| b.standard == 0);
            if !front.complete() && !flush {
                break;
            }
            // The next block's decision comes before this block's absorption.
            if gated && !draining && lag > 0 && front.complete() {
                let next = self.ordered.blocks.get(lag as usize);
                if next.is_some_and(|b| !b.decided) {
                    break;
                }
            }
            let block = self.ordered.blocks.pop_front().expect("front block");
            self.ordered.returned = block.seq + 1;
            // Without gating the archives hold still for the whole round, so
            // a claim stands until then; retiring it on return would make
            // decisions depend on when blocks come back.
            if gated {
                self.ordered
                    .claimed
                    .retain(|_, &mut (seq, _)| seq != block.seq);
            }
            let mut members = Vec::with_capacity(block.members.len());
            let mut metrics = Vec::with_capacity(block.members.len());
            for (i, slot) in block.members.iter().zip(block.slots) {
                self.ordered.owner.remove(i);
                self.cfg_of.remove(i);
                if let Slot::Final(metric) = slot {
                    members.push(*i);
                    metrics.push(metric);
                }
            }
            if !members.is_empty() {
                out.push((members, metrics));
            }
        }
        out
    }
}
