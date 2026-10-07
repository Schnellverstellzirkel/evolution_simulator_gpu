//! This module decides which creatures of a block get a confirmation trial,
//! and it sets the bar of the early screen. `Experiment::verdict` picks the
//! creatures that need a trial at the fine physics before an archive may take
//! them, and it gives each result its final score. `ScreenWindow` keeps the
//! newest distances at the screen, and `Experiment::next_screen` and
//! `Experiment::wild_bars` turn them into the bars of the blocks bred next.
//! The ring (`ring.rs`) calls `verdict`, and `Experiment::absorb` feeds the
//! windows.

use super::*;

/// Distances at the screen that the bar's window holds at least, when the
/// ring has them. The kept share of a quantile over this many varies by about
/// 0.3%, so a larger window only adds lag. A block of the game's ring holds
/// 196,608 creatures, so the bar of the evolved creatures comes from the
/// newest block alone.
const SCREEN_WINDOW_DISTANCES: usize = 16_384;

/// Distances at the screen of the newest absorbed blocks, one entry per
/// block, oldest first: as few blocks as hold `SCREEN_WINDOW_DISTANCES`, and
/// at most a ring. A bar is recomputed from them at every absorption. Each
/// entry also counts the results of the block that belong to other kinds of
/// creature and are not in the window. The `Experiment` keeps one window for
/// the evolved creatures, one for the nurseries' new bodies, one for the
/// nurseries' reshaped bodies and one for each wild island.
#[derive(Clone, Default)]
pub(super) struct ScreenWindow(VecDeque<(Vec<f32>, usize)>);

impl ScreenWindow {
    /// Adds one block's distances, and the number of results of other kinds
    /// it had, and drops the oldest blocks that are no longer needed. The
    /// newest blocks that hold enough distances for a steady quantile stay,
    /// at most `ring` of them (the number of blocks in the ring): older
    /// results lag the population more.
    pub(super) fn push(&mut self, distances: Vec<f32>, others: usize, ring: usize) {
        self.0.push_back((distances, others));
        let mut held: usize = self.0.iter().map(|(d, _)| d.len()).sum();
        while let Some(oldest) = self.0.front().map(|(d, _)| d.len())
            && (self.0.len() > ring || held - oldest >= SCREEN_WINDOW_DISTANCES)
        {
            self.0.pop_front();
            held -= oldest;
        }
    }
    /// Forgets every block, so the bar is unknown until new results come in.
    /// A world change does this (`Experiment::reset_search_context`).
    pub(super) fn clear(&mut self) {
        self.0.clear();
    }
    /// The bar that the best `keep` share of all the results of the window's
    /// blocks reached when the results of other kinds fall short of it. With
    /// no other kind it is the bar of the best `keep` share of the window.
    /// There is no bar, negative infinity, when the window holds fewer than
    /// 64 distances or the share to keep reaches 1 (`physics::screen_bar`).
    fn bar(&self, keep: f32) -> f32 {
        let own: usize = self.0.iter().map(|(d, _)| d.len()).sum();
        let others: usize = self.0.iter().map(|(_, o)| *o).sum();
        let share = if own == 0 {
            keep
        } else {
            keep * (own + others) as f32 / own as f32
        };
        crate::physics::screen_bar(self.0.iter().flat_map(|(d, _)| d.iter().copied()), share)
    }
}

/// Confirmation trials a block asks for per archive at once while it waits
/// for the ones it needs. Many record claims fail the fine trial, so asking
/// a few at a time chained round trips while the ring waited. Measured on
/// 2026-10-02 with the earlier kernel, at 3M per generation: generations 11
/// to 15 ran 127k to 196k creatures/s with 8, 196k to 330k with 64 and 308k
/// to 421k with 512. Without a limit an empty archive asks for nearly every
/// creature (1.07M trials in generation 0). `ConfirmHint` raises the limit
/// of an archive that stands on a plateau.
const SPECULATIVE_CONFIRMS: usize = 512;
/// An entrant gets its fine trial when it is at least this share of its
/// archive's best.
const ENTRANT_SHARE: f32 = 0.5;
/// Most confirmation trials one round asks for from the record-setters of one
/// archive, and from the entrants of a block together.
const MAX_CONFIRMS_PER_ROUND: usize = 16_384;

/// How many confirmation results the verdicts of the last blocks used, per
/// archive (decaying). A block that stands on a plateau, where every
/// candidate that ties the record fails its fine trial, needs all of them.
/// So the next block asks for `SPECULATIVE_CONFIRMS` plus twice that many in
/// its first round. With `SPECULATIVE_CONFIRMS` alone it would ask again,
/// round after round, at the front of the ring, where every round waits for
/// the GPU. The hint changes what is asked for and when, never what a verdict
/// decides: a verdict reads only the results of the candidates its own loop
/// reaches. It is not saved.
#[derive(Default)]
pub(super) struct ConfirmHint(std::sync::Mutex<Vec<usize>>);

impl Clone for ConfirmHint {
    fn clone(&self) -> Self {
        Self(std::sync::Mutex::new(
            self.0.lock().unwrap_or_else(|e| e.into_inner()).clone(),
        ))
    }
}

impl ConfirmHint {
    /// Trials a round asks for from archive `arena`: `SPECULATIVE_CONFIRMS`
    /// plus twice what the last blocks used, and at most
    /// `MAX_CONFIRMS_PER_ROUND`. A verdict asks for more when the results it
    /// has read show a plateau (`Experiment::verdict`).
    fn limit(&self, arena: usize) -> usize {
        let used = self
            .0
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(arena)
            .copied()
            .unwrap_or(0);
        (SPECULATIVE_CONFIRMS + 2 * used).min(MAX_CONFIRMS_PER_ROUND)
    }
    /// A block was decided: `used[arena]` results were read. Each archive's
    /// count becomes that number, or the old count less a quarter if that is
    /// more.
    fn learn(&self, used: &[usize]) {
        let mut hint = self.0.lock().unwrap_or_else(|e| e.into_inner());
        let len = used.len().max(hint.len());
        hint.resize(len, 0);
        for (kept, &now) in hint.iter_mut().zip(used) {
            *kept = now.max(*kept - *kept / 4);
        }
    }
}

impl Experiment {
    /// Whether the result `m` may enter an archive at all: its score is
    /// finite and not a failed trial, and it was neither screened nor
    /// excluded.
    fn eligible(m: &EvaluationMetrics) -> bool {
        m.fitness.is_finite() && m.fitness > FAILED && !m.screened && !m.excluded
    }
    /// The positions in `block` of the creatures that would take a cell: the
    /// best of each cell of an archive among the block's creatures, when it
    /// beats the elite there or the cell is empty, and when it is at least
    /// `ENTRANT_SHARE` of the archive's best. A creature competes in its own
    /// archive (an island or a nursery) and, when that is an island, in the
    /// global archive too. `results` are the block's results so far. Only
    /// eligible results count, and a wild island's creatures are not asked
    /// for.
    fn entrants(&self, block: &Block, results: &[EvaluationMetrics]) -> Vec<usize> {
        let population = &*block.population;
        let arenas = arena_count();
        let islands = &self.islands;
        let fallback = QdArchive::default();
        // A creature's position, its archive with its cell there, and its
        // cell in the global archive (only an island's creature has one).
        type Keyed = (usize, Option<(usize, qd::Niche)>, Option<qd::Niche>);
        let keyed: Vec<Keyed> = (0..results.len())
            .collect::<Vec<usize>>()
            .par_iter()
            .copied()
            .filter(|&j| Self::eligible(&results[j]))
            .filter(|&j| !qd::is_wild(qd::island_of_slot(block.first + j, island_count())))
            .map(|j| {
                let genome = &population.genomes[j];
                let nodes =
                    &population.nodes[genome.node_start..genome.node_start + genome.node_count];
                let muscles = &population.muscles
                    [genome.muscle_start..genome.muscle_start + genome.muscle_count];
                let descriptor = qd::descriptor(nodes, muscles, results[j].behavior);
                let arena = qd::arena_of_slot(block.first + j, arenas);
                let own = islands.get(arena).unwrap_or(&fallback).cell_of(descriptor);
                let global = (arena < island_count()).then(|| self.archive.cell_of(descriptor));
                (j, Some((arena, own)), global)
            })
            .collect();
        // Best creature of each cell: (fitness, position).
        let mut island_best: FastMap<(usize, qd::Niche), (f32, usize)> = FastMap::default();
        let mut global_best: FastMap<qd::Niche, (f32, usize)> = FastMap::default();
        for (j, own, global) in keyed {
            let fitness = results[j].fitness;
            let better = |held: &(f32, usize)| fitness > held.0;
            if let Some(key) = own {
                let held = island_best.entry(key).or_insert((fitness, j));
                if better(held) {
                    *held = (fitness, j);
                }
            }
            if let Some(key) = global {
                let held = global_best.entry(key).or_insert((fitness, j));
                if better(held) {
                    *held = (fitness, j);
                }
            }
        }
        // Whether `fitness` would take the cell `niche` of `archive`: it beats
        // the elite there, or the cell is free.
        let beats =
            |archive: &QdArchive, niche: &qd::Niche, fitness: f32| match archive.slot_for(niche) {
                Some(slot) => fitness > archive.entries[slot].fitness,
                None => archive.behavior_count() < archive.limit(),
            };
        // Only the fast ones matter: a creature that runs on a flaw of the
        // standard trial reaches the top of its archive, and a slow entrant
        // that runs on one stays slow.
        let global_bar = ENTRANT_SHARE * self.archive.best_fitness();
        let mut out: Vec<usize> = Vec::new();
        // `best_fitness` scans a whole archive: once per archive, not once
        // per cell of the block.
        let mut bars: FastMap<usize, f32> = FastMap::default();
        for ((arena, niche), (fitness, j)) in island_best {
            let bar = *bars.entry(arena).or_insert_with(|| {
                islands
                    .get(arena)
                    .map_or(0.0, |a| ENTRANT_SHARE * a.best_fitness())
            });
            if fitness >= bar && islands.get(arena).is_none_or(|a| beats(a, &niche, fitness)) {
                out.push(j);
            }
        }
        for (niche, (fitness, j)) in global_best {
            if fitness >= global_bar && beats(&self.archive, &niche, fitness) {
                out.push(j);
            }
        }
        // A creature can be the best of a cell in its own archive and in the
        // global archive.
        out.sort_unstable();
        out.dedup();
        out
    }
    /// Decides block `k`'s `standard` results against the archives as they
    /// stand now. A creature that would set or tie the record of its island
    /// (or nursery) needs a confirmation trial at the fine physics, and its
    /// score is the lower of the two. The record-setters of each archive are
    /// taken fastest first, each against the record the ones before it set,
    /// so no unconfirmed score becomes a record. A creature that would take
    /// a cell of an archive (`entrants`) needs a trial too. `confirmed` holds
    /// the confirmations that came back, by position. The result is
    /// `Verdict::Confirm` with the positions that still need a trial, and the
    /// caller runs them and calls `verdict` again with their results added to
    /// `confirmed`. Otherwise it is `Verdict::Final` with every result in
    /// block order. A block from a world that has since changed enters no
    /// archive.
    pub fn verdict(
        &self,
        k: usize,
        standard: &[EvaluationMetrics],
        confirmed: &HashMap<usize, EvaluationMetrics>,
    ) -> Verdict {
        let block = &self.blocks[k];
        // `out` starts as the standard results and ends as the final ones.
        let mut out = standard.to_vec();
        // A block from a world that has since changed enters no archive and
        // has no distance at the screen.
        if block.config.physics_differs(&self.config) {
            for m in &mut out {
                m.excluded = true;
                m.screen_x = f32::NAN;
            }
            return Verdict::Final(out);
        }
        let arenas = arena_count();
        // Each record once, and only for the archives a candidate is in:
        // `best_fitness` scans the whole archive, and the wild islands'
        // archives, which no candidate is in, hold most of the elites.
        let mut bars: Vec<Option<f32>> = vec![None; arenas];
        let mut bar = |arena: usize| {
            *bars[arena].get_or_insert_with(|| {
                self.islands
                    .get(arena)
                    .map_or(f32::NEG_INFINITY, QdArchive::best_fitness)
            })
        };
        // The positions that still need a confirmation trial.
        let mut need = Vec::new();
        Self::exclude_audit_below_bar(block, standard, &mut out);
        // Per archive, the positions that would set or tie its record.
        let mut candidates: Vec<Vec<usize>> = vec![Vec::new(); arenas];
        for (j, m) in standard.iter().enumerate() {
            let arena = qd::arena_of_slot(block.first + j, arenas);
            // A wild island's creatures take no confirmation trial: they ran
            // in a world of their own, and a trial runs in the block's world.
            let wild = qd::is_wild(qd::island_of_slot(block.first + j, island_count()));
            // A tie with the record is confirmed too: an integrator glitch
            // drives many bodies to one exact speed, so its ties are common,
            // and an unconfirmed tie never had to beat the fine trial.
            if !wild && Self::eligible(&out[j]) && m.fitness >= bar(arena) {
                candidates[arena].push(j);
            }
        }
        // `used[arena]` counts the results the loop of that archive reads,
        // for the hint. `applied` holds the positions whose confirmation is
        // in `out`: the entrants skip them, because applying one twice could
        // clear its `fine` flag.
        let mut used = vec![0usize; arenas];
        let mut applied: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for (arena, mut members) in candidates.into_iter().enumerate() {
            // Fastest first, and the lower position first among equals.
            members.sort_by(|&a, &b| {
                standard[b]
                    .fitness
                    .total_cmp(&standard[a].fitness)
                    .then(a.cmp(&b))
            });
            let mut record = bar(arena);
            let mut asked = 0;
            // Results this loop read, and whether one of them raised the
            // record.
            let mut read = 0;
            let mut raised = false;
            let speculative = self.confirm_hint.limit(arena);
            for j in members {
                if standard[j].fitness < record {
                    break;
                }
                let Some(check) = confirmed.get(&j) else {
                    need.push(j);
                    asked += 1;
                    // Results are in and none raised the record: the
                    // candidates left tie or beat a record that nothing
                    // reached, so they need their trial too. A round asks
                    // for four times as many as have failed so far, which
                    // takes a plateau of thousands in two or three rounds
                    // and wastes at most four times the failed prefix.
                    let limit = if read > 0 && !raised {
                        speculative.max(4 * read)
                    } else {
                        speculative
                    };
                    if asked >= limit.min(MAX_CONFIRMS_PER_ROUND) {
                        break;
                    }
                    continue;
                };
                read += 1;
                applied.insert(j);
                let m = &mut out[j];
                // The replay must show the trial the score came from.
                m.fine = check.fitness < m.fitness;
                m.fitness = m.fitness.min(check.fitness);
                // A confirmation that the screen stopped, or that has no finite
                // distance, leaves the creature out of every archive.
                m.excluded |= check.screened || !check.fitness.is_finite();
                if Self::eligible(m) {
                    raised |= m.fitness > record;
                    record = record.max(m.fitness);
                }
            }
            used[arena] = read;
        }
        // Every creature that would take a cell in an archive gets its fine
        // trial before it does, and its score is the lower of the two. A
        // standard trial alone is not enough for an entrant: evolution finds
        // the flaws of the standard trial. With the earlier kernel, nearly
        // every elite of every island in a game of 180 generations ran on a
        // flaw of that kernel's single substep and lost its distance at the
        // fine trial. Only the best creature of each cell of the block can
        // take it, so this asks for about as many trials as cells change.
        let entrants = self.entrants(block, &out);
        // The entrants whose trial has not come back.
        let mut rest: Vec<usize> = Vec::new();
        for j in entrants {
            if applied.contains(&j) {
                continue;
            }
            match confirmed.get(&j) {
                Some(check) => {
                    let m = &mut out[j];
                    m.fine = check.fitness < m.fitness;
                    m.fitness = m.fitness.min(check.fitness);
                    m.excluded |= check.screened || !check.fitness.is_finite();
                }
                None => rest.push(j),
            }
        }
        // Fastest first, so the limit of the round leaves out the slowest.
        rest.sort_by(|&a, &b| {
            standard[b]
                .fitness
                .total_cmp(&standard[a].fitness)
                .then(a.cmp(&b))
        });
        for (rank, &j) in rest.iter().enumerate() {
            if rank < MAX_CONFIRMS_PER_ROUND {
                need.push(j);
            } else {
                // Past the limit of the round. A later round asks for it:
                // `need` is not empty here, so the verdict is `Confirm`, this
                // result is not final and the mark is never seen.
                out[j].excluded = true;
            }
        }
        if need.is_empty() {
            // The block is decided: the hint learns how many results the
            // loop of each archive read.
            self.confirm_hint.learn(&used);
            Verdict::Final(out)
        } else {
            // A creature can be asked for twice, as a record-setter and as an
            // entrant.
            need.sort_unstable();
            need.dedup();
            Verdict::Confirm(need)
        }
    }
    /// An audit creature (`rungs::AUDIT`) runs without the 5 s screen so that
    /// its trial can calibrate the rungs. Its result is still held to the
    /// screen's rule: one that lived past 5 s below the bar enters no
    /// archive, as it would not have in a trial with the screen.
    fn exclude_audit_below_bar(
        block: &Block,
        standard: &[EvaluationMetrics],
        out: &mut [EvaluationMetrics],
    ) {
        let Some(screen) = block.config.screen else {
            return;
        };
        for (j, m) in standard.iter().enumerate() {
            let audit =
                block.population.flags.get(j).copied().unwrap_or(0) & crate::rungs::AUDIT != 0;
            let bar = block.screen_bar(&screen, j);
            if audit
                && bar.is_finite()
                && m.trace.steps() > crate::rungs::SCREEN_STEPS
                && m.screen_x < bar
            {
                out[j].excluded = true;
            }
        }
    }
    /// The early screen for the blocks bred from now on, or `None` when a
    /// trial of `duration` seconds is no longer than the screen time. Its bar
    /// is the distance at the screen that the best `physics::screen_keep()`
    /// share of the newest results reached (`screen_window`). The nurseries'
    /// new bodies and reshaped bodies have a bar of their own kind
    /// (`young_window`, `reshaped_window`). The bars move at every
    /// absorption, so a population that improves fast (the first
    /// generations, and every world change) keeps about its share instead
    /// of the 50 to 80% a generation-old bar kept. Each block carries the bar
    /// it was bred with, so the history depends on ring order only. With no
    /// distances yet (a new game, a load, a world change, which forgets the
    /// old world's distances) the blocks run every trial in full until the
    /// first block of results is in.
    pub(super) fn next_screen(&self, duration: f32) -> Option<crate::physics::Screen> {
        // A trial no longer than the screen time has nothing to screen.
        let seconds = crate::physics::screen_seconds().filter(|&s| s < duration)?;
        let keep = crate::physics::screen_keep();
        if self.dump_breeding() {
            // The generation dump runs every trial in full.
            return Some(crate::physics::Screen::uniform(seconds, f32::NEG_INFINITY));
        }
        Some(crate::physics::Screen {
            seconds,
            bar: self.screen_window.bar(keep),
            young_bar: self.young_window.bar(keep),
            reshaped_bar: self.reshaped_window.bar(keep),
        })
    }
    /// The screen bar of each wild island, in island order, for a block bred
    /// with `cfg`: the distance at the screen that the best
    /// `physics::screen_keep()` share of the island's newest evolved
    /// creatures reached in its own world. An island with too few distances
    /// gets negative infinity, which is no bar. The list is empty when `cfg`
    /// has no screen, or when the bar of its screen is off (negative
    /// infinity), so the wild creatures run in full. A wild island's new
    /// bodies are never screened.
    pub(super) fn wild_bars(&self, cfg: &Config) -> Vec<f32> {
        match cfg.screen {
            Some(screen) if screen.bar > f32::NEG_INFINITY => {
                let keep = crate::physics::screen_keep();
                self.wild_windows.iter().map(|w| w.bar(keep)).collect()
            }
            _ => Vec::new(),
        }
    }
}

#[cfg(test)]
mod confirm_tests {
    use super::*;

    #[test]
    fn the_confirmation_hint_follows_the_results_the_verdicts_used() {
        let hint = ConfirmHint::default();
        assert_eq!(hint.limit(3), SPECULATIVE_CONFIRMS);
        hint.learn(&[0, 0, 0, 800]);
        assert_eq!(hint.limit(3), SPECULATIVE_CONFIRMS + 1600);
        assert_eq!(hint.limit(0), SPECULATIVE_CONFIRMS);
        // It fades by a quarter with every block that used fewer.
        hint.learn(&[0, 0, 0, 0]);
        assert_eq!(hint.limit(3), SPECULATIVE_CONFIRMS + 2 * 600);
        hint.learn(&[0, 0, 0, 100_000]);
        assert_eq!(hint.limit(3), MAX_CONFIRMS_PER_ROUND);
    }
}
