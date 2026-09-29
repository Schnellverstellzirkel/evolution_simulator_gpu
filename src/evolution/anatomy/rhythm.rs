//! Operators that change joint ranges, timing patterns and mass together.
use super::{Context, branch, child_bones, degree, is_neck, muscles_on, parent_bones};
use crate::config::Config;
use crate::evolution::{
    Bone, Creature, JOINT_LIMIT, MAX_ORGAN_MASS, MIN_ORGAN_MASS, Rng, max_bone_length,
    organ_center, organ_range,
};
use crate::qd::gaussian;

/// Narrows one joint's range and widens a neighbouring joint's range by the
/// same angle (within `JOINT_LIMIT`): flexibility moves along the limb.
pub(crate) fn redistribute_joint_flex(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let width = |b: &Bone| b.max_angle - b.min_angle;
    // A bone and the bone above or below it, as (narrowed, widened), with the
    // most angle that can move. The neck's joint is free, so it takes no part.
    let mut pairs = Vec::new();
    for from in 0..c.bones.len() {
        for to in 0..c.bones.len() {
            let (x, y) = (c.bones[from], c.bones[to]);
            let most = width(&x).min(2.0 * JOINT_LIMIT - width(&y));
            if (x.b == y.a || y.b == x.a) && !is_neck(c, from) && !is_neck(c, to) && most > 0.01 {
                pairs.push((from, to, most));
            }
        }
    }
    if pairs.is_empty() {
        return false;
    }
    let (from, to, most) = pairs[rng.index(pairs.len())];
    let angle = most * rng.range(0.2, 0.8);
    // The narrowed joint keeps the ratio of its two sides. The widened joint
    // grows each side in proportion to its room up to `JOINT_LIMIT`.
    let x = &mut c.bones[from];
    let scale = 1.0 - angle / width(x);
    x.min_angle *= scale;
    x.max_angle *= scale;
    let y = &mut c.bones[to];
    let (below, above) = (JOINT_LIMIT + y.min_angle, JOINT_LIMIT - y.max_angle);
    let share = angle / (below + above);
    y.min_angle -= below * share;
    y.max_angle += above * share;
    true
}

/// Finds two branches of the same shape (same bone count, similar lengths)
/// and applies one random change to both: bone lengths, joint ranges or
/// anchors. Their timing difference stays.
pub(crate) fn mutate_matching_limbs(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let pairs = matching_limbs(c);
    if pairs.is_empty() {
        return false;
    }
    let (x, y) = &pairs[rng.index(pairs.len())];
    let at = rng.index(x.len());
    let (p, q) = (x[at], y[at]);
    let mut changed = false;
    match rng.index(3) {
        0 => {
            let scale = (0.15 * gaussian(rng)).exp();
            for b in [p, q] {
                let bone = &mut c.bones[b];
                let length = (bone.rest_length * scale).clamp(0.03, max_bone_length());
                changed |= length != bone.rest_length;
                bone.rest_length = length;
            }
        }
        1 => {
            let (low, high) = (0.15 * gaussian(rng), 0.15 * gaussian(rng));
            for b in [p, q] {
                let bone = &mut c.bones[b];
                let old = *bone;
                bone.min_angle += low;
                bone.max_angle += high;
                bone.clamp_range();
                changed |= *bone != old;
            }
        }
        _ => {
            let shift = 0.1 * gaussian(rng);
            for m in &mut c.muscles {
                for (bone, anchor) in [(m.bone_a, &mut m.anchor_a), (m.bone_b, &mut m.anchor_b)] {
                    if bone as usize == p || bone as usize == q {
                        let moved = (*anchor + shift).clamp(0.0, 1.0);
                        changed |= moved != *anchor;
                        *anchor = moved;
                    }
                }
            }
        }
    }
    changed
}

/// Pairs of branches of the same shape: apart from each other, with the same
/// bone count and bone lengths within 25% of each other, position by position.
pub(super) fn matching_limbs(c: &Creature) -> Vec<(Vec<usize>, Vec<usize>)> {
    let limbs: Vec<Vec<usize>> = (0..c.bones.len())
        .filter(|&b| !is_neck(c, b))
        .map(|b| branch(c, b))
        .collect();
    let similar = |(&p, &q): (&usize, &usize)| {
        let (a, b) = (c.bones[p].rest_length, c.bones[q].rest_length);
        a.max(b) <= 1.25 * a.min(b)
    };
    let mut pairs = Vec::new();
    for (i, x) in limbs.iter().enumerate() {
        for y in &limbs[i + 1..] {
            if x.len() == y.len()
                && !x.contains(&y[0])
                && !y.contains(&x[0])
                && x.iter().zip(y).all(similar)
            {
                pairs.push((x.clone(), y.clone()));
            }
        }
    }
    pairs
}

/// Along a chain of bones (a path down one branch), sets the phase of the
/// muscles on successive bones to grow by one step per bone: a contraction
/// wave, for curling and crawling.
pub(crate) fn chain_phase_wave(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(chain) = chain_below(c, rng) else {
        return false;
    };
    let step = rng.range(0.05, 0.25) * if rng.unit() < 0.5 { -1.0 } else { 1.0 };
    phase_wave(c, &chain, step)
}

/// A path of at least two bones from a random bone (not the neck) down to a
/// foot, taking a random child bone at each junction.
fn chain_below(c: &Creature, rng: &mut Rng) -> Option<Vec<usize>> {
    let children = child_bones(c);
    let starts: Vec<usize> = (0..c.bones.len())
        .filter(|&b| !is_neck(c, b) && !children[c.bones[b].b as usize].is_empty())
        .collect();
    if starts.is_empty() {
        return None;
    }
    let mut chain = vec![starts[rng.index(starts.len())]];
    loop {
        let below = &children[c.bones[chain[chain.len() - 1]].b as usize];
        if below.is_empty() {
            return Some(chain);
        }
        chain.push(below[rng.index(below.len())]);
    }
}

/// Sets every muscle on `chain[i]` to the phase of the chain's first muscle
/// plus `i` steps. A muscle on two chain bones counts for the upper one.
fn phase_wave(c: &mut Creature, chain: &[usize], step: f32) -> bool {
    let parts: Vec<Vec<usize>> = chain.iter().map(|&b| vec![b]).collect();
    let groups = muscle_groups(c, &parts);
    let Some(&first) = groups.iter().flatten().next() else {
        return false;
    };
    let start = c.muscles[first].phase;
    for (i, group) in groups.iter().enumerate() {
        for &m in group {
            c.muscles[m].phase = (start + i as f32 * step).rem_euclid(1.0);
        }
    }
    true
}

/// Shifts whole limbs (the branches at one junction, or every leaf branch) to
/// a pattern of phase offsets: all together, alternating halves, or evenly
/// staggered, keeping the timing inside each limb.
pub(crate) fn limb_phase_pattern(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let limbs = if rng.unit() < 0.5 {
        let children = child_bones(c);
        let junctions: Vec<&Vec<usize>> = children.iter().filter(|list| list.len() > 1).collect();
        if junctions.is_empty() {
            return false;
        }
        junctions[rng.index(junctions.len())]
            .iter()
            .map(|&b| branch(c, b))
            .collect()
    } else {
        leaf_limbs(c)
    };
    limbs.len() > 1 && shift_limbs(c, &limbs, rng.index(3))
}

/// Every limb that ends in a foot: the bones from a leaf node up to the node
/// where the body branches, or up to the neck.
pub(super) fn leaf_limbs(c: &Creature) -> Vec<Vec<usize>> {
    let parents = parent_bones(c);
    let children = child_bones(c);
    (1..c.nodes.len())
        .filter(|&n| children[n].is_empty())
        .filter_map(|n| {
            let mut root = parents[n].filter(|&b| !is_neck(c, b))?;
            loop {
                let top = c.bones[root].a as usize;
                match parents[top] {
                    Some(above) if !is_neck(c, above) && children[top].len() == 1 => root = above,
                    _ => return Some(branch(c, root)),
                }
            }
        })
        .collect()
}

/// Shifts the muscles of each limb together so that the first muscle of limb
/// `i` sits at the first limb's phase plus the pattern's offset for `i`:
/// 0 all together, 1 alternating halves, 2 evenly staggered.
fn shift_limbs(c: &mut Creature, limbs: &[Vec<usize>], pattern: usize) -> bool {
    let groups = muscle_groups(c, limbs);
    let Some(&first) = groups.iter().flatten().next() else {
        return false;
    };
    let start = c.muscles[first].phase;
    for (i, group) in groups.iter().enumerate() {
        let Some(&lead) = group.first() else {
            continue;
        };
        let offset = match pattern {
            0 => 0.0,
            1 => 0.5 * (i % 2) as f32,
            _ => i as f32 / limbs.len() as f32,
        };
        let shift = start + offset - c.muscles[lead].phase;
        for &m in group {
            c.muscles[m].phase = (c.muscles[m].phase + shift).rem_euclid(1.0);
        }
    }
    true
}

/// The muscles with an end on each part (a list of bones). A muscle on two
/// parts belongs to the first.
fn muscle_groups(c: &Creature, parts: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut taken = vec![false; c.muscles.len()];
    parts
        .iter()
        .map(|bones| {
            muscles_on(c, bones, false)
                .into_iter()
                .filter(|&m| !std::mem::replace(&mut taken[m], true))
                .collect()
        })
        .collect()
}

/// Changes the duty of every muscle in a limb by one amount and moves their
/// phases so each contraction keeps its middle: a slower push with a quicker
/// return, or the reverse.
pub(crate) fn limb_duty_cycle(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let roots: Vec<usize> = (0..c.bones.len()).filter(|&b| !is_neck(c, b)).collect();
    if roots.is_empty() {
        return false;
    }
    let limb = branch(c, roots[rng.index(roots.len())]);
    let amount = rng.range(0.03, 0.2) * if rng.unit() < 0.5 { -1.0 } else { 1.0 };
    let mut changed = false;
    // A muscle contracts while its cycle position is below `duty`, so the
    // middle of the contraction comes at cycle position `duty / 2`.
    for i in muscles_on(c, &limb, false) {
        let m = &mut c.muscles[i];
        let duty = (m.duty + amount).clamp(0.05, 0.95);
        m.phase = (m.phase + 0.5 * (duty - m.duty)).rem_euclid(1.0);
        changed |= duty != m.duty;
        m.duty = duty;
    }
    changed
}

/// Picks a foot (a leaf node) and makes every muscle on a bone at that foot
/// sense it, with reset phases that keep their current order, so landing
/// restarts the limb's movement as a whole.
pub(crate) fn touchdown_package(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let parents = parent_bones(c);
    let feet: Vec<(usize, Vec<usize>)> = (1..c.nodes.len())
        .filter(|&n| degree(c, n) == 1)
        .filter_map(|n| Some((n, muscles_on(c, &[parents[n]?], false))))
        .filter(|(_, muscles)| !muscles.is_empty())
        .collect();
    if feet.is_empty() {
        return false;
    }
    let (foot, muscles) = &feet[rng.index(feet.len())];
    // The resets keep the phase differences, so after a landing the muscles
    // run in the same order and spacing as before.
    let reset = rng.unit();
    let first = c.muscles[muscles[0]].phase;
    for &i in muscles {
        let m = c.muscles[i];
        let (a, b) = (c.bones[m.bone_a as usize], c.bones[m.bone_b as usize]);
        let sensor = [a.a, a.b, b.a, b.b]
            .iter()
            .position(|&n| n as usize == *foot)
            .expect("the muscle is on the foot's bone");
        let m = &mut c.muscles[i];
        m.sensor = sensor as u32;
        m.reset = (reset + m.phase - first).rem_euclid(1.0);
    }
    true
}

/// Moves part of one organ's mass to another bone (creating an organ there
/// if needed), keeping the total organ mass and the organ limits.
pub(crate) fn redistribute_organ_mass(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let organs: Vec<usize> = (0..c.bones.len())
        .filter(|&b| c.bones[b].organ_mass > 0.0)
        .collect();
    if organs.is_empty() {
        return false;
    }
    let from = organs[rng.index(organs.len())];
    // The source keeps at least the lightest organ, and a new organ starts
    // with at least that much. Each target comes with the least and most mass
    // it can take.
    let spare = c.bones[from].organ_mass - MIN_ORGAN_MASS;
    let center = organ_center(&c.nodes);
    let targets: Vec<(usize, f32, f32)> = (0..c.bones.len())
        .filter(|&b| b != from)
        .filter_map(|b| {
            let bone = &c.bones[b];
            let new = bone.organ_mass <= 0.0;
            if new && organ_range(bone, &c.nodes, center).is_none() {
                return None;
            }
            let least = if new { MIN_ORGAN_MASS } else { 0.0 };
            let most = spare.min(MAX_ORGAN_MASS - bone.organ_mass);
            (most > least + 0.001).then_some((b, least, most))
        })
        .collect();
    if targets.is_empty() {
        return false;
    }
    let (to, least, most) = targets[rng.index(targets.len())];
    let moved = rng.range(least, most);
    if c.bones[to].organ_mass <= 0.0
        && let Some((low, high)) = organ_range(&c.bones[to], &c.nodes, center)
    {
        c.bones[to].organ_at = rng.range(low, high);
    }
    c.bones[from].organ_mass -= moved;
    c.bones[to].organ_mass += moved;
    true
}

#[cfg(test)]
mod tests {
    use super::super::tests::bodies;
    use super::super::{Context, Operator};
    use super::*;
    use crate::evolution::{Muscle, NO_SENSOR, Population, repair};

    fn cx() -> Context<'static> {
        Context { donor: None }
    }

    /// Whether two phases are the same point of the cycle.
    fn same_phase(a: f32, b: f32) -> bool {
        let d = (a - b).rem_euclid(1.0);
        !(1e-4..=1.0 - 1e-4).contains(&d)
    }

    /// Runs `op` on a copy of each body and returns the changed copies with
    /// their body indices. It must apply to at least a tenth of the bodies.
    fn applied(op: Operator, bodies: &[Creature]) -> Vec<(usize, Creature)> {
        let cfg = Config::default();
        let out: Vec<(usize, Creature)> = bodies
            .iter()
            .enumerate()
            .filter_map(|(i, body)| {
                let mut c = body.clone();
                op(&mut c, &cfg, &mut Rng::new(5, 0, i), &cx()).then_some((i, c))
            })
            .collect();
        assert!(
            out.len() * 10 >= bodies.len(),
            "applied to {} of {} bodies",
            out.len(),
            bodies.len()
        );
        out
    }

    #[test]
    fn redistribute_joint_flex_keeps_the_total_range_of_two_neighbours() {
        let bodies = bodies(&Config::default(), 160);
        for (i, c) in applied(redistribute_joint_flex, &bodies) {
            let body = &bodies[i];
            let changed: Vec<usize> = (0..c.bones.len())
                .filter(|&b| c.bones[b] != body.bones[b])
                .collect();
            assert_eq!(changed.len(), 2);
            let (x, y) = (c.bones[changed[0]], c.bones[changed[1]]);
            assert!(x.b == y.a || y.b == x.a, "the two joints are neighbours");
            let width = |b: &Bone| b.max_angle - b.min_angle;
            let before = width(&body.bones[changed[0]]) + width(&body.bones[changed[1]]);
            assert!((width(&x) + width(&y) - before).abs() < 1e-4);
            assert!((width(&x) - width(&body.bones[changed[0]])).abs() > 1e-3);
            for b in [x, y] {
                assert!(b.min_angle <= 0.0 && b.min_angle >= -JOINT_LIMIT - 1e-5);
                assert!(b.max_angle >= 0.0 && b.max_angle <= JOINT_LIMIT + 1e-5);
            }
        }
    }

    #[test]
    fn mutate_matching_limbs_changes_one_position_of_two_matching_limbs() {
        let bodies = bodies(&Config::default(), 160);
        for (x, y) in bodies.iter().flat_map(matching_limbs) {
            assert_eq!(x.len(), y.len());
            assert!(x.iter().all(|b| !y.contains(b)));
        }
        let results = applied(mutate_matching_limbs, &bodies);
        let mut both = 0;
        for (i, c) in &results {
            let body = &bodies[*i];
            let mut touched: Vec<usize> = (0..c.bones.len())
                .filter(|&b| c.bones[b] != body.bones[b])
                .collect();
            for (m, n) in c.muscles.iter().zip(&body.muscles) {
                assert_eq!((m.phase, m.duty, m.period), (n.phase, n.duty, n.period));
                if m.anchor_a != n.anchor_a {
                    touched.push(m.bone_a as usize);
                }
                if m.anchor_b != n.anchor_b {
                    touched.push(m.bone_b as usize);
                }
            }
            touched.sort_unstable();
            touched.dedup();
            assert!(!touched.is_empty() && touched.len() <= 2);
            let fits = matching_limbs(body).iter().any(|(x, y)| {
                (0..x.len()).any(|k| touched.iter().all(|t| *t == x[k] || *t == y[k]))
            });
            assert!(
                fits,
                "changed bones {touched:?} are not one position of a pair"
            );
            if touched.len() == 2 {
                both += 1;
            }
        }
        // Clamping can hide the change on one of the two limbs, but rarely.
        assert!(both * 2 >= results.len(), "{both} of {}", results.len());
    }

    #[test]
    fn chain_phase_wave_grows_the_phase_one_step_per_bone() {
        let bodies = bodies(&Config::default(), 160);
        applied(chain_phase_wave, &bodies);
        for (i, body) in bodies.iter().enumerate() {
            let Some(chain) = chain_below(body, &mut Rng::new(3, 0, i)) else {
                continue;
            };
            assert!(chain.len() >= 2);
            for pair in chain.windows(2) {
                assert_eq!(body.bones[pair[0]].b, body.bones[pair[1]].a);
            }
            let parts: Vec<Vec<usize>> = chain.iter().map(|&b| vec![b]).collect();
            let groups = muscle_groups(body, &parts);
            let mut c = body.clone();
            assert!(phase_wave(&mut c, &chain, 0.1));
            let start = body.muscles[*groups.iter().flatten().next().unwrap()].phase;
            for (k, group) in groups.iter().enumerate() {
                for &m in group {
                    assert!(same_phase(c.muscles[m].phase, start + 0.1 * k as f32));
                }
            }
            for m in 0..c.muscles.len() {
                if !groups.iter().flatten().any(|&g| g == m) {
                    assert_eq!(c.muscles[m], body.muscles[m]);
                }
            }
        }
    }

    #[test]
    fn limb_phase_pattern_shifts_whole_limbs_to_the_pattern() {
        let bodies = bodies(&Config::default(), 160);
        applied(limb_phase_pattern, &bodies);
        for body in &bodies {
            let limbs = leaf_limbs(body);
            let bones: Vec<usize> = limbs.iter().flatten().copied().collect();
            assert!((1..bones.len()).all(|k| !bones[..k].contains(&bones[k])));
            if limbs.len() < 2 {
                continue;
            }
            let groups = muscle_groups(body, &limbs);
            let start = body.muscles[*groups.iter().flatten().next().unwrap()].phase;
            for pattern in 0..3 {
                let mut c = body.clone();
                assert!(shift_limbs(&mut c, &limbs, pattern));
                for (i, group) in groups.iter().enumerate() {
                    let Some(&lead) = group.first() else {
                        continue;
                    };
                    let offset = [0.0, 0.5 * (i % 2) as f32, i as f32 / limbs.len() as f32];
                    assert!(same_phase(c.muscles[lead].phase, start + offset[pattern]));
                    let shift = c.muscles[lead].phase - body.muscles[lead].phase;
                    for &m in group {
                        let moved = c.muscles[m].phase - body.muscles[m].phase;
                        assert!(same_phase(moved, shift), "timing inside the limb stays");
                    }
                }
            }
        }
    }

    #[test]
    fn limb_duty_cycle_keeps_each_contraction_middle() {
        let bodies = bodies(&Config::default(), 160);
        for (i, c) in applied(limb_duty_cycle, &bodies) {
            let mut amounts = Vec::new();
            for (after, before) in c.muscles.iter().zip(&bodies[i].muscles) {
                assert!(same_phase(
                    after.phase - 0.5 * after.duty,
                    before.phase - 0.5 * before.duty
                ));
                if after.duty != before.duty && after.duty != 0.05 && after.duty != 0.95 {
                    amounts.push(after.duty - before.duty);
                }
            }
            assert!(amounts.iter().all(|a| (a - amounts[0]).abs() < 1e-5));
        }
    }

    #[test]
    fn touchdown_package_makes_one_foot_restart_its_muscles() {
        let bodies = bodies(&Config::default(), 160);
        for (i, c) in applied(touchdown_package, &bodies) {
            let body = &bodies[i];
            let parents = parent_bones(&c);
            let sensed = |m: &Muscle| {
                let (a, b) = (c.bones[m.bone_a as usize], c.bones[m.bone_b as usize]);
                (m.sensor != NO_SENSOR).then(|| [a.a, a.b, b.a, b.b][m.sensor as usize] as usize)
            };
            let package = |foot: usize| muscles_on(&c, &[parents[foot].unwrap()], false);
            let foot = (1..c.nodes.len())
                .filter(|&n| degree(&c, n) == 1)
                .find(|&foot| {
                    let muscles = package(foot);
                    !muscles.is_empty() && {
                        let lead = c.muscles[muscles[0]];
                        muscles.iter().all(|&m| {
                            let m = c.muscles[m];
                            sensed(&m) == Some(foot)
                                && same_phase(m.reset - m.phase, lead.reset - lead.phase)
                        })
                    }
                })
                .expect("one foot senses the touchdowns of its muscles");
            let muscles = package(foot);
            for (m, (after, before)) in c.muscles.iter().zip(&body.muscles).enumerate() {
                if muscles.contains(&m) {
                    let sensor_only = Muscle {
                        sensor: before.sensor,
                        reset: before.reset,
                        tendon: 0.0,
                        ..*after
                    };
                    assert_eq!(sensor_only, *before);
                } else {
                    assert_eq!(after, before);
                }
            }
        }
    }

    #[test]
    fn redistribute_organ_mass_keeps_the_total_organ_mass() {
        let cfg = Config::default();
        let total = |c: &Creature| c.bones.iter().map(|b| b.organ_mass).sum::<f32>();
        // The test bodies have no organs, so each one gets an organ first.
        let bodies: Vec<Creature> = bodies(&cfg, 160)
            .into_iter()
            .enumerate()
            .filter_map(|(i, mut c)| {
                let center = organ_center(&c.nodes);
                let (bone, (low, high)) = (0..c.bones.len()).find_map(|b| {
                    organ_range(&c.bones[b], &c.nodes, center).map(|range| (b, range))
                })?;
                c.bones[bone].organ_mass = Rng::new(9, 0, i).range(0.03, MAX_ORGAN_MASS);
                c.bones[bone].organ_at = 0.5 * (low + high);
                Some(c)
            })
            .collect();
        assert!(bodies.len() > 100);
        for (i, mut c) in applied(redistribute_organ_mass, &bodies) {
            let body = &bodies[i];
            let changed = (0..c.bones.len())
                .filter(|&b| c.bones[b] != body.bones[b])
                .count();
            assert_eq!(changed, 2);
            assert!((total(&c) - total(body)).abs() < 1e-5);
            for b in &c.bones {
                let organ = MIN_ORGAN_MASS - 1e-6..=MAX_ORGAN_MASS;
                assert!(b.organ_mass == 0.0 || organ.contains(&b.organ_mass));
            }
            repair(&mut c, &cfg, &mut Rng::new(1, 0, i));
            assert!(
                (total(&c) - total(body)).abs() < 1e-5,
                "repair keeps the mass"
            );
            let mut pop = Population::default();
            pop.push(c);
            let check = Config {
                population: 1,
                ..cfg.clone()
            };
            pop.validate(&check).unwrap();
        }
    }
}
