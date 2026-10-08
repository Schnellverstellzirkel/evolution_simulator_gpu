//! The seven operators here retune a body and add or remove no bone and no
//! muscle. They trade joint range between neighbours, change two matching
//! limbs alike, shift the phases and duty of a limb's muscles, tie a foot's
//! touchdown to its muscles and move organ mass between bones. `BASE_OPERATORS`
//! in `anatomy/mod.rs` lists them, and each has a pick slot of its own. The
//! limb finders that the other operator files share are here too, such as
//! `leaf_limbs` and `matching_limbs`.
use super::muscles::shift_phase;
use super::{
    BoneIds, Children, Context, Limbs, MuscleIds, branch, branch_in, child_bones, degree, is_neck,
    muscles_on, parent_bones,
};
use crate::config::Config;
use crate::evolution::{
    Bone, Bounded, Creature, JOINT_LIMIT, MAX_MUSCLES, MAX_NODES, MAX_ORGAN_MASS, MIN_ORGAN_MASS,
    Rng, max_bone_length, organ_center, organ_range,
};
use crate::qd::gaussian;

/// Moves joint range from one joint to a neighbouring joint. The neighbour is
/// the bone above or a bone below, and neither bone is the neck. The first
/// joint narrows and the second widens by the same angle, so the two ranges
/// keep their total width. The angle is 20 to 80% of the most that can move,
/// which is the smaller of the narrowed joint's width and the widened joint's
/// room up to `JOINT_LIMIT`. It returns false when no pair has more than 0.01
/// rad to move.
pub(crate) fn redistribute_joint_flex(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let width = |b: &Bone| b.max_angle - b.min_angle;
    // A bone and the bone above or below it, as (narrowed, widened, most angle
    // that can move). The neck's range limits nothing, so it takes no part.
    // Each bone has one bone above it, and each such link gives two pairs, one
    // for each bone as the narrowed one. So there are at most two pairs per
    // bone.
    let mut pairs: Bounded<(usize, usize, f32), { 2 * MAX_NODES }> = Bounded::new();
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
    // grows each side by the same share of its room up to `JOINT_LIMIT`. Here
    // `below` is the room under its lower limit and `above` the room over its
    // upper limit.
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

/// Picks a pair of matching limbs (`matching_limbs`) and one position along
/// them. Then it makes one random change to the two bones at that position,
/// the same change to both. The change is one of three, with equal odds. It
/// scales both lengths by `exp(0.15 * g)`. Or it moves both joint ranges, the
/// lower limit by `0.15 * g` and the upper limit by another `0.15 * g`. Or it
/// moves every muscle end on either bone along the bone by `0.1 * g` of its
/// length. Each `g` is a new gaussian draw, and lengths, ranges and anchors
/// stay within their limits. Muscle timing is untouched, so the pair keeps its
/// timing difference. It returns false when there is no pair, or when the
/// change moves nothing.
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
    let (x, y) = pairs.get(rng.index(pairs.len()));
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

/// Pairs of matching branches (`matching_limbs`): the branches, and the pairs
/// as indices into them.
pub(super) struct LimbPairs {
    /// The branch that starts with each bone but the neck, in bone order
    /// (`branch`).
    pub limbs: Limbs,
    /// Each pair as two indices into `limbs`, the lower index first. The pairs
    /// run in order of the first index, then the second.
    pub pairs: Bounded<(u8, u8), { MAX_NODES * MAX_NODES / 2 }>,
}
impl LimbPairs {
    /// The number of pairs.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }
    /// Whether there is no pair.
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
    /// The two branches of pair `k`.
    pub fn get(&self, k: usize) -> (&BoneIds, &BoneIds) {
        let (x, y) = self.pairs[k];
        (&self.limbs[x as usize], &self.limbs[y as usize])
    }
    /// The pairs in order, as two branches each.
    pub fn iter(&self) -> impl Iterator<Item = (&BoneIds, &BoneIds)> {
        (0..self.len()).map(|k| self.get(k))
    }
}
impl IntoIterator for LimbPairs {
    type Item = (BoneIds, BoneIds);
    type IntoIter = std::vec::IntoIter<(BoneIds, BoneIds)>;
    /// The pairs by value, for tests.
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
            .map(|(x, y)| (*x, *y))
            .collect::<Vec<_>>()
            .into_iter()
    }
}

/// Finds the pairs of branches that match. The two branches of a pair share no
/// bone and have the same bone count. Taken in `branch` order, their bones are
/// of similar length: at each position the longer is at most 1.25 times the
/// shorter. The trees may still differ in shape. Every branch that starts with
/// a bone but the neck is a candidate, so a pair can be two legs or two larger
/// branches.
pub(super) fn matching_limbs(c: &Creature) -> LimbPairs {
    let children = child_bones(c);
    let limbs: Limbs = (0..c.bones.len())
        .filter(|&b| !is_neck(c, b))
        .map(|b| branch_in(c, &children, b))
        .collect();
    let similar = |(&p, &q): (&usize, &usize)| {
        let (a, b) = (c.bones[p].rest_length, c.bones[q].rest_length);
        a.max(b) <= 1.25 * a.min(b)
    };
    let mut pairs = Bounded::new();
    for (i, x) in limbs.iter().enumerate() {
        for (j, y) in limbs.iter().enumerate().skip(i + 1) {
            if x.len() == y.len()
                && !x.contains(&y[0])
                && !y.contains(&x[0])
                && x.iter().zip(y.iter()).all(similar)
            {
                pairs.push((i as u8, j as u8));
            }
        }
    }
    LimbPairs { limbs, pairs }
}

/// Along a chain of bones (a path down one branch), sets the phase of the
/// muscles on successive bones to grow by one step per bone: a contraction
/// wave, for curling and crawling. The step is 0.05 to 0.25 of a cycle, with
/// equal odds of a positive or a negative sign. It returns false when the body
/// has no chain, or when no muscle is on the chain.
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

/// A path from a random bone that is not the neck and has a bone below it,
/// down to a bone with none below it. It takes a random child bone at each
/// junction, and it has at least two bones. It is `None` when no bone can start
/// such a path.
fn chain_below(c: &Creature, rng: &mut Rng) -> Option<BoneIds> {
    let children = child_bones(c);
    let starts: BoneIds = (0..c.bones.len())
        .filter(|&b| !is_neck(c, b) && !children[c.bones[b].b as usize].is_empty())
        .collect();
    if starts.is_empty() {
        return None;
    }
    let mut chain = BoneIds::from_slice(&[starts[rng.index(starts.len())]]);
    loop {
        let below = &children[c.bones[chain[chain.len() - 1]].b as usize];
        if below.is_empty() {
            return Some(chain);
        }
        chain.push(below[rng.index(below.len())]);
    }
}

/// Sets every muscle on `chain[i]` to the phase of the chain's first muscle
/// plus `i` steps of `step` cycles. The first muscle is the first one found
/// from the top bone down. A muscle on two chain bones counts for the upper
/// one. It returns false when no muscle is on the chain.
fn phase_wave(c: &mut Creature, chain: &[usize], step: f32) -> bool {
    let parts: Limbs = chain.iter().map(|&b| BoneIds::from_slice(&[b])).collect();
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

/// Shifts whole limbs to a pattern of phase offsets: all together, every
/// second limb half a cycle later, or limb `i` of `n` a share `i / n` of a
/// cycle later (evenly staggered). The three patterns have equal odds. The
/// limbs are the branches at one junction (a node with two or more bones below
/// it) or every leaf limb, with equal odds. Each limb's muscles move by one
/// amount, so the timing inside a limb stays. It returns false when the
/// junction pick finds no junction, when there are fewer than two limbs, or
/// when no muscle is on them.
pub(crate) fn limb_phase_pattern(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let limbs = if rng.unit() < 0.5 {
        let children = child_bones(c);
        let junctions: Bounded<&[usize], MAX_NODES> =
            children.iter().filter(|list| list.len() > 1).collect();
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

/// Every limb that ends in a foot, one for each leaf node, in node order. A
/// limb is the chain of bones from a leaf node up to the node where the body
/// branches, or up to the neck. It can be a single bone, and it never holds
/// the neck.
pub(super) fn leaf_limbs(c: &Creature) -> Limbs {
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
                    _ => return Some(branch_in(c, &children, root)),
                }
            }
        })
        .collect()
}

/// The bones that hang from `node` and start a limb without junctions: a chain
/// with no branching below it, which ends in one tip. `children` are the child
/// lists of `c` (`child_bones`).
pub(super) fn leaf_limbs_at(c: &Creature, children: &Children, node: usize) -> BoneIds {
    children[node]
        .iter()
        .copied()
        .filter(|&l| {
            branch_in(c, children, l)
                .iter()
                .all(|&x| children[c.bones[x].b as usize].len() <= 1)
        })
        .collect()
}

/// The node a limb's last bone ends on: its foot or tip.
pub(super) fn foot(c: &Creature, limb: &[usize]) -> usize {
    c.bones[limb[limb.len() - 1]].b as usize
}

/// The node a limb hangs from: its hip.
pub(super) fn hip(c: &Creature, limb: &[usize]) -> usize {
    c.bones[limb[0]].a as usize
}

/// Where a limb's last bone ends along x (m) in the starting pose.
pub(super) fn tip_x(c: &Creature, limb: &[usize]) -> f32 {
    c.nodes[foot(c, limb)].x
}

/// Where a limb's last bone ends in height (m) in the starting pose.
pub(super) fn tip_y(c: &Creature, limb: &[usize]) -> f32 {
    c.nodes[foot(c, limb)].y
}

/// The leaf limbs from front (large x) to back, by the x of their feet. Limbs
/// with the same x stay in the order of `leaf_limbs`.
pub(super) fn limbs_front_to_back(c: &Creature) -> Limbs {
    let mut limbs = leaf_limbs(c);
    limbs.sort_stable_by(|x, y| tip_x(c, y).total_cmp(&tip_x(c, x)));
    limbs
}

/// The bones that carry an organ (organ mass above zero).
pub(super) fn organ_bones(c: &Creature) -> BoneIds {
    (0..c.bones.len())
        .filter(|&b| c.bones[b].organ_mass > 0.0)
        .collect()
}

/// Shifts the muscles of each limb by one common amount, so that the first
/// muscle of limb `i` lands on the phase of the first muscle of the first limb
/// that has one, plus an offset for `i`. Pattern 0 has offset 0 (all together).
/// Pattern 1 has 0.5 when `i` is odd and 0 when it is even (alternating
/// halves). Any other pattern has `i / limbs.len()` of a cycle (evenly
/// staggered). A muscle on two limbs moves with the earlier one. It returns
/// false when no limb has a muscle.
fn shift_limbs(c: &mut Creature, limbs: &[BoneIds], pattern: usize) -> bool {
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
            shift_phase(c, m, shift);
        }
    }
    true
}

/// The muscles with an end on each part (a list of bones), one group for each
/// part, in order. A muscle on two parts belongs to the first.
pub(super) fn muscle_groups(c: &Creature, parts: &[BoneIds]) -> Bounded<MuscleIds, MAX_NODES> {
    let mut taken = [false; MAX_MUSCLES];
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
/// phases so each contraction keeps its middle: a slower contraction with a
/// quicker release, or the reverse. The limb is the branch that starts with a
/// random bone other than the neck, and a muscle counts if either of its ends
/// is on it. The amount is 0.03 to 0.2 of a cycle, up or down with equal odds,
/// and each duty stays within 0.05 and 0.95. It returns false when no duty
/// changes.
pub(crate) fn limb_duty_cycle(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let roots: BoneIds = (0..c.bones.len()).filter(|&b| !is_neck(c, b)).collect();
    if roots.is_empty() {
        return false;
    }
    let limb = branch(c, roots[rng.index(roots.len())]);
    let amount = rng.range(0.03, 0.2) * if rng.unit() < 0.5 { -1.0 } else { 1.0 };
    let mut changed = false;
    // A muscle contracts while its cycle position is below `duty`, and the
    // cycle position is the time in periods plus `phase`. So the middle of the
    // contraction comes when the time is `duty / 2` minus `phase`, and `phase`
    // moves by half the change in duty to keep that time.
    for i in muscles_on(c, &limb, false) {
        let m = &mut c.muscles[i];
        let duty = (m.duty + amount).clamp(0.05, 0.95);
        m.phase = (m.phase + 0.5 * (duty - m.duty)).rem_euclid(1.0);
        changed |= duty != m.duty;
        m.duty = duty;
    }
    changed
}

/// Picks a foot (a node other than the head with one bone) that has a muscle
/// on its bone. Every muscle on that bone then senses the foot's touchdown, in
/// place of any sensor it had. The first muscle resets to a random phase, and
/// each other muscle resets to that phase plus its own phase gap to the first.
/// So a landing restarts the limb's movement as a whole, with the muscles in
/// the same order and spacing as before. It returns false when no foot has a
/// muscle on its bone.
pub(crate) fn touchdown_package(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let parents = parent_bones(c);
    let feet: Bounded<(usize, MuscleIds), MAX_NODES> = (1..c.nodes.len())
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

/// Moves part of one organ's mass to another bone, so the total organ mass
/// stays the same. The source keeps at least `MIN_ORGAN_MASS`. The target is
/// either a bone with an organ, which stays within `MAX_ORGAN_MASS`, or a bone
/// with no organ that has a stretch within `ORGAN_RADIUS` of the organ center.
/// That bone gets a new organ of at least `MIN_ORGAN_MASS` at a random place
/// in the stretch. It returns false when the body has no organ, or when no
/// other bone can take mass.
pub(crate) fn redistribute_organ_mass(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let organs = organ_bones(c);
    if organs.is_empty() {
        return false;
    }
    let from = organs[rng.index(organs.len())];
    // The source keeps at least the lightest organ, and a new organ starts
    // with at least that much. Each target comes with the least and most mass
    // it can take. It counts only when the most is over 0.001 kg above the
    // least.
    let spare = c.bones[from].organ_mass - MIN_ORGAN_MASS;
    let center = organ_center(&c.nodes);
    let targets: Bounded<(usize, f32, f32), MAX_NODES> = (0..c.bones.len())
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
    use super::super::tests::{bodies, same_phase};
    use super::super::{Context, Operator};
    use super::*;
    use crate::evolution::{Muscle, NO_SENSOR, Population, repair};

    fn cx() -> Context<'static> {
        Context::of(None)
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
        // A limit, or a bone with no muscle end in an anchor change, can leave
        // one of the two bones as it was. So only half of the results must
        // change both.
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
            let parts: Vec<BoneIds> = chain.iter().map(|&b| BoneIds::from_slice(&[b])).collect();
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
