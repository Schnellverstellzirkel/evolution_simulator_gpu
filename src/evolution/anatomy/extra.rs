//! A second set of operators, after the audit in docs/anatomy-operators.md:
//! muscle and rhythm changes kept most of a parent's gait, skeletal ones a
//! few percent. These copy programs between limbs of the same shape, trade
//! the body clock against stroke length, change leverage or strength as a
//! whole, duplicate or grow limbs so the gait survives, and remove the parts
//! that do the least.
//!
//! Operators here that add or remove bones close the motor ring themselves
//! with passive muscles (`passive_ring`). Left to `repair`, the ring
//! would get new random muscles that drive from the first step.
use super::limbs::{clamped, fuse_pair, limb_roots, pick};
use super::muscles::{actuation, ring, shared_node};
use super::rhythm::{leaf_limbs, matching_limbs};
use super::{
    BoneIds, Context, Limbs, MuscleIds, branch, child_bones, copy_branch, degree, fit_stroke,
    is_neck, muscles_on, new_muscle, parent_bones, pick_each, remove_parts, room,
};
use crate::config::Config;
use crate::evolution::{
    Bone, Bounded, Creature, MAX_MUSCLES, MAX_NODES, Muscle, NodeGene, Rng, max_bone_length,
    min_muscle_period,
};

/// Copies one limb's program (each muscle's phase and duty) onto the
/// matching muscles of a limb of the same shape, half a cycle later: the two
/// limbs run one program in alternation.
pub(crate) fn mirror_limb_timing(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(pairs) = pick_counterparts(c, rng) else {
        return false;
    };
    let mut changed = false;
    for (p, q) in pairs {
        let phase = (c.muscles[p].phase + 0.5).rem_euclid(1.0);
        let duty = c.muscles[p].duty;
        let m = &mut c.muscles[q];
        changed |= (m.phase, m.duty) != (phase, duty);
        (m.phase, m.duty) = (phase, duty);
    }
    changed
}

/// Exchanges the programs (phase and duty of matching muscles) of two limbs
/// of the same shape. For limbs of one shape this is the same as swapping
/// their places: the leading limb becomes the trailing one.
pub(crate) fn swap_limb_programs(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(pairs) = pick_counterparts(c, rng) else {
        return false;
    };
    let mut changed = false;
    for (p, q) in pairs {
        let (x, y) = (c.muscles[p], c.muscles[q]);
        changed |= (x.phase, x.duty) != (y.phase, y.duty);
        (c.muscles[p].phase, c.muscles[p].duty) = (y.phase, y.duty);
        (c.muscles[q].phase, c.muscles[q].duty) = (x.phase, x.duty);
    }
    changed
}

/// Copies a muscle that one limb has and its same-shaped partner lacks onto
/// the partner's matching bones, stroke fitted to the new span, and timed to
/// the partner's program (the phase offset of the first pair of matching
/// muscles, or half a cycle without one).
pub(crate) fn copy_muscle_to_partner(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    if !room(c, cfg, 0, 1) {
        return false;
    }
    // (source muscle, partner pair, phase offset)
    let partners = partners(c);
    let options = |push: &mut dyn FnMut((usize, usize, f32))| {
        for k in 0..partners.len() {
            let (from, to) = partners.get(k);
            let pairs = counterparts(c, from, to);
            let offset = pairs
                .first()
                .map_or(0.5, |&(p, q)| c.muscles[q].phase - c.muscles[p].phase);
            for p in muscles_on(c, from, true) {
                if pairs.iter().all(|&(s, _)| s != p) {
                    push((p, k, offset));
                }
            }
        }
    };
    let Some((p, k, offset)) = pick_each(rng, options) else {
        return false;
    };
    let (from, to) = partners.get(k);
    let old = c.muscles[p];
    let at = |b: u32| {
        to[from
            .iter()
            .position(|&x| x == b as usize)
            .expect("limb bone")] as u32
    };
    let mut m = Muscle {
        bone_a: at(old.bone_a),
        bone_b: at(old.bone_b),
        phase: (old.phase + offset).rem_euclid(1.0),
        reset: (old.reset + offset).rem_euclid(1.0),
        tendon: 0.0,
        ..old
    };
    fit_stroke(c, &mut m, Some(&old));
    c.muscles.push(m);
    true
}

/// Duplicates a limb in place: the copy hangs from the same joint in the
/// same pose, with the same muscles at the same phase, so the child moves
/// like its parent and later mutations can make the two limbs differ.
pub(crate) fn twin_limb(c: &mut Creature, cfg: &Config, rng: &mut Rng, _cx: &Context) -> bool {
    let roots: BoneIds = limb_roots(c)
        .into_iter()
        .filter(|&b| room(c, cfg, branch(c, b).len(), 0))
        .collect();
    let Some(root) = pick(&roots, rng) else {
        return false;
    };
    let joint = c.bones[root].a as usize;
    if copy_branch(c, cfg, root, joint, |p| p, false, 0.0).is_none() {
        return false;
    }
    passive_ring(c, cfg, rng);
    true
}

/// Grows the same actuated tip on both limbs of a same-shaped pair: a short
/// bone (the same share of each tip bone), turned the same way (mirrored when
/// the limbs point to opposite sides), with the same narrow joint range and a
/// muscle to the tip bone timed like a muscle of its own limb.
pub(crate) fn grow_matching_tips(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    if !room(c, cfg, 2, 2) {
        return false;
    }
    let children = child_bones(c);
    let tip = |b: usize| children[c.bones[b].b as usize].is_empty();
    let matching = matching_limbs(c);
    let options = |push: &mut dyn FnMut((usize, usize))| {
        matching
            .iter()
            .flat_map(|(x, y)| x.iter().copied().zip(y.iter().copied()))
            .filter(|&(p, q)| tip(p) && tip(q))
            .for_each(push);
    };
    let Some((p, q)) = pick_each(rng, options) else {
        return false;
    };
    let share = rng.range(0.25, 0.5);
    let turn = rng.range(-1.5, 1.5);
    let (low, high) = (rng.range(0.15, 0.5), rng.range(0.15, 0.5));
    let anchors = (rng.range(0.3, 1.0), rng.range(0.2, 0.8));
    let direction = |b: usize| {
        let bone = c.bones[b];
        c.nodes[bone.b as usize].x - c.nodes[bone.a as usize].x
    };
    let mirrored = direction(p) * direction(q) < 0.0;
    for (bone, side) in [(p, 1.0), (q, if mirrored { -1.0 } else { 1.0 })] {
        let old = c.bones[bone];
        let (start, end) = (c.nodes[old.a as usize], c.nodes[old.b as usize]);
        let length = (old.rest_length * share).max(0.03);
        let angle = (end.y - start.y).atan2(end.x - start.x) + side * turn;
        let [x, y] = clamped(end.x + length * angle.cos(), end.y + length * angle.sin());
        c.nodes.push(NodeGene { x, y, ..end });
        let mut toe = Bone::new(old.b, c.nodes.len() as u32 - 1, length);
        (toe.min_angle, toe.max_angle) = if side > 0.0 {
            (-low, high)
        } else {
            (-high, low)
        };
        c.bones.push(toe);
        let template = muscles_on(c, &[bone], false).first().map(|&i| c.muscles[i]);
        let m = new_muscle(c, c.bones.len() - 1, bone, anchors, template.as_ref(), rng);
        c.muscles.push(m);
    }
    passive_ring(c, cfg, rng);
    true
}

/// Moves every muscle with an end on a limb 2 to 12% of a cycle earlier or
/// later (touchdown resets too), so the limb leads or lags the body.
pub(crate) fn nudge_limb_phase(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let roots: BoneIds = limb_roots(c)
        .into_iter()
        .filter(|&b| !muscles_on(c, &branch(c, b), false).is_empty())
        .collect();
    let Some(root) = pick(&roots, rng) else {
        return false;
    };
    let shift = rng.range(0.02, 0.12) * if rng.unit() < 0.5 { -1.0 } else { 1.0 };
    for i in muscles_on(c, &branch(c, root), false) {
        let m = &mut c.muscles[i];
        m.phase = (m.phase + shift).rem_euclid(1.0);
        m.reset = (m.reset + shift).rem_euclid(1.0);
    }
    true
}

/// Scales the body clock's period and every muscle's stroke by one factor
/// (0.7 to 1.4), each muscle keeping its relaxed length. A muscle's drive
/// follows the speed of its target length, which stays the same: quicker,
/// shorter strokes or slower, longer ones with the same force.
pub(crate) fn cadence_stride_trade(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some(period) = c.muscles.first().map(|m| m.period) else {
        return false;
    };
    let wanted = rng.range(0.7f32.ln(), 1.4f32.ln()).exp();
    let factor = (period * wanted).clamp(min_muscle_period(), 10.0) / period;
    if (factor - 1.0).abs() < 0.02 {
        return false;
    }
    for m in &mut c.muscles {
        m.period *= factor;
        m.short = (m.long - (m.long - m.short) * factor).clamp(0.01, m.long);
    }
    true
}

/// Moves both ends of a muscle across one joint toward the joint or away from
/// it by one factor (0.5 to 2), and refits its stroke to the new span: the
/// same muscle with a shorter or longer lever.
pub(crate) fn scale_muscle_leverage(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    // How far an anchor sits from `node`, as a share of its bone.
    let from_joint = |c: &Creature, bone: u32, anchor: f32, node: u32| {
        if c.bones[bone as usize].a == node {
            anchor
        } else {
            1.0 - anchor
        }
    };
    let options: Bounded<(usize, u32), MAX_MUSCLES> = (0..c.muscles.len())
        .filter_map(|i| {
            let m = c.muscles[i];
            let node = shared_node(c, m.bone_a as usize, m.bone_b as usize)?;
            let far = from_joint(c, m.bone_a, m.anchor_a, node)
                .max(from_joint(c, m.bone_b, m.anchor_b, node));
            (far >= 0.05).then_some((i, node))
        })
        .collect();
    let Some((i, node)) = pick(&options, rng) else {
        return false;
    };
    let factor = rng.range(0.5f32.ln(), 2.0f32.ln()).exp();
    let old = c.muscles[i];
    let mut m = old;
    for (bone, anchor) in [(m.bone_a, &mut m.anchor_a), (m.bone_b, &mut m.anchor_b)] {
        let distance = (from_joint(c, bone, *anchor, node) * factor).min(1.0);
        *anchor = if c.bones[bone as usize].a == node {
            distance
        } else {
            1.0 - distance
        };
    }
    fit_stroke(c, &mut m, Some(&old));
    c.muscles[i] = m;
    m != old
}

/// Scales the stiffness of every active muscle with an end on a limb by one
/// factor (0.6 to 1.6): the limb pushes harder or softer with the same
/// timing and geometry.
pub(crate) fn scale_limb_strength(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let active = |c: &Creature, root: usize| -> MuscleIds {
        muscles_on(c, &branch(c, root), false)
            .into_iter()
            .filter(|&i| c.muscles[i].long > c.muscles[i].short)
            .collect()
    };
    let roots: BoneIds = limb_roots(c)
        .into_iter()
        .filter(|&b| !active(c, b).is_empty())
        .collect();
    let Some(root) = pick(&roots, rng) else {
        return false;
    };
    let factor = rng.range(0.6f32.ln(), 1.6f32.ln()).exp();
    let mut changed = false;
    for i in active(c, root) {
        let m = &mut c.muscles[i];
        let stiffness = (m.stiffness * factor).clamp(1.0, 120.0);
        changed |= stiffness != m.stiffness;
        m.stiffness = stiffness;
    }
    changed
}

/// Removes the weakest of three random muscles outside the motor ring: the
/// one with the least drive (stiffness times stroke), whose loss changes the
/// gait least.
pub(crate) fn prune_weakest_muscle(
    c: &mut Creature,
    _cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    if c.bones.len() < 3 {
        return false;
    }
    let free: MuscleIds = (0..c.muscles.len())
        .filter(|&i| !ring(c, &c.muscles[i]))
        .collect();
    if free.is_empty() {
        return false;
    }
    let weakest = (0..3)
        .map(|_| free[rng.index(free.len())])
        .min_by(|&x, &y| drive(&c.muscles[x]).total_cmp(&drive(&c.muscles[y])))
        .expect("three picks");
    c.muscles.remove(weakest);
    true
}

/// Removes the idlest of three random limb tips (a leaf bone that is not the
/// neck): the one whose muscles have the least drive in total, with its node
/// and its muscles.
pub(crate) fn prune_idle_limb(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    if c.nodes.len() <= 3 {
        return false;
    }
    let tips: BoneIds = (0..c.bones.len())
        .filter(|&b| !is_neck(c, b) && degree(c, c.bones[b].b as usize) == 1)
        .collect();
    if tips.is_empty() {
        return false;
    }
    let work = |b: usize| -> f32 {
        muscles_on(c, &[b], false)
            .iter()
            .map(|&i| drive(&c.muscles[i]))
            .sum()
    };
    let idlest = (0..3)
        .map(|_| tips[rng.index(tips.len())])
        .min_by(|&x, &y| work(x).total_cmp(&work(y)))
        .expect("three picks");
    let tip = c.bones[idlest].b as usize;
    remove_parts(c, &[idlest], &[tip]);
    passive_ring(c, cfg, rng);
    true
}

/// Merges the last two bones of a limb (a joint with one bone below it that
/// ends in a foot) into one bone from the upper joint to the foot, which
/// stays where it was. Muscles on either bone keep their place on the body.
pub(crate) fn merge_leaf_bones(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    if c.nodes.len() <= 3 {
        return false;
    }
    let parents = parent_bones(c);
    let children = child_bones(c);
    let pairs: Bounded<(usize, usize), MAX_NODES> = (1..c.nodes.len())
        .filter_map(|joint| {
            let upper = parents[joint]?;
            let &[lower] = &children[joint][..] else {
                return None;
            };
            let foot = c.bones[lower].b as usize;
            let (top, end) = (c.nodes[c.bones[upper].a as usize], c.nodes[foot]);
            let length = (end.x - top.x).hypot(end.y - top.y);
            (!is_neck(c, upper)
                && children[foot].is_empty()
                && (0.03..=max_bone_length()).contains(&length))
            .then_some((upper, lower))
        })
        .collect();
    let Some((upper, lower)) = pick(&pairs, rng) else {
        return false;
    };
    fuse_pair(c, upper, lower);
    passive_ring(c, cfg, rng);
    true
}

/// The owner saw evolution settle on one strong leg at one end of the body
/// while the other end drags. Copies the working leg (the leg whose muscles
/// drive most) to the other end of the body, mirrored front to back, half a
/// cycle apart (gallop or bound) or in phase (hop). A weak leg already at
/// that end goes first, so the body's mass stays about the same.
pub(crate) fn leg_to_dragging_end(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    let Some((working, dragging)) = drag_ends(c) else {
        return false;
    };
    let root = c.bones[working[0]].a as usize;
    let mut next = c.clone();
    let (mut leg, mut at) = (working[0], dragging.node);
    if let Some(weak) = dragging.weak_leg {
        let nodes = super::branch_nodes(c, &weak);
        remove_parts(&mut next, &weak, &nodes);
        leg -= weak.iter().filter(|&&b| b < leg).count();
        at -= nodes.iter().filter(|&&n| n < at).count();
    }
    let (from, to) = (c.nodes[root], next.nodes[at]);
    let place = |[x, y]: [f32; 2]| [to.x - (x - from.x), to.y + y - from.y];
    let phase = if rng.unit() < 0.5 { 0.5 } else { 0.0 };
    if copy_branch(&mut next, cfg, leg, at, place, true, phase).is_none() {
        return false;
    }
    passive_ring(&mut next, cfg, rng);
    *c = next;
    true
}

/// Adds a muscle across the joint of the leg at the dragging end, from that
/// leg's top bone to the bone above the joint, placed so its pull lifts the
/// foot and timed like the working leg's strongest muscle: the dragging end
/// lifts while the working leg pushes.
pub(crate) fn lift_dragging_end(
    c: &mut Creature,
    cfg: &Config,
    rng: &mut Rng,
    _cx: &Context,
) -> bool {
    if !room(c, cfg, 0, 1) {
        return false;
    }
    let Some((working, dragging)) = drag_ends(c) else {
        return false;
    };
    let Some(leg) = dragging.leg else {
        return false;
    };
    let top = leg[0];
    let joint = c.bones[top].a as usize;
    let Some(above) = parent_bones(c)[joint] else {
        return false;
    };
    let Some(&strongest) = muscles_on(c, &working, false)
        .iter()
        .max_by(|&&x, &&y| drive(&c.muscles[x]).total_cmp(&drive(&c.muscles[y])))
    else {
        return false;
    };
    // Turning the leg lifts its foot when the turn has the sign of the foot's
    // horizontal offset from the joint.
    let (j, foot) = (
        c.nodes[joint],
        c.nodes[c.bones[leg[leg.len() - 1]].b as usize],
    );
    let side = (foot.x - j.x).signum();
    let at = rng.range(0.5, 1.0);
    let p = crate::evolution::bone_point(c.bones[top], &c.nodes, at);
    let lift = |anchor: f32| {
        let q = crate::evolution::bone_point(c.bones[above], &c.nodes, anchor);
        let (dx, dy) = (q[0] - p[0], q[1] - p[1]);
        let length = dx.hypot(dy).max(1e-6);
        side * ((p[0] - j.x) * dy - (p[1] - j.y) * dx) / length
    };
    let best = [0.0, 0.25, 0.5, 0.75, 1.0]
        .into_iter()
        .max_by(|&x, &y| lift(x).total_cmp(&lift(y)))
        .expect("five anchors");
    if lift(best) < 1e-3 {
        return false;
    }
    let template = c.muscles[strongest];
    let m = new_muscle(c, top, above, (at, best), Some(&template), rng);
    c.muscles.push(m);
    true
}

/// The other end of the body from the working leg.
struct DraggingEnd {
    /// The node a leg copied there hangs from.
    node: usize,
    /// The leg already at that end, if any.
    leg: Option<BoneIds>,
    /// That leg, when its muscles drive less than a quarter of the working
    /// leg's: the copy replaces it.
    weak_leg: Option<BoneIds>,
}

/// The working leg (the leg, from `leaf_limbs`, whose muscles drive most)
/// and the other end of the body: the leg whose foot lies farthest from the
/// working leg's top joint along x in the rest pose, or without another leg
/// the node farthest along x (not the head, not in the working leg).
fn drag_ends(c: &Creature) -> Option<(BoneIds, DraggingEnd)> {
    let legs = leaf_limbs(c);
    let work = |leg: &BoneIds| -> f32 {
        muscles_on(c, leg, false)
            .iter()
            .map(|&i| drive(&c.muscles[i]))
            .sum()
    };
    let working = *legs.iter().max_by(|x, y| work(x).total_cmp(&work(y)))?;
    if work(&working) <= 0.0 {
        return None;
    }
    let x = c.nodes[c.bones[working[0]].a as usize].x;
    let away = |node: usize| (c.nodes[node].x - x).abs();
    let foot = |leg: &BoneIds| c.bones[leg[leg.len() - 1]].b as usize;
    let other = legs
        .iter()
        .filter(|leg| **leg != working)
        .max_by(|p, q| away(foot(p)).total_cmp(&away(foot(q))));
    let end = match other {
        Some(leg) => DraggingEnd {
            node: c.bones[leg[0]].a as usize,
            leg: Some(*leg),
            weak_leg: (work(leg) < 0.25 * work(&working)).then_some(*leg),
        },
        None => {
            let inside = super::branch_nodes(c, &working);
            let node = (1..c.nodes.len())
                .filter(|n| !inside.contains(n))
                .max_by(|&p, &q| away(p).total_cmp(&away(q)))?;
            DraggingEnd {
                node,
                leg: None,
                weak_leg: None,
            }
        }
    };
    Some((working, end))
}

/// A muscle's drive: stiffness times stroke. Zero for a passive muscle.
pub(super) fn drive(m: &Muscle) -> f32 {
    m.stiffness * (m.long - m.short)
}

/// Spring stiffness of the passive muscles `passive_ring` adds.
pub(super) const PASSIVE_STIFFNESS: f32 = 5.0;

/// Adds a passive muscle (random anchors, no stroke) on each pair of
/// consecutively numbered bones that has no muscle, as `repair` would
/// with an active one, while there is room.
fn passive_ring(c: &mut Creature, cfg: &Config, rng: &mut Rng) {
    let n = c.bones.len();
    for a in 0..n {
        let b = (a + 1) % n;
        let joined = c.muscles.iter().any(|m| {
            let ends = (m.bone_a as usize, m.bone_b as usize);
            ends == (a, b) || ends == (b, a)
        });
        if (n > 2 || a < b) && !joined && c.muscles.len() < cfg.max_muscles {
            let mut m = crate::evolution::muscle(a, b, &c.bones, &c.nodes, rng);
            m.short = m.long;
            m.stiffness = PASSIVE_STIFFNESS;
            c.muscles.push(m);
        }
    }
}

/// Pairs of limbs of the same shape (`matching_limbs`), each limb followed by
/// the bone above it, in both orders: (source, recipient).
struct Partners {
    limbs: Limbs,
    pairs: Bounded<(u8, u8), { MAX_NODES * MAX_NODES }>,
}
impl Partners {
    fn len(&self) -> usize {
        self.pairs.len()
    }
    fn get(&self, k: usize) -> (&BoneIds, &BoneIds) {
        let (x, y) = self.pairs[k];
        (&self.limbs[x as usize], &self.limbs[y as usize])
    }
    #[cfg(test)]
    fn iter(&self) -> impl Iterator<Item = (&BoneIds, &BoneIds)> {
        (0..self.len()).map(|k| self.get(k))
    }
}
#[cfg(test)]
impl IntoIterator for Partners {
    type Item = (BoneIds, BoneIds);
    type IntoIter = std::vec::IntoIter<(BoneIds, BoneIds)>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
            .map(|(x, y)| (*x, *y))
            .collect::<Vec<_>>()
            .into_iter()
    }
}
fn partners(c: &Creature) -> Partners {
    let matching = matching_limbs(c);
    let limbs: Limbs = matching
        .limbs
        .iter()
        .map(|limb| actuation(c, limb[0]).0)
        .collect();
    let pairs = matching
        .pairs
        .iter()
        .flat_map(|&(x, y)| [(x, y), (y, x)])
        .collect();
    Partners { limbs, pairs }
}

/// Pairs each muscle of limb `to` with an unpaired muscle of limb `from` on
/// the bones at the same positions (both limbs with the bone above them).
/// Returns (muscle of `from`, muscle of `to`).
fn counterparts(
    c: &Creature,
    from: &[usize],
    to: &[usize],
) -> Bounded<(usize, usize), MAX_MUSCLES> {
    let mut source = muscles_on(c, from, true);
    let mut out = Bounded::new();
    for q in muscles_on(c, to, true) {
        let m = c.muscles[q];
        let at = |b: u32| from[to.iter().position(|&x| x == b as usize).expect("limb bone")];
        let ends = (at(m.bone_a), at(m.bone_b));
        let found = source.iter().position(|&p| {
            let s = c.muscles[p];
            let (x, y) = (s.bone_a as usize, s.bone_b as usize);
            ends == (x, y) || ends == (y, x)
        });
        if let Some(k) = found {
            out.push((source.swap_remove(k), q));
        }
    }
    out
}

/// The matching muscles of a random pair of same-shaped limbs (source first),
/// among the pairs that have any.
fn pick_counterparts(c: &Creature, rng: &mut Rng) -> Option<Bounded<(usize, usize), MAX_MUSCLES>> {
    let partners = partners(c);
    let counterparts_of = |k: usize| {
        let (from, to) = partners.get(k);
        counterparts(c, from, to)
    };
    let options: Bounded<u16, { MAX_NODES * MAX_NODES }> = (0..partners.len())
        .filter(|&k| !counterparts_of(k).is_empty())
        .map(|k| k as u16)
        .collect();
    (!options.is_empty()).then(|| counterparts_of(options[rng.index(options.len())] as usize))
}

#[cfg(test)]
mod tests {
    use super::super::{Operator, tests::bodies};
    use super::*;

    /// Runs `op` on 160 grown bodies . A changed body must pass `check(before, after)`; an
    /// unchanged one must be as it was. Returns how many it changed.
    fn run(op: Operator, bodies: &[Creature], check: impl Fn(&Creature, &Creature)) -> usize {
        let cfg = Config::default();
        let mut applied = 0;
        for (i, body) in bodies.iter().enumerate() {
            let mut c = body.clone();
            let cx = Context { donor: None };
            if op(&mut c, &cfg, &mut Rng::new(31, 0, i), &cx) {
                applied += 1;
                assert!(c.nodes.len() <= cfg.max_nodes && c.muscles.len() <= cfg.max_muscles);
                check(body, &c);
            } else {
                assert!(
                    c.nodes == body.nodes && c.bones == body.bones && c.muscles == body.muscles
                );
            }
        }
        applied
    }

    fn grown() -> Vec<Creature> {
        bodies(&Config::default(), 160)
    }

    /// The test bodies with their first muscle-bearing limb twinned, so every
    /// one has a pair of same-shaped limbs with matching muscles.
    fn twinned() -> Vec<Creature> {
        twinned_at(0.0)
    }

    /// `twinned`, with the copy's muscles `phase` of a cycle later.
    fn twinned_at(phase: f32) -> Vec<Creature> {
        let cfg = Config::default();
        grown()
            .into_iter()
            .filter_map(|mut c| {
                let root = limb_roots(&c)
                    .into_iter()
                    .find(|&b| !muscles_on(&c, &actuation(&c, b).0, true).is_empty())?;
                let joint = c.bones[root].a as usize;
                copy_branch(&mut c, &cfg, root, joint, |p| p, false, phase)?;
                crate::evolution::repair(&mut c, &cfg, &mut Rng::new(3, 0, 0));
                Some(c)
            })
            .collect()
    }

    fn same_phase(a: f32, b: f32) -> bool {
        let d = (a - b).rem_euclid(1.0);
        !(1e-4..=1.0 - 1e-4).contains(&d)
    }

    fn ring_is_closed(c: &Creature) -> bool {
        let n = c.bones.len();
        (0..n).all(|a| {
            let b = (a + 1) % n;
            c.muscles.iter().any(|m| {
                let ends = (m.bone_a as usize, m.bone_b as usize);
                ends == (a, b) || ends == (b, a)
            })
        })
    }

    #[test]
    fn counterparts_pair_muscles_on_the_same_positions() {
        let bodies = twinned();
        assert!(bodies.len() > 100);
        for c in &bodies {
            for (from, to) in partners(c) {
                assert_eq!(from.len(), to.len());
                for (p, q) in counterparts(c, &from, &to) {
                    let (s, m) = (c.muscles[p], c.muscles[q]);
                    let at = |b: u32| to[from.iter().position(|&x| x == b as usize).unwrap()];
                    let mapped = (at(s.bone_a), at(s.bone_b));
                    let ends = (m.bone_a as usize, m.bone_b as usize);
                    assert!(mapped == ends || mapped == (ends.1, ends.0));
                }
            }
        }
    }

    #[test]
    fn mirror_limb_timing_runs_the_partner_half_a_cycle_later() {
        let bodies = twinned();
        let applied = run(mirror_limb_timing, &bodies, |before, after| {
            assert_eq!(before.bones, after.bones);
            let changed: Vec<usize> = (0..before.muscles.len())
                .filter(|&i| before.muscles[i] != after.muscles[i])
                .collect();
            assert!(!changed.is_empty());
            // Every changed muscle now runs half a cycle after a muscle on the
            // same positions of a same-shaped limb, with its duty.
            let fits = partners(before).iter().any(|(from, to)| {
                let pairs = counterparts(before, from, to);
                changed.iter().all(|&q| {
                    pairs.iter().any(|&(p, r)| {
                        let (s, m) = (after.muscles[p], after.muscles[r]);
                        r == q && m.duty == s.duty && same_phase(m.phase, s.phase + 0.5)
                    })
                })
            });
            assert!(fits);
            for (x, y) in before.muscles.iter().zip(&after.muscles) {
                let program = Muscle {
                    phase: x.phase,
                    duty: x.duty,
                    ..*y
                };
                assert_eq!(program, *x, "only phase and duty change");
            }
        });
        assert!(applied >= 100, "applied {applied} of {}", bodies.len());
    }

    #[test]
    fn swap_limb_programs_exchanges_two_programs() {
        let bodies = twinned_at(0.25);
        let applied = run(swap_limb_programs, &bodies, |before, after| {
            let program = |m: &Muscle| (m.phase.to_bits(), m.duty.to_bits());
            let mut old: Vec<_> = before.muscles.iter().map(program).collect();
            let mut new: Vec<_> = after.muscles.iter().map(program).collect();
            old.sort_unstable();
            new.sort_unstable();
            assert_eq!(old, new, "the same programs, exchanged");
            let swapped = partners(before).iter().any(|(from, to)| {
                counterparts(before, from, to).iter().all(|&(p, q)| {
                    program(&after.muscles[p]) == program(&before.muscles[q])
                        && program(&after.muscles[q]) == program(&before.muscles[p])
                })
            });
            assert!(swapped);
        });
        assert!(applied >= 100, "applied {applied} of {}", bodies.len());
    }

    #[test]
    fn copy_muscle_to_partner_adds_the_missing_muscle() {
        // Twins, and then one extra muscle on the source limb only.
        let bodies: Vec<Creature> = twinned()
            .into_iter()
            .enumerate()
            .filter_map(|(i, mut c)| {
                let (from, _) = partners(&c).into_iter().next()?;
                let m = new_muscle(
                    &c,
                    from[0],
                    from[from.len() - 1],
                    (0.5, 0.5),
                    None,
                    &mut Rng::new(4, 0, i),
                );
                c.muscles.push(m);
                Some(c)
            })
            .collect();
        let applied = run(copy_muscle_to_partner, &bodies, |before, after| {
            assert_eq!(after.muscles.len(), before.muscles.len() + 1);
            assert_eq!(&after.muscles[..before.muscles.len()], &before.muscles[..]);
            let copy = *after.muscles.last().unwrap();
            // The copy is the counterpart of a source muscle that had none.
            let found = partners(before).iter().any(|(from, to)| {
                let paired = counterparts(before, from, to);
                muscles_on(before, from, true).iter().any(|&p| {
                    let s = before.muscles[p];
                    let at =
                        |b: u32| to[from.iter().position(|&x| x == b as usize).unwrap()] as u32;
                    paired.iter().all(|&(x, _)| x != p)
                        && (copy.bone_a, copy.bone_b) == (at(s.bone_a), at(s.bone_b))
                        && (copy.anchor_a, copy.anchor_b) == (s.anchor_a, s.anchor_b)
                })
            });
            assert!(found);
        });
        assert!(applied >= 100, "applied {applied} of {}", bodies.len());
    }

    #[test]
    fn twin_limb_copies_a_limb_in_place_and_closes_the_ring() {
        let bodies = grown();
        let applied = run(twin_limb, &bodies, |before, after| {
            let added = after.bones.len() - before.bones.len();
            assert!(added >= 1);
            assert_eq!(after.nodes.len(), before.nodes.len() + added);
            assert_eq!(&after.bones[..before.bones.len()], &before.bones[..]);
            // The copy's nodes sit on top of the original's.
            let root = before.bones.len();
            let original = (0..root)
                .find(|&b| {
                    let (x, y) = (before.bones[b], after.bones[root]);
                    x.a == y.a && x.rest_length == y.rest_length && {
                        let (p, q) = (before.nodes[x.b as usize], after.nodes[y.b as usize]);
                        (p.x, p.y) == (q.x, q.y)
                    }
                })
                .expect("the copy starts at the original's joint");
            assert_eq!(branch(before, original).len(), added);
            assert!(ring_is_closed(after));
            // Only the copied muscles drive; the ring muscles start passive.
            let copied = muscles_on(before, &branch(before, original), false).len();
            let active = after.muscles[before.muscles.len()..]
                .iter()
                .filter(|m| m.short < m.long)
                .count();
            assert!(active <= copied);
        });
        assert!(applied >= 100, "applied {applied}");
    }

    #[test]
    fn grow_matching_tips_grows_the_same_tip_on_both_limbs() {
        let bodies = twinned();
        let applied = run(grow_matching_tips, &bodies, |before, after| {
            assert_eq!(after.nodes.len(), before.nodes.len() + 2);
            assert_eq!(after.bones.len(), before.bones.len() + 2);
            let n = before.bones.len();
            let (p, q) = (after.bones[n], after.bones[n + 1]);
            let tip = |node: u32| (0..n).find(|&b| before.bones[b].b == node).unwrap();
            let (x, y) = (tip(p.a), tip(q.a));
            assert!(child_bones(before)[p.a as usize].is_empty());
            assert!(child_bones(before)[q.a as usize].is_empty());
            let share = |toe: Bone, b: usize| toe.rest_length / before.bones[b].rest_length;
            if p.rest_length > 0.031 && q.rest_length > 0.031 {
                assert!((share(p, x) - share(q, y)).abs() < 1e-4);
            }
            let width = |b: Bone| b.max_angle - b.min_angle;
            assert!((width(p) - width(q)).abs() < 1e-5 && width(p) <= 1.0);
            // A muscle from each new bone to its tip bone.
            for (toe, bone) in [(n, x), (n + 1, y)] {
                assert!(
                    after
                        .muscles
                        .iter()
                        .any(|m| { (m.bone_a as usize, m.bone_b as usize) == (toe, bone) })
                );
            }
            assert!(ring_is_closed(after));
        });
        assert!(applied >= 80, "applied {applied} of {}", bodies.len());
    }

    #[test]
    fn nudge_limb_phase_moves_one_limb_by_a_small_step() {
        let bodies = grown();
        let applied = run(nudge_limb_phase, &bodies, |before, after| {
            assert_eq!(before.bones, after.bones);
            let changed: Vec<usize> = (0..before.muscles.len())
                .filter(|&i| before.muscles[i] != after.muscles[i])
                .collect();
            let steps: Vec<f32> = changed
                .iter()
                .map(|&i| {
                    let (x, y) = (before.muscles[i], after.muscles[i]);
                    assert!(same_phase(y.reset - x.reset, y.phase - x.phase));
                    (y.phase - x.phase + 0.5).rem_euclid(1.0) - 0.5
                })
                .collect();
            assert!(steps.iter().all(|s| (0.019..=0.121).contains(&s.abs())));
            assert!(steps.iter().all(|s| same_phase(*s, steps[0])));
            let one_limb = limb_roots(before)
                .into_iter()
                .any(|root| muscles_on(before, &branch(before, root), false) == changed);
            assert!(one_limb);
        });
        assert!(applied >= 150, "applied {applied}");
    }

    #[test]
    fn cadence_stride_trade_keeps_every_muscle_speed() {
        let bodies = grown();
        let applied = run(cadence_stride_trade, &bodies, |before, after| {
            let factor = after.muscles[0].period / before.muscles[0].period;
            assert!((0.69..=1.41).contains(&factor));
            for (x, y) in before.muscles.iter().zip(&after.muscles) {
                assert_eq!(x.long, y.long);
                assert!((y.period / x.period - factor).abs() < 1e-4);
                // Stroke over period stays unless the stroke hit its floor.
                if y.short > 0.0101 {
                    let speed = |m: &Muscle| (m.long - m.short) / m.period;
                    assert!((speed(x) - speed(y)).abs() <= 1e-4 * speed(x).max(1.0));
                }
            }
        });
        assert!(applied >= 140, "applied {applied}");
    }

    #[test]
    fn scale_muscle_leverage_moves_both_ends_along_their_bones() {
        let bodies = grown();
        let applied = run(scale_muscle_leverage, &bodies, |before, after| {
            let changed: Vec<usize> = (0..before.muscles.len())
                .filter(|&i| before.muscles[i] != after.muscles[i])
                .collect();
            assert_eq!(changed.len(), 1);
            let (x, y) = (before.muscles[changed[0]], after.muscles[changed[0]]);
            assert_eq!((x.bone_a, x.bone_b, x.phase), (y.bone_a, y.bone_b, y.phase));
            let node = shared_node(before, x.bone_a as usize, x.bone_b as usize).unwrap();
            let from_joint = |bone: u32, anchor: f32| {
                if before.bones[bone as usize].a == node {
                    anchor
                } else {
                    1.0 - anchor
                }
            };
            // Each end moves along its bone by one factor from the joint,
            // unless it sat on the joint or reached the far end.
            let ratios: Vec<f32> = [
                (x.bone_a, x.anchor_a, y.anchor_a),
                (x.bone_b, x.anchor_b, y.anchor_b),
            ]
            .iter()
            .filter_map(|&(bone, old, new)| {
                let (d, e) = (from_joint(bone, old), from_joint(bone, new));
                (d > 1e-3 && e < 1.0).then_some(e / d)
            })
            .collect();
            assert!(ratios.iter().all(|r| (0.49..=2.01).contains(r)));
            assert!(ratios.iter().all(|r| (r - ratios[0]).abs() < 1e-3));
            // The stroke keeps its ratios to the span.
            let ratio = |c: &Creature, m: &Muscle| m.long / super::super::span(c, m).max(0.05);
            assert!((ratio(before, &x) - ratio(after, &y)).abs() < 1e-3 * ratio(before, &x));
        });
        assert!(applied >= 150, "applied {applied}");
    }

    #[test]
    fn scale_limb_strength_scales_one_limb_by_one_factor() {
        let bodies = grown();
        let applied = run(scale_limb_strength, &bodies, |before, after| {
            let changed: Vec<usize> = (0..before.muscles.len())
                .filter(|&i| before.muscles[i] != after.muscles[i])
                .collect();
            let factors: Vec<f32> = changed
                .iter()
                .map(|&i| {
                    let (x, y) = (before.muscles[i], after.muscles[i]);
                    assert_eq!(
                        Muscle {
                            stiffness: x.stiffness,
                            ..y
                        },
                        x
                    );
                    y.stiffness / x.stiffness
                })
                .collect();
            let unclamped: Vec<f32> = changed
                .iter()
                .zip(&factors)
                .filter(|(i, _)| ![1.0, 120.0].contains(&after.muscles[**i].stiffness))
                .map(|(_, f)| *f)
                .collect();
            assert!(unclamped.iter().all(|f| (0.59..=1.61).contains(f)));
            assert!(unclamped.iter().all(|f| (f - unclamped[0]).abs() < 1e-3));
            let one_limb = limb_roots(before).into_iter().any(|root| {
                let on = muscles_on(before, &branch(before, root), false);
                changed.iter().all(|i| on.contains(i))
            });
            assert!(one_limb);
        });
        assert!(applied >= 150, "applied {applied}");
    }

    /// The share of `values` below `value`.
    fn rank(values: &[f32], value: f32) -> f32 {
        values.iter().filter(|v| **v < value).count() as f32 / values.len() as f32
    }

    #[test]
    fn prune_weakest_muscle_removes_a_weak_muscle_outside_the_ring() {
        let bodies = grown();
        let ranks = std::cell::RefCell::new(Vec::new());
        let applied = run(prune_weakest_muscle, &bodies, |before, after| {
            assert_eq!(after.muscles.len() + 1, before.muscles.len());
            let gone = (0..before.muscles.len())
                .find(|&i| after.muscles.get(i) != Some(&before.muscles[i]))
                .unwrap();
            assert!(!ring(before, &before.muscles[gone]));
            let free: Vec<f32> = before
                .muscles
                .iter()
                .filter(|m| !ring(before, m))
                .map(drive)
                .collect();
            ranks
                .borrow_mut()
                .push(rank(&free, drive(&before.muscles[gone])));
        });
        assert!(applied >= 80, "applied {applied}");
        // The weakest of three sits a quarter of the way up on average.
        let ranks = ranks.into_inner();
        let mean = ranks.iter().sum::<f32>() / ranks.len() as f32;
        assert!(mean < 0.35, "mean rank {mean}");
    }

    #[test]
    fn prune_idle_limb_removes_a_limb_tip() {
        let bodies = grown();
        let ranks = std::cell::RefCell::new(Vec::new());
        let applied = run(prune_idle_limb, &bodies, |before, after| {
            assert_eq!(after.nodes.len() + 1, before.nodes.len());
            assert_eq!(after.bones.len() + 1, before.bones.len());
            assert!(ring_is_closed(after));
            // The head and the neck stay.
            assert_eq!(after.nodes[0], before.nodes[0]);
            assert!(is_neck(after, 0));
            let work = |b: usize| -> f32 {
                muscles_on(before, &[b], false)
                    .iter()
                    .map(|&i| drive(&before.muscles[i]))
                    .sum()
            };
            let tips: Vec<usize> = (0..before.bones.len())
                .filter(|&b| !is_neck(before, b) && degree(before, before.bones[b].b as usize) == 1)
                .collect();
            let gone = tips
                .iter()
                .copied()
                .find(|&b| {
                    let foot = before.nodes[before.bones[b].b as usize];
                    after.nodes.iter().all(|n| (n.x, n.y) != (foot.x, foot.y))
                })
                .expect("a foot is gone");
            let all: Vec<f32> = tips.iter().map(|&b| work(b)).collect();
            if tips.len() > 1 {
                ranks.borrow_mut().push(rank(&all, work(gone)));
            }
        });
        assert!(applied >= 150, "applied {applied}");
        let ranks = ranks.into_inner();
        let mean = ranks.iter().sum::<f32>() / ranks.len() as f32;
        assert!(mean < 0.35, "mean rank {mean}");
    }

    #[test]
    fn merge_leaf_bones_joins_the_last_two_bones_of_a_limb() {
        let bodies = grown();
        let applied = run(merge_leaf_bones, &bodies, |before, after| {
            assert_eq!(after.nodes.len() + 1, before.nodes.len());
            assert_eq!(after.bones.len() + 1, before.bones.len());
            // Every foot of the parent is still a foot at the same place.
            let feet = |c: &Creature| {
                let mut out: Vec<(u32, u32)> = (1..c.nodes.len())
                    .filter(|&n| degree(c, n) == 1)
                    .map(|n| (c.nodes[n].x.to_bits(), c.nodes[n].y.to_bits()))
                    .collect();
                out.sort_unstable();
                out
            };
            assert_eq!(feet(before), feet(after));
            assert!(ring_is_closed(after));
        });
        assert!(applied >= 80, "applied {applied}");
    }

    #[test]
    fn leg_to_dragging_end_copies_the_working_leg_mirrored_to_the_other_end() {
        let bodies = grown();
        let (replaced, added) = (std::cell::Cell::new(0), std::cell::Cell::new(0));
        let applied = run(leg_to_dragging_end, &bodies, |before, after| {
            let (working, end) = drag_ends(before).unwrap();
            let gone = end.weak_leg.as_ref().map_or(0, |leg| leg.len());
            if gone > 0 {
                replaced.set(replaced.get() + 1);
            } else {
                added.set(added.get() + 1);
            }
            assert_eq!(after.nodes.len(), before.nodes.len() - gone + working.len());
            assert_eq!(after.bones.len(), before.bones.len() - gone + working.len());
            // The copy is appended: the working leg's bones, mirrored front to
            // back about its top joint and moved to the other end.
            let copy = &after.bones[after.bones.len() - working.len()..];
            let root = before.nodes[before.bones[working[0]].a as usize];
            let at = after.nodes[copy[0].a as usize];
            let extent = crate::evolution::body_extent();
            for (&b, new) in working.iter().zip(copy) {
                let old = before.bones[b];
                assert_eq!(old.rest_length, new.rest_length);
                assert_eq!(
                    (new.min_angle, new.max_angle),
                    (-old.max_angle, -old.min_angle)
                );
                let (p, q) = (before.nodes[old.b as usize], after.nodes[new.b as usize]);
                if q.x.abs() < extent - 1e-4 {
                    assert!((q.x - at.x + (p.x - root.x)).abs() < 1e-4);
                }
            }
            if gone == 0 {
                assert_eq!(&after.bones[..before.bones.len()], &before.bones[..]);
            }
        });
        assert!(applied >= 150, "applied {applied}");
        assert!(
            replaced.get() > 0 && added.get() > 0,
            "{replaced:?} {added:?}"
        );
    }

    #[test]
    fn lift_dragging_end_adds_a_lifting_muscle_timed_with_the_working_leg() {
        let bodies = grown();
        let applied = run(lift_dragging_end, &bodies, |before, after| {
            assert_eq!(after.muscles.len(), before.muscles.len() + 1);
            assert_eq!(&after.muscles[..before.muscles.len()], &before.muscles[..]);
            let m = *after.muscles.last().unwrap();
            let (working, end) = drag_ends(before).unwrap();
            let leg = end.leg.unwrap();
            let joint = before.bones[leg[0]].a as usize;
            assert_eq!(m.bone_a as usize, leg[0]);
            assert_eq!(Some(m.bone_b as usize), parent_bones(before)[joint]);
            // Timed like a working-leg muscle.
            assert!(muscles_on(before, &working, false).iter().any(|&i| {
                let w = before.muscles[i];
                (w.phase, w.duty, w.period) == (m.phase, m.duty, m.period)
            }));
            // Its pull turns the leg so the foot rises.
            let j = before.nodes[joint];
            let foot = before.nodes[before.bones[leg[leg.len() - 1]].b as usize];
            let p = crate::evolution::bone_point(before.bones[leg[0]], &before.nodes, m.anchor_a);
            let q = crate::evolution::bone_point(
                before.bones[m.bone_b as usize],
                &before.nodes,
                m.anchor_b,
            );
            let torque = (p[0] - j.x) * (q[1] - p[1]) - (p[1] - j.y) * (q[0] - p[0]);
            assert!(torque * (foot.x - j.x).signum() > 0.0);
        });
        assert!(applied >= 50, "applied {applied}");
    }
}
