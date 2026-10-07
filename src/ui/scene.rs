//! Painting a creature with the egui painter: bones, organs, muscles, force
//! arrows, nodes, break marks and the head's eye. `draw_creature` paints the
//! replay in the viewport and in the race lanes. `thumbnail` paints a body in
//! its start pose at a small size, for the cards and tiles of the tabs and the
//! "How evolution works" window.

use super::{playback::FrameMarks, widgets::mix_color};
use crate::{
    config::Config,
    evolution::{Bone, Creature},
    physics::{self, Node},
    theme::scene::{
        BONE, BONE_SHINE, EYE, FALLEN, FORCE_GROUND, FORCE_MUSCLE, MUSCLE_ACTIVE, MUSCLE_REST,
        MUSCLE_TIRED, NODE_GRIPPY, NODE_SLICK, ORGAN, OUTLINE, TOUCHDOWN,
    },
};
use eframe::egui::{self, Color32, Pos2, Rect, Stroke, Vec2};

/// A red cross over the node at `center`. It is as wide as the node's disc of
/// `radius` pixels, and at least 7 pixels wide. `draw_creature` draws it on
/// the nodes of a broken joint and on the head after the trial ended early.
fn draw_break_mark(p: &egui::Painter, center: Pos2, radius: f32) {
    let d = radius.max(3.5);
    let arm = |dx: f32, dy: f32| {
        p.line_segment(
            [
                center + Vec2::new(-dx * d, -dy * d),
                center + Vec2::new(dx * d, dy * d),
            ],
            Stroke::new(2.0, FALLEN),
        );
    };
    arm(1.0, 1.0);
    arm(1.0, -1.0);
}
/// The shell color of a node with this `friction`: slick steel blue at the
/// lowest friction a gene allows, brass at the highest. The range is that of
/// the default `Config`, read once. The GIF export uses this color too.
pub(super) fn node_color(friction: f32) -> Color32 {
    static RANGE: std::sync::OnceLock<(f32, f32)> = std::sync::OnceLock::new();
    let &(low, high) = RANGE.get_or_init(|| {
        let config = Config::default();
        (config.min_friction, config.max_friction)
    });
    mix_color(
        NODE_SLICK,
        NODE_GRIPPY,
        (friction - low) / (high - low).max(1e-3),
    )
}
/// An arrow from `from` along `delta`. It draws nothing when `delta` is
/// shorter than 3 pixels.
fn draw_arrow(p: &egui::Painter, from: Pos2, delta: Vec2, color: Color32) {
    if delta.length() < 3.0 {
        return;
    }
    let tip = from + delta;
    let dir = delta.normalized();
    let side = Vec2::new(-dir.y, dir.x) * 4.0;
    p.line_segment([from, tip], Stroke::new(2.5, color));
    p.add(egui::Shape::convex_polygon(
        vec![
            tip + dir * 3.0,
            tip - dir * 6.0 + side,
            tip - dir * 6.0 - side,
        ],
        color,
        Stroke::NONE,
    ));
}
/// Paints `c` with its nodes at the positions in `nodes`. A world point
/// `(x, y)` in meters lands at `origin + (x, -y) * scale` on the screen, so
/// `origin` is the world origin and `scale` is pixels per meter. The layers
/// from the back are the bones, the organs, the muscles with their force
/// arrows, the ground force arrows, the nodes with their contact and break
/// marks, and the head's eye. `marks` holds what the current frame shows.
pub(super) fn draw_creature(
    p: &egui::Painter,
    nodes: &[Node],
    c: &Creature,
    origin: Pos2,
    scale: f32,
    marks: &FrameMarks,
) {
    let position = |n: &Node| origin + Vec2::new(n.pos[0] * scale, -n.pos[1] * scale);
    // The screen point a share `t` of the way along `bone`, from its node `a`
    // to its node `b`.
    let bone_point = |bone: &Bone, t: f32| {
        let a = nodes[bone.a as usize].pos;
        let b = nodes[bone.b as usize].pos;
        origin
            + Vec2::new(
                (a[0] + (b[0] - a[0]) * t) * scale,
                -(a[1] + (b[1] - a[1]) * t) * scale,
            )
    };
    let sphere = crate::assets::Art::Sphere.texture(p.ctx());
    for bone in &c.bones {
        let a = position(&nodes[bone.a as usize]);
        let b = position(&nodes[bone.b as usize]);
        let width = (scale * 0.032).max(3.0);
        p.line_segment([a, b], Stroke::new(width + 3.0, OUTLINE));
        p.line_segment([a, b], Stroke::new(width, BONE));
        // A steel rod: a bright streak along its upper side.
        let across = (b - a).normalized().rot90();
        if across.x.is_finite() {
            let lift = across * (if across.y > 0.0 { -1.0 } else { 1.0 }) * width * 0.22;
            p.line_segment(
                [a + lift, b + lift],
                Stroke::new((width * 0.28).max(1.0), BONE_SHINE),
            );
        }
    }
    // An organ sits on its bone and is drawn as large as a node of the same
    // mass would be.
    for bone in c.bones.iter().filter(|b| b.organ_mass > 0.0) {
        let center = bone_point(bone, bone.organ_at);
        let r = (0.04 * (bone.organ_mass / 0.1).sqrt() * scale).max(2.5);
        p.circle_filled(center, r + 1.5, OUTLINE);
        p.circle_filled(center, r, ORGAN);
        p.circle_filled(
            center + Vec2::new(-r * 0.25, -r * 0.3),
            r * 0.4,
            Color32::from_white_alpha(40),
        );
    }
    for (mi, m) in c.muscles.iter().enumerate() {
        let a = bone_point(&c.bones[m.bone_a as usize], m.anchor_a);
        let b = bone_point(&c.bones[m.bone_b as usize], m.anchor_b);
        // How far the waveform has pulled the muscle in: 0 at its longest
        // length and 1 at its shortest. A fallen creature's muscles are limp.
        let contraction = if marks.fallen {
            0.
        } else {
            1. - ((physics::target(m, marks.time) - m.short) / (m.long - m.short).max(1e-5))
        };
        // A contracting muscle bulges. A tired muscle thins and goes grey.
        let energy = marks.energy.get(mi).copied().unwrap_or(1.0).clamp(0.0, 1.0);
        let width = (scale * 0.017 * (1. + 0.45 * contraction) * (0.45 + 0.55 * energy)).max(2.);
        p.line_segment([a, b], Stroke::new(width + 3., OUTLINE));
        // Pale flesh at rest, deep red at full contraction, grey when spent.
        let flesh = mix_color(
            MUSCLE_TIRED,
            mix_color(MUSCLE_REST, MUSCLE_ACTIVE, contraction),
            energy,
        );
        p.line_segment([a, b], Stroke::new(width, flesh));
        // A pale sheen along the muscle, when it is thick enough to show one.
        let across = (b - a).normalized().rot90();
        if across.x.is_finite() && width > 3.0 {
            let lift = across * (if across.y > 0.0 { -1.0 } else { 1.0 }) * width * 0.2;
            p.line_segment(
                [a + lift, b + lift],
                Stroke::new(width * 0.25, mix_color(flesh, Color32::WHITE, 0.35)),
            );
        }
        // Force arrows at both ends of the muscle. Each points along the muscle
        // toward the other end for a pull, and a negative force flips them. A
        // force of 100 N or more gives the longest arrow.
        if marks.arrows
            && let Some(&force) = marks.muscle_force.get(mi)
        {
            let along = (b - a).normalized();
            if along.x.is_finite() {
                let len = (force / 100.0).abs().min(1.0) * scale * 0.3;
                let sign = if force >= 0. { 1. } else { -1. };
                draw_arrow(p, a, along * sign * len, FORCE_MUSCLE);
                draw_arrow(p, b, -along * sign * len, FORCE_MUSCLE);
            }
        }
    }
    // Ground pushes: an arrow under each pushed node, pointing up into it. A
    // push of two body weights or more gives the longest arrow.
    if marks.arrows {
        let weight: f32 = nodes.iter().map(|n| n.mass).sum::<f32>() * 9.81;
        for (i, n) in nodes.iter().enumerate() {
            let push = marks.ground_force.get(i).copied().unwrap_or(0.0);
            if push > 0.0 {
                let len = (push / weight.max(1e-3)).min(2.0) * scale * 0.6;
                let foot = position(n) + Vec2::new(0., n.radius * scale);
                draw_arrow(
                    p,
                    foot + Vec2::new(0., len),
                    Vec2::new(0., -len),
                    FORCE_GROUND,
                );
            }
        }
    }
    for (i, n) in nodes.iter().enumerate() {
        let center = position(n);
        let r = (n.radius * scale).max(2.);
        // A lit metal ball: the shaded sphere sprite tinted by friction.
        p.circle_filled(center, r + 1.5, OUTLINE);
        p.image(
            sphere,
            Rect::from_center_size(center, Vec2::splat(r * 2.0 + 0.5)),
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            node_color(n.friction),
        );
        if marks.contact.get(i).copied().unwrap_or(false) {
            p.circle_stroke(center, r + 2.5, Stroke::new(2., TOUCHDOWN));
        }
        if marks.broken.get(i).copied().unwrap_or(false) {
            p.circle_stroke(center, r + 2.5, Stroke::new(2., FALLEN));
            draw_break_mark(p, center, r);
        }
    }
    // The head (node 0) carries the fallen mark and looks ahead with one eye.
    if let Some(head) = nodes.first() {
        let center = position(head);
        let r = (head.radius * scale).max(2.);
        if marks.fallen {
            p.circle_stroke(center, r + 1.5, Stroke::new(2., FALLEN));
            draw_break_mark(p, center, r);
        }
        let eye = center + Vec2::new(r * 0.4, -r * 0.2);
        p.image(
            crate::assets::Art::Glow.texture(p.ctx()),
            Rect::from_center_size(eye, Vec2::splat(r * 1.8)),
            Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
            Color32::from_rgba_unmultiplied(EYE.r(), EYE.g(), EYE.b(), 120),
        );
        p.circle_filled(eye, r * 0.3, EYE);
        p.circle_filled(eye + Vec2::new(r * 0.08, 0.), r * 0.15, OUTLINE);
    }
}
/// Paints `c` in its start pose, centered in `rect` and scaled so the body
/// fills 82% of the tighter side. It draws no contact rings, break marks or
/// arrows, and every muscle is rested.
pub(crate) fn thumbnail(p: &egui::Painter, c: &Creature, rect: Rect) {
    let nodes = physics::nodes(c);
    let minx = nodes
        .iter()
        .map(|n| n.pos[0] - n.radius)
        .fold(f32::INFINITY, f32::min);
    let maxx = nodes
        .iter()
        .map(|n| n.pos[0] + n.radius)
        .fold(f32::NEG_INFINITY, f32::max);
    let miny = nodes
        .iter()
        .map(|n| n.pos[1] - n.radius)
        .fold(f32::INFINITY, f32::min);
    let maxy = nodes
        .iter()
        .map(|n| n.pos[1] + n.radius)
        .fold(f32::NEG_INFINITY, f32::max);
    let scale =
        (rect.width() / (maxx - minx).max(0.1)).min(rect.height() / (maxy - miny).max(0.1)) * 0.82;
    let origin =
        rect.center() + Vec2::new(-(minx + maxx) * 0.5 * scale, (miny + maxy) * 0.5 * scale);
    draw_creature(p, &nodes, c, origin, scale, &FrameMarks::default());
}
