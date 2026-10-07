//! Painting a creature: bones, nodes, muscles, force arrows and break marks,
//! and the small thumbnails of the archive cards.

use super::{playback::FrameMarks, widgets::mix_color};
use crate::{
    config::Config,
    evolution::Creature,
    physics::{self, Node},
    theme::scene::{
        BONE, EYE, FALLEN, FORCE_GROUND, FORCE_MUSCLE, MUSCLE_ACTIVE, MUSCLE_REST, MUSCLE_TIRED,
        NODE_GRIPPY, NODE_SLICK, ORGAN, OUTLINE, TOUCHDOWN,
    },
};
use eframe::egui::{self, Color32, Pos2, Rect, Stroke, Vec2};

/// A red cross over a node that fell, shook, or broke its joint.
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
/// A node's shell: slick steel blue at the lowest friction a gene allows,
/// brass at the highest.
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
/// An arrow starting at `from` and pointing along `delta`.
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
/// Renders a creature with bones, nodes, muscles, organs, forces and status marks.
pub(super) fn draw_creature(
    p: &egui::Painter,
    nodes: &[Node],
    c: &Creature,
    origin: Pos2,
    scale: f32,
    marks: &FrameMarks,
) {
    let position = |n: &Node| origin + Vec2::new(n.pos[0] * scale, -n.pos[1] * scale);
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
                Stroke::new((width * 0.28).max(1.0), crate::theme::scene::BONE_SHINE),
            );
        }
    }
    // Organs ride on their bones; drawn with the density of a node.
    for bone in c.bones.iter().filter(|b| b.organ_mass > 0.0) {
        let a = nodes[bone.a as usize].pos;
        let b = nodes[bone.b as usize].pos;
        let t = bone.organ_at;
        let center = origin
            + Vec2::new(
                (a[0] + (b[0] - a[0]) * t) * scale,
                -(a[1] + (b[1] - a[1]) * t) * scale,
            );
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
        let bone_a = c.bones[m.bone_a as usize];
        let bone_b = c.bones[m.bone_b as usize];
        let point = |bone: crate::evolution::Bone, t: f32| {
            let a = [nodes[bone.a as usize].pos[0], nodes[bone.a as usize].pos[1]];
            let b = [nodes[bone.b as usize].pos[0], nodes[bone.b as usize].pos[1]];
            origin
                + Vec2::new(
                    (a[0] + (b[0] - a[0]) * t) * scale,
                    -(a[1] + (b[1] - a[1]) * t) * scale,
                )
        };
        let a = point(bone_a, m.anchor_a);
        let b = point(bone_b, m.anchor_b);
        // A fallen creature's muscles are limp.
        let contraction = if marks.fallen {
            0.
        } else {
            1. - ((physics::target(m, marks.time) - m.short) / (m.long - m.short).max(1e-5))
        };
        // A tired muscle thins and goes grey.
        let energy = marks.energy.get(mi).copied().unwrap_or(1.0).clamp(0.0, 1.0);
        let width = (scale * 0.017 * (1. + 0.45 * contraction) * (0.45 + 0.55 * energy)).max(2.);
        p.line_segment([a, b], Stroke::new(width + 3., OUTLINE));
        // Pale flesh at rest, deep red at full contraction, grey when spent,
        // with a wet sheen along the fibre.
        let flesh = mix_color(
            MUSCLE_TIRED,
            mix_color(MUSCLE_REST, MUSCLE_ACTIVE, contraction),
            energy,
        );
        p.line_segment([a, b], Stroke::new(width, flesh));
        let across = (b - a).normalized().rot90();
        if across.x.is_finite() && width > 3.0 {
            let lift = across * (if across.y > 0.0 { -1.0 } else { 1.0 }) * width * 0.2;
            p.line_segment(
                [a + lift, b + lift],
                Stroke::new(width * 0.25, mix_color(flesh, Color32::WHITE, 0.35)),
            );
        }
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
    // The head (node 0) looks ahead with one eye.
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
/// Renders a creature as a small thumbnail centered in `rect`.
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
