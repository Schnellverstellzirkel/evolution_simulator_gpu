//! The picture exports: the creature GIF and the window screenshot. The GIF is
//! painted on the CPU, one pixel buffer for each recorded frame, by a small
//! rasterizer that copies the viewport's scene. The screenshot saves the image
//! that egui captured from the window as a PNG. `dialogs` calls the GIF export
//! and `ui.rs` calls the screenshot save.

use super::{
    playback::{Playback, broken_nodes, node_contact},
    scene::node_color,
    widgets::mix_color,
};
use crate::{
    config::Config,
    evolution::Creature,
    physics::{self, Node},
    theme::scene::{
        BONE, EYE, FALLEN, GROUND_EDGE, GROUND_TOP, MUSCLE_ACTIVE, MUSCLE_REST, ORGAN, OUTLINE,
        SKY_HORIZON, SKY_TOP, TOUCHDOWN,
    },
};
use eframe::egui::{self, Color32};
use image::{
    Delay as GifDelay, Frame as GifFrame, Rgba, RgbaImage,
    codecs::gif::{GifEncoder, Repeat as GifRepeat},
};
use std::path::PathBuf;

/// Width of an exported GIF frame in pixels.
const GIF_WIDTH: u32 = 400;
/// Height of an exported GIF frame in pixels.
const GIF_HEIGHT: u32 = 224;
/// Most frames an exported GIF keeps. A longer trial is sampled evenly.
const GIF_MAX_FRAMES: usize = 360;
/// Most pixels per meter the GIF camera zooms to. A small creature is shown at
/// this zoom and does not fill the frame.
const GIF_MAX_SCALE: f32 = 200.0;
/// The camera of an exported GIF. Its zoom and bottom edge are fitted once to
/// all the frames and stay fixed. Sideways it follows the creature, which stays
/// at `anchor_x`.
struct GifCamera {
    /// World y at the bottom edge.
    y0: f32,
    /// Pixels per meter.
    scale: f32,
    /// Screen x the followed center of mass stays at, in pixels.
    anchor_x: f32,
}
impl GifCamera {
    /// Fits the zoom and the bottom edge to the lowest and highest node edges
    /// in `frames`, plus a margin. The range always spans at least y = -0.15 m
    /// to 0.4 m, so the flat ground stays in view.
    fn fit<'a>(nodes: &[Node], frames: impl Iterator<Item = &'a Vec<[f32; 2]>>) -> Self {
        let mut min_y = 0.0f32;
        let mut max_y = 0.4f32;
        for frame in frames {
            for (position, node) in frame.iter().zip(nodes) {
                min_y = min_y.min(position[1] - node.radius);
                max_y = max_y.max(position[1] + node.radius);
            }
        }
        min_y -= 0.15;
        let range = (max_y - min_y).max(0.4);
        let scale = (GIF_HEIGHT as f32 / (range * 1.15)).min(GIF_MAX_SCALE);
        let y0 = 0.5 * (min_y + max_y) - 0.5 * GIF_HEIGHT as f32 / scale;
        Self {
            y0,
            scale,
            anchor_x: GIF_WIDTH as f32 * 0.38,
        }
    }
    /// The world x at the left edge of the frame that puts `center_x` at
    /// `anchor_x`.
    fn origin_x(&self, center_x: f32) -> f32 {
        center_x - self.anchor_x / self.scale
    }
    /// Screen position in pixels for a world position. `origin_x` is the world
    /// x at the left edge, and screen y grows downward.
    fn screen(&self, origin_x: f32, position: [f32; 2]) -> (f32, f32) {
        (
            (position[0] - origin_x) * self.scale,
            GIF_HEIGHT as f32 - (position[1] - self.y0) * self.scale,
        )
    }
}
/// The viewport scene painted into a pixel buffer, one recorded pose at a time.
/// It holds what stays the same for every frame of one GIF. It leaves out the
/// force arrows and the tired-muscle fade that the viewport can show.
struct GifScene<'a> {
    creature: &'a Creature,
    /// The world the trial ran in. It decides the ground.
    config: &'a Config,
    /// The creature's nodes for their radius, mass and friction. Their
    /// positions are those of each recorded frame.
    nodes: &'a [Node],
    camera: &'a GifCamera,
}
impl GifScene<'_> {
    /// Paints one pose over every pixel of `buffer`. `positions` holds one
    /// entry for each node. `time` is the trial time in seconds that drives the
    /// muscles, and `fallen` makes them limp. `contact` marks the nodes on the
    /// ground and `broken` marks the nodes at the ends of a bone with a broken
    /// joint.
    fn render(
        &self,
        buffer: &mut RgbaImage,
        positions: &[[f32; 2]],
        time: f32,
        fallen: bool,
        contact: &[bool],
        broken: &[bool],
    ) {
        let dark = gif_color(OUTLINE);
        let origin_x = self.camera.origin_x(pose_center_x(self.nodes, positions));
        let at = |position: [f32; 2]| self.camera.screen(origin_x, position);
        // The overcast sky: `SKY_TOP` at the top edge, fading to the
        // `SKY_HORIZON` haze at four fifths of the height.
        for (_, y, pixel) in buffer.enumerate_pixels_mut() {
            *pixel = gif_color(mix_color(
                SKY_TOP,
                SKY_HORIZON,
                y as f32 / (GIF_HEIGHT as f32 * 0.8),
            ));
        }
        // A grid line at every whole meter of world x. The grid scrolls with the
        // follow camera, so motion reads even when the creature holds its
        // screen position.
        let right = origin_x + GIF_WIDTH as f32 / self.camera.scale;
        let grid = gif_color(Color32::from_rgb(104, 110, 112));
        for meter in origin_x.floor() as i32..=right.ceil() as i32 {
            let screen_x = (meter as f32 - origin_x) * self.camera.scale;
            gif_line(
                buffer,
                (screen_x, 0.0),
                (screen_x, GIF_HEIGHT as f32),
                0.5,
                grid,
            );
        }
        // The ground, when the world has one, one pixel column at a time with
        // a two pixel lip along its surface. The earthquake phase and strength
        // come from the creature's id, so the ground matches the one the
        // kernel scored.
        if self.config.ground {
            let hash = physics::quake_hash(self.creature.id);
            let amplitude = physics::terrain_amplitude(self.config.terrain)
                + self.config.quake * physics::quake_scale(hash);
            let phase = if self.config.quake > 0.0 {
                physics::quake_phase(hash)
            } else {
                0.0
            };
            let ground = gif_color(GROUND_TOP);
            let edge = gif_color(GROUND_EDGE);
            for px in 0..GIF_WIDTH {
                let world_x = origin_x + px as f32 / self.camera.scale;
                let (height, _) = physics::ground(
                    world_x,
                    amplitude,
                    self.config.slope,
                    self.config.gaps,
                    self.config.hurdles,
                    phase,
                );
                let surface = GIF_HEIGHT as f32 - (height - self.camera.y0) * self.camera.scale;
                let top = surface.floor().max(0.0) as u32;
                for y in top..GIF_HEIGHT {
                    buffer.put_pixel(px, y, ground);
                }
                if top < GIF_HEIGHT {
                    buffer.put_pixel(px, top, edge);
                    if top + 1 < GIF_HEIGHT {
                        buffer.put_pixel(px, top + 1, edge);
                    }
                }
            }
        }
        // A bone is a dark outline with the bone color on top. `half` is half
        // its width in pixels.
        for bone in &self.creature.bones {
            let a = at(positions[bone.a as usize]);
            let b = at(positions[bone.b as usize]);
            let half = (self.camera.scale * 0.032).max(3.0) * 0.5;
            gif_line(buffer, a, b, half + 1.5, dark);
            gif_line(buffer, a, b, half, gif_color(BONE));
        }
        // Organs ride on their bones, `organ_at` of the way from the first node
        // to the second.
        for bone in self.creature.bones.iter().filter(|b| b.organ_mass > 0.0) {
            let a = positions[bone.a as usize];
            let b = positions[bone.b as usize];
            let t = bone.organ_at;
            let center = at([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]);
            let r = (0.04 * (bone.organ_mass / 0.1).sqrt() * self.camera.scale).max(2.5);
            gif_disc(buffer, center, r + 1.5, dark);
            gif_disc(buffer, center, r, gif_color(ORGAN));
        }
        // A muscle joins a point on one bone to a point on another. As it
        // contracts it gets thicker and goes from pale flesh to deep red.
        for m in &self.creature.muscles {
            let bone_a = self.creature.bones[m.bone_a as usize];
            let bone_b = self.creature.bones[m.bone_b as usize];
            let point = |bone: crate::evolution::Bone, t: f32| {
                let a = positions[bone.a as usize];
                let b = positions[bone.b as usize];
                at([a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t])
            };
            let a = point(bone_a, m.anchor_a);
            let b = point(bone_b, m.anchor_b);
            // Contraction is 0 at the long length and 1 at the short length. A
            // fallen creature's muscles are limp.
            let contraction = if fallen {
                0.0
            } else {
                1. - ((physics::target(m, time) - m.short) / (m.long - m.short).max(1e-5))
            };
            let half = (self.camera.scale * 0.017 * (1. + 0.45 * contraction)).max(2.) * 0.5;
            gif_line(buffer, a, b, half + 1.5, dark);
            gif_line(
                buffer,
                a,
                b,
                half,
                gif_color(mix_color(MUSCLE_REST, MUSCLE_ACTIVE, contraction)),
            );
        }
        // A node is an outlined disc in its friction color. A ring marks a node
        // on the ground, and a red ring with a cross marks a broken joint.
        for (i, n) in self.nodes.iter().enumerate() {
            let center = at(positions[i]);
            let r = (n.radius * self.camera.scale).max(2.);
            gif_disc(buffer, center, r + 1.5, dark);
            gif_disc(buffer, center, r, gif_color(node_color(n.friction)));
            if contact.get(i).copied().unwrap_or(false) {
                gif_ring(buffer, center, r + 2.5, 2.0, gif_color(TOUCHDOWN));
            }
            if broken.get(i).copied().unwrap_or(false) {
                gif_ring(buffer, center, r + 2.5, 2.0, gif_color(FALLEN));
                gif_cross(buffer, center, r, gif_color(FALLEN));
            }
        }
        // The head (node 0) looks ahead with one eye. A fallen creature's head
        // carries the damage mark.
        if let Some(head) = self.nodes.first() {
            let center = at(positions[0]);
            let r = (head.radius * self.camera.scale).max(2.);
            gif_disc(
                buffer,
                (center.0 + r * 0.4, center.1 - r * 0.2),
                r * 0.3,
                gif_color(EYE),
            );
            gif_disc(
                buffer,
                (center.0 + r * 0.48, center.1 - r * 0.2),
                r * 0.15,
                dark,
            );
            if fallen {
                gif_ring(buffer, center, r + 1.5, 2.0, gif_color(FALLEN));
                gif_cross(buffer, center, r, gif_color(FALLEN));
            }
        }
    }
}
/// Saves a window capture as `screenshot-<n>.png` in `dir`, where `n` is the
/// time in milliseconds since the Unix epoch. It creates `dir` first and
/// returns the path of the file.
pub(super) fn save_screenshot(
    capture: &egui::ColorImage,
    dir: &std::path::Path,
) -> anyhow::Result<PathBuf> {
    std::fs::create_dir_all(dir)?;
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_millis());
    let path = dir.join(format!("screenshot-{stamp}.png"));
    let bytes: Vec<u8> = capture.pixels.iter().flat_map(|p| p.to_array()).collect();
    image::save_buffer(
        &path,
        &bytes,
        capture.size[0] as u32,
        capture.size[1] as u32,
        image::ColorType::Rgba8,
    )?;
    Ok(path)
}
/// The fully opaque pixel of a color.
fn gif_color(c: Color32) -> Rgba<u8> {
    Rgba([c.r(), c.g(), c.b(), 255])
}
/// Sets the pixel nearest to (`x`, `y`) and skips a position outside the
/// buffer.
fn gif_put(buffer: &mut RgbaImage, x: f32, y: f32, color: Rgba<u8>) {
    let x = x.round() as i32;
    let y = y.round() as i32;
    if x >= 0 && y >= 0 && (x as u32) < buffer.width() && (y as u32) < buffer.height() {
        buffer.put_pixel(x as u32, y as u32, color);
    }
}
/// Fills a disc of `radius` pixels around `center`. A radius under half a pixel
/// counts as half a pixel.
fn gif_disc(buffer: &mut RgbaImage, center: (f32, f32), radius: f32, color: Rgba<u8>) {
    let radius = radius.max(0.5);
    let r = radius.ceil() as i32;
    let r2 = radius * radius;
    for dy in -r..=r {
        for dx in -r..=r {
            if (dx * dx + dy * dy) as f32 <= r2 {
                gif_put(buffer, center.0 + dx as f32, center.1 + dy as f32, color);
            }
        }
    }
}
/// Fills a ring around `center`. `radius` is the middle of the ring and `width`
/// is how thick it is.
fn gif_ring(buffer: &mut RgbaImage, center: (f32, f32), radius: f32, width: f32, color: Rgba<u8>) {
    let outer = radius + width * 0.5;
    let inner = (radius - width * 0.5).max(0.0);
    let r = outer.ceil() as i32;
    let (outer2, inner2) = (outer * outer, inner * inner);
    for dy in -r..=r {
        for dx in -r..=r {
            let d2 = (dx * dx + dy * dy) as f32;
            if d2 <= outer2 && d2 >= inner2 {
                gif_put(buffer, center.0 + dx as f32, center.1 + dy as f32, color);
            }
        }
    }
}
/// Draws a line from `a` to `b` as a chain of discs of `radius`, so the line is
/// twice `radius` thick.
fn gif_line(buffer: &mut RgbaImage, a: (f32, f32), b: (f32, f32), radius: f32, color: Rgba<u8>) {
    let radius = radius.max(0.5);
    let length = (b.0 - a.0).hypot(b.1 - a.1);
    let steps = (length / (radius * 0.5).max(1.0)).ceil().max(1.0) as u32;
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        gif_disc(
            buffer,
            (a.0 + (b.0 - a.0) * t, a.1 + (b.1 - a.1) * t),
            radius,
            color,
        );
    }
}
/// Draws an X around `center`. Its corners are `radius` pixels from the center
/// along both axes, or 3.5 pixels if `radius` is smaller.
fn gif_cross(buffer: &mut RgbaImage, center: (f32, f32), radius: f32, color: Rgba<u8>) {
    let d = radius.max(3.5);
    gif_line(
        buffer,
        (center.0 - d, center.1 - d),
        (center.0 + d, center.1 + d),
        1.25,
        color,
    );
    gif_line(
        buffer,
        (center.0 - d, center.1 + d),
        (center.0 + d, center.1 - d),
        1.25,
        color,
    );
}
/// Mass-weighted center x of a pose. The GIF camera follows it. It is 0 when
/// the nodes have no mass.
fn pose_center_x(nodes: &[Node], positions: &[[f32; 2]]) -> f32 {
    let mut mass = 0.0;
    let mut x = 0.0;
    for (node, position) in nodes.iter().zip(positions) {
        mass += node.mass;
        x += node.mass * position[0];
    }
    if mass > 0.0 { x / mass } else { 0.0 }
}
/// Writes an animated GIF of one recorded trial and returns the frame count.
/// `frames` holds the node positions of each recorded frame, and `ticks` are
/// the indices of the frames to show, in increasing order. `broken_joints` holds
/// the broken joint bits of each frame (`replay_forces::Forces::broken`). `fall`
/// is the frame where the trial ended and the distance it kept. From that frame
/// on the creature shows as fallen. The frame delay follows the average spacing
/// of `ticks`, so the GIF plays at the speed the trial was simulated. The delay
/// is rounded to hundredths of a second and kept between 0.02 s and 2 s, and a
/// GIF of one frame waits 0.1 s. A tick past the last frame ends the GIF there.
fn write_creature_gif(
    creature: &Creature,
    config: &Config,
    frames: &[Vec<[f32; 2]>],
    broken_joints: &[u64],
    ticks: &[u32],
    fall: Option<(u32, f32)>,
    path: &std::path::Path,
) -> anyhow::Result<usize> {
    let nodes = &physics::nodes(creature);
    // The camera fits the frames the GIF shows.
    let camera = GifCamera::fit(
        nodes,
        ticks.iter().filter_map(|&tick| frames.get(tick as usize)),
    );
    let scene = GifScene {
        creature,
        config,
        nodes,
        camera: &camera,
    };
    let hundredths = if ticks.len() > 1 {
        let span = ticks[ticks.len() - 1].saturating_sub(ticks[0]) as f32;
        let mean = span / (ticks.len() - 1) as f32;
        ((mean * physics::dt() * 100.0).round() as u32).clamp(2, 200)
    } else {
        10
    };
    let delay = GifDelay::from_numer_denom_ms(hundredths * 10, 1);
    let file = std::io::BufWriter::new(std::fs::File::create(path)?);
    let mut encoder = GifEncoder::new_with_speed(file, 30);
    encoder.set_repeat(GifRepeat::Infinite)?;
    let mut buffer = RgbaImage::new(GIF_WIDTH, GIF_HEIGHT);
    let mut contact = vec![false; nodes.len()];
    let mut broken = vec![false; nodes.len()];
    let mut written = 0usize;
    for &tick in ticks {
        let Some(frame) = frames.get(tick as usize) else {
            break;
        };
        node_contact(nodes, frame, creature, config, &mut contact);
        let bits = broken_joints.get(tick as usize).copied().unwrap_or(0);
        broken_nodes(creature, bits, &mut broken);
        let time = tick.saturating_sub(physics::settle()) as f32 * physics::dt();
        let fallen = fall.is_some_and(|(fall_tick, _)| tick >= fall_tick);
        scene.render(&mut buffer, frame, time, fallen, &contact, &broken);
        encoder.encode_frame(GifFrame::from_parts(buffer.clone(), 0, 0, delay))?;
        written += 1;
    }
    // Dropping the encoder writes the GIF trailer and flushes the writer.
    drop(encoder);
    Ok(written)
}
/// Writes the GIF of a playback's trial to `path` and returns the frame count.
/// It runs from the start of the trial to the last recorded frame and keeps
/// every n-th frame, with the smallest n that leaves at most `GIF_MAX_FRAMES`
/// frames.
pub(super) fn export_creature_gif(
    playback: &Playback,
    path: &std::path::Path,
) -> anyhow::Result<usize> {
    let first = playback.trial_start();
    let last = playback.last_frame();
    let total = last.saturating_sub(first) as usize + 1;
    let stride = total.div_ceil(GIF_MAX_FRAMES).max(1);
    let ticks: Vec<u32> = (first..=last).step_by(stride).collect();
    write_creature_gif(
        &playback.creature,
        &playback.config,
        &playback.frames,
        &playback.forces.broken,
        &ticks,
        playback.fall,
        path,
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::test_support::test_creature;
    #[test]
    fn gif_export_encodes_three_synthetic_frames() {
        use image::AnimationDecoder;
        use image::codecs::gif::GifDecoder;
        let creature = test_creature();
        let config = Config::default();
        let frames: Vec<Vec<[f32; 2]>> = vec![
            vec![[0.0, 0.10], [0.5, 0.10], [1.0, 0.10]],
            vec![[0.1, 0.20], [0.6, 0.20], [1.1, 0.20]],
            vec![[0.2, 0.10], [0.7, 0.10], [1.2, 0.10]],
        ];
        let dir = std::env::temp_dir().join(format!("evolution-gif-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("creature.gif");
        let written =
            write_creature_gif(&creature, &config, &frames, &[], &[0, 1, 2], None, &path).unwrap();
        assert_eq!(written, 3);
        // image::open proves the file is a decodable GIF.
        let first = image::open(&path).unwrap();
        assert_eq!((first.width(), first.height()), (GIF_WIDTH, GIF_HEIGHT));
        let decoder =
            GifDecoder::new(std::io::BufReader::new(std::fs::File::open(&path).unwrap())).unwrap();
        let decoded = decoder.into_frames().collect_frames().unwrap();
        assert_eq!(decoded.len(), 3);
        std::fs::remove_file(&path).ok();
        std::fs::remove_dir(&dir).ok();
    }
}
