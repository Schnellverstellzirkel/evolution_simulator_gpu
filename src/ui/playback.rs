//! One creature's replay. A `Playback` holds the frames the scoring kernel
//! recorded, the follow camera's track and the replay clock. A `FrameMarks`
//! holds what the scene draws over the pose of one frame. The viewport, the
//! Race tab and the GIF export read a `Playback`, and `scene` draws its pose
//! with the marks.

use crate::{
    config::Config,
    creature_kernel,
    evolution::Creature,
    physics::{self, Node},
};
use std::time::Duration;

/// What ended a trial early. The scoring kernel stops a trial when the head
/// falls, a joint breaks or the head shakes too hard, and the replay names
/// which of the three it was.
#[derive(Clone, Copy, PartialEq, Debug)]
pub(super) enum Ending {
    /// The head dropped below its neck.
    Fell,
    /// A joint was forced past its break angle.
    Broke,
    /// The head's averaged acceleration passed the 8 g limit.
    Shook,
}
impl Ending {
    /// The sentence the viewport shows once the replay reaches the event,
    /// given the time of the event in seconds into the trial.
    pub(super) fn sentence(self, seconds: f32) -> String {
        match self {
            Self::Fell => format!("Fell over at {seconds:.1} s: head below its neck"),
            Self::Broke => format!("Broke a joint at {seconds:.1} s"),
            Self::Shook => format!("Shook its head too hard at {seconds:.1} s (over 8 g)"),
        }
    }
    /// A word or two for a race lane once its creature has fallen.
    pub(super) fn short(self) -> &'static str {
        match self {
            Self::Fell => "fell",
            Self::Broke => "broke a joint",
            Self::Shook => "shook too hard",
        }
    }
}
/// One creature's trial as the scoring kernel recorded it, and how far the
/// replay has played. `nodes` holds the pose on screen.
pub(super) struct Playback {
    /// The creature, with its bones in canonical order.
    pub(super) creature: Creature,
    /// The world the trial ran in.
    pub(super) config: Config,
    /// The creature's nodes. Their positions are the pose on screen, which
    /// `show` and `show_between` set.
    pub(super) nodes: Vec<Node>,
    /// `[frame][node]`: node positions, recorded by the scoring kernel. Frames
    /// 0 to `physics::settle()` all show the start pose, and frame `settle() + n`
    /// shows the pose after `n` steps. A replay without a recording has one
    /// frame.
    pub(super) frames: Vec<Vec<[f32; 2]>>,
    /// Index of the frame on screen.
    pub(super) tick: u32,
    /// Replay time (s) that the player has added and not yet played as a
    /// step. The player adds the time of each UI frame and takes one step's
    /// time off for each step it plays. `blend` turns what is left into a
    /// share of the next step.
    pub(super) accumulator: f32,
    /// The frame at which the trial ended early (a fall, a broken joint or a
    /// shaken head), and the distance the kernel scored there. `None` when the
    /// trial ran its full length.
    pub(super) fall: Option<(u32, f32)>,
    /// What ended the trial early. While `fall` is `None` it is `Fell` and
    /// means nothing.
    pub(super) ending: Ending,
    /// The distance the kernel scored in the run that recorded these frames.
    /// It is 0 for a replay without a recording.
    pub(super) distance: f32,
    /// The x position (m) where the follow camera looks at each frame: the
    /// body's center of mass averaged over `CAMERA_WINDOW` seconds on either
    /// side. Every frame is recorded in advance, so the average cancels the
    /// swing of each stride without lagging behind a steady walk.
    track: Vec<f32>,
    /// Typical body height (m) over the scored frames, for the default zoom.
    pub(super) height: f32,
    /// Highest point of the body (m) over the scored frames.
    pub(super) peak: f32,
    /// What the kernel recorded with each frame: muscle energy, muscle force,
    /// ground push and broken joints. When it recorded none,
    /// `replay_forces::analyze` estimates the first three from the frames.
    pub(super) forces: crate::replay_forces::Forces,
    /// Whether the replay is a still first pose while the recording runs.
    pub(super) preparing: bool,
    /// Whether the replay is a still first pose because the GPU did not record
    /// it.
    pub(super) unavailable: bool,
}
/// Half-width of the follow camera's average of the center of mass (s).
const CAMERA_WINDOW: f32 = 1.0;
impl Playback {
    /// A replay that waits up to 3 s for the GPU's recording. Race lanes use
    /// it, because the UI thread builds them in one go.
    pub(super) fn new(creature: Creature, config: Config) -> Self {
        Self::recorded(creature, config, Duration::from_secs(3))
    }
    /// A replay that waits `patience` for the GPU's recording. Without one it
    /// holds the first pose and sets `unavailable`. The bones go into canonical
    /// order first, because the kernel numbers them that way and its record of
    /// broken joints follows those numbers.
    pub(super) fn recorded(creature: Creature, config: Config, patience: Duration) -> Self {
        let mut normalized = creature.clone();
        crate::evolution::canonicalize_bone_order(&mut normalized);
        // The kernel that recorded the frames also decides when the trial
        // ended and how far it got, so the replay shows exactly its score.
        let Some((frames, result, recorded_forces)) =
            crate::engine::replay(&normalized, &config, patience)
        else {
            let mut playback = Self::preparing(normalized, config);
            playback.preparing = false;
            playback.unavailable = true;
            return playback;
        };
        Self::from_recording(normalized, config, frames, result, recorded_forces)
    }
    /// The creature's first pose, held still while its replay is recorded.
    pub(super) fn preparing(creature: Creature, config: Config) -> Self {
        let mut normalized = creature;
        crate::evolution::canonicalize_bone_order(&mut normalized);
        let start: Vec<[f32; 2]> = physics::nodes(&normalized).iter().map(|n| n.pos).collect();
        let mut playback = Self::from_recording(
            normalized,
            config,
            vec![start],
            creature_kernel::GpuResult::default(),
            None,
        );
        playback.preparing = true;
        playback
    }
    /// A replay of `frames` and the `result` the kernel scored in the same run,
    /// for a creature whose bones are in canonical order. It finds the fall
    /// frame, what ended the trial, the camera track and the body heights. The
    /// forces are `recorded_forces`, or an estimate from the frames.
    fn from_recording(
        normalized: Creature,
        config: Config,
        frames: Vec<Vec<[f32; 2]>>,
        result: creature_kernel::GpuResult,
        recorded_forces: Option<crate::replay_forces::Forces>,
    ) -> Self {
        let nodes = physics::nodes(&normalized);
        let last_frame = frames.len().saturating_sub(1).min(u32::MAX as usize) as u32;
        // The fall time counts seconds into the timed trial, which starts at
        // frame `settle()`.
        let fall = (result.fall_time > 0.0).then(|| {
            let tick = physics::settle()
                .saturating_add((result.fall_time * physics::rate() as f32).round() as u32);
            (tick.min(last_frame), result.fitness)
        });
        let track = camera_track(&frames, &nodes);
        // Frames after the trial ended keep moving (a fallen body tumbles), so
        // the zoom looks only at the scored part.
        let scored = &frames[(physics::settle() as usize).min(frames.len().saturating_sub(1))
            ..fall.map_or(frames.len(), |(tick, _)| {
                (tick as usize + 1).min(frames.len())
            })];
        let height = body_height(scored, &nodes);
        let peak = body_peak(scored, &nodes);
        let contact: Vec<Vec<bool>> = frames
            .iter()
            .map(|frame| {
                let mut down = vec![false; nodes.len()];
                node_contact(&nodes, frame, &normalized, &config, &mut down);
                down
            })
            .collect();
        // The values the engine recorded, else an estimate from the frames.
        let forces = recorded_forces.unwrap_or_else(|| {
            crate::replay_forces::analyze(
                &normalized,
                &nodes,
                &frames,
                &contact,
                fall.map(|(tick, _)| tick),
                &config,
            )
        });
        // The head-shake average stops updating when the trial ends, so it
        // still holds the value that ended it. The kernel tests the joints on
        // the pose after each step and records what it found with that pose.
        // A break is looked for in the fall frame and in the frame before it.
        let ending = match fall {
            _ if result.head_shake > physics::HEAD_SHAKE_LIMIT => Ending::Shook,
            Some((tick, _)) => {
                let broke = [tick, tick.saturating_sub(1)]
                    .iter()
                    .any(|&t| forces.broken.get(t as usize).is_some_and(|&b| b != 0));
                if broke { Ending::Broke } else { Ending::Fell }
            }
            None => Ending::Fell,
        };
        let mut playback = Self {
            nodes,
            fall,
            ending,
            distance: result.fitness,
            height,
            peak,
            forces,
            track,
            creature: normalized,
            config,
            tick: physics::settle()
                .min(last_frame)
                .saturating_add(1)
                .min(last_frame),
            frames,
            accumulator: 0.0,
            preparing: false,
            unavailable: false,
        };
        playback.show();
        playback
    }
    /// Moves the replay back to the first step of the trial, the frame after
    /// `trial_start`.
    pub(super) fn reset(&mut self) {
        self.tick = self.trial_start().saturating_add(1).min(self.last_frame());
        self.show();
    }
    /// Index of the last recorded frame, 0 when there is one frame or none.
    pub(super) fn last_frame(&self) -> u32 {
        self.frames.len().saturating_sub(1).min(u32::MAX as usize) as u32
    }
    /// Index of the frame at time 0 of the trial. The frames up to it all show
    /// the start pose.
    pub(super) fn trial_start(&self) -> u32 {
        physics::settle().min(self.last_frame())
    }
    /// Seconds since the trial started, at most the length of a trial.
    pub(super) fn elapsed_seconds(&self) -> f32 {
        self.tick
            .saturating_sub(self.trial_start())
            .min(self.config.steps()) as f32
            * physics::dt()
    }
    /// Moves to a frame counted from `trial_start`, at most the last frame. It
    /// also clears `accumulator`.
    pub(super) fn seek(&mut self, elapsed_frame: u32) {
        self.tick = self
            .trial_start()
            .saturating_add(elapsed_frame)
            .min(self.last_frame());
        self.accumulator = 0.0;
        self.show();
    }
    /// Moves one recorded frame forward, to the last frame at most.
    pub(super) fn advance(&mut self) {
        self.tick = self.tick.saturating_add(1).min(self.last_frame());
        self.show();
    }
    /// The `fall`, once the replay has reached its frame.
    pub(super) fn fallen(&self) -> Option<(u32, f32)> {
        self.fall.filter(|&(tick, _)| self.tick >= tick)
    }
    /// Puts the nodes at the pose of the current frame.
    fn show(&mut self) {
        if let Some(frame) = self.frames.get(self.tick as usize) {
            for (node, position) in self.nodes.iter_mut().zip(frame) {
                node.pos = *position;
            }
        }
    }
    /// Share of the way to the next recorded frame that the replay clock
    /// has gone, from `accumulator`.
    fn blend(&self) -> f32 {
        (self.accumulator / physics::dt()).clamp(0.0, 1.0)
    }
    /// Places the nodes between the current frame and the next by `blend`,
    /// so motion looks smooth when the screen refreshes faster than the
    /// 60 Hz recording. At the last frame it shows that frame.
    pub(super) fn show_between(&mut self) {
        let alpha = self.blend();
        let (Some(now), Some(next)) = (
            self.frames.get(self.tick as usize),
            self.frames.get(self.tick as usize + 1),
        ) else {
            return self.show();
        };
        for ((node, a), b) in self.nodes.iter_mut().zip(now).zip(next) {
            node.pos = [a[0] + (b[0] - a[0]) * alpha, a[1] + (b[1] - a[1]) * alpha];
        }
    }
    /// Where the follow camera looks now (m): `track` at this frame, moved
    /// toward the next frame by `blend`.
    pub(super) fn camera_x(&self) -> f32 {
        let at = |tick: usize| {
            self.track
                .get(tick)
                .or(self.track.last())
                .copied()
                .unwrap_or(0.0)
        };
        let tick = self.tick as usize;
        let alpha = self.blend();
        at(tick) + (at(tick + 1) - at(tick)) * alpha
    }
    /// Mass-weighted center of the body as drawn, from the pose in `nodes`.
    pub(super) fn shown_center(&self) -> Option<[f32; 2]> {
        let mass: f32 = self.nodes.iter().map(|n| n.mass).sum();
        (mass > 0.0).then(|| {
            let x = self.nodes.iter().map(|n| n.mass * n.pos[0]).sum::<f32>();
            let y = self.nodes.iter().map(|n| n.mass * n.pos[1]).sum::<f32>();
            [x / mass, y / mass]
        })
    }
    /// Mass-weighted center of the body at a recorded frame, or `None` when
    /// there is no such frame or the body has no mass.
    pub(super) fn center_of_mass(&self, tick: u32) -> Option<[f32; 2]> {
        let frame = self.frames.get(tick as usize)?;
        let mut mass = 0.0;
        let mut center = [0.0; 2];
        for (node, position) in self.nodes.iter().zip(frame) {
            mass += node.mass;
            center[0] += node.mass * position[0];
            center[1] += node.mass * position[1];
        }
        (mass > 0.0).then(|| [center[0] / mass, center[1] / mass])
    }
    /// Center-of-mass speed over the last fifth of a second of recorded
    /// frames, in meters per second.
    pub(super) fn speed(&self) -> f32 {
        let window = (physics::rate() / 5).max(1);
        let start = self.tick.saturating_sub(window);
        let (Some(now), Some(before)) =
            (self.center_of_mass(self.tick), self.center_of_mass(start))
        else {
            return 0.0;
        };
        let seconds = self.tick.saturating_sub(start) as f32 * physics::dt();
        if seconds <= 0.0 {
            return 0.0;
        }
        (now[0] - before[0]).hypot(now[1] - before[1]) / seconds
    }
    /// Distance covered so far. It stops at the distance the kernel scored
    /// once the replay reaches the fall.
    pub(super) fn current_distance(&self) -> f32 {
        self.fallen()
            .map_or_else(|| physics::fitness(&self.nodes), |(_, distance)| distance)
    }
}
/// Frame-varying drawing state: muscle time, fallen look, and the per-node
/// marks for ground contact and broken joints.
#[derive(Default)]
pub(super) struct FrameMarks {
    pub(super) time: f32,
    pub(super) fallen: bool,
    pub(super) contact: Vec<bool>,
    pub(super) broken: Vec<bool>,
    /// Stored energy per muscle (1 is rested), for fading tired muscles.
    pub(super) energy: Vec<f32>,
    /// Muscle force per muscle (N) and ground push per node (N), drawn as
    /// arrows when `arrows` is on.
    pub(super) muscle_force: Vec<f32>,
    pub(super) ground_force: Vec<f32>,
    pub(super) arrows: bool,
}
impl FrameMarks {
    /// Contact and broken-joint marks of a playback's current frame.
    pub(super) fn of(playback: &Playback) -> Self {
        let mut marks = Self {
            time: playback.tick.saturating_sub(physics::settle()) as f32 * physics::dt(),
            fallen: playback.fallen().is_some(),
            contact: vec![false; playback.nodes.len()],
            broken: vec![false; playback.nodes.len()],
            energy: playback
                .forces
                .energy
                .get(playback.tick as usize)
                .cloned()
                .unwrap_or_default(),
            muscle_force: playback
                .forces
                .muscle
                .get(playback.tick as usize)
                .cloned()
                .unwrap_or_default(),
            ground_force: playback
                .forces
                .ground
                .get(playback.tick as usize)
                .cloned()
                .unwrap_or_default(),
            arrows: false,
        };
        // Developer screenshots: EVOLUTION_SMOKE_ENERGY=0.15 draws every muscle at that store.
        if let Some(level) = std::env::var("EVOLUTION_SMOKE_ENERGY")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
        {
            marks.energy.fill(level);
        }
        if let Some(frame) = playback.frames.get(playback.tick as usize) {
            node_contact(
                &playback.nodes,
                frame,
                &playback.creature,
                &playback.config,
                &mut marks.contact,
            );
            broken_nodes(
                &playback.creature,
                playback
                    .forces
                    .broken
                    .get(playback.tick as usize)
                    .copied()
                    .unwrap_or(0),
                &mut marks.broken,
            );
        }
        marks
    }
}
/// The highest point the body reaches in a recording (m above the ground).
fn body_peak(frames: &[Vec<[f32; 2]>], nodes: &[Node]) -> f32 {
    frames
        .iter()
        .flat_map(|frame| frame.iter().zip(nodes).map(|(p, n)| p[1] + n.radius))
        .fold(0.1f32, f32::max)
}
/// A creature's typical height over a recording (m above the ground): the
/// 90th percentile of the top of the body, so one leap does not shrink it.
pub(super) fn body_height(frames: &[Vec<[f32; 2]>], nodes: &[Node]) -> f32 {
    let mut tops: Vec<f32> = frames
        .iter()
        .map(|frame| {
            frame
                .iter()
                .zip(nodes)
                .map(|(p, n)| p[1] + n.radius)
                .fold(0.0f32, f32::max)
        })
        .collect();
    if tops.is_empty() {
        return 1.0;
    }
    tops.sort_by(f32::total_cmp);
    tops[(tops.len() - 1) * 9 / 10].max(0.1)
}
/// The follow camera's target per recorded frame: the mass-weighted center of
/// the body, averaged over `CAMERA_WINDOW` seconds on either side.
fn camera_track(frames: &[Vec<[f32; 2]>], nodes: &[Node]) -> Vec<f32> {
    let mass: f32 = nodes.iter().map(|n| n.mass).sum::<f32>().max(1e-6);
    let centers: Vec<f64> = frames
        .iter()
        .map(|frame| {
            let x: f32 = frame.iter().zip(nodes).map(|(p, n)| n.mass * p[0]).sum();
            f64::from(x / mass)
        })
        .collect();
    let mut sums = Vec::with_capacity(centers.len() + 1);
    sums.push(0.0f64);
    for c in &centers {
        sums.push(sums.last().unwrap() + c);
    }
    let half = (CAMERA_WINDOW * physics::rate() as f32).round() as usize;
    (0..centers.len())
        .map(|i| {
            let (a, b) = (i.saturating_sub(half), (i + half + 1).min(centers.len()));
            ((sums[b] - sums[a]) / (b - a) as f64) as f32
        })
        .collect()
}
/// Marks the nodes touching the ground in `positions`, with the threshold
/// `size_report` uses: a node is down when its center sits within 2 mm of the
/// terrain surface plus its own radius measured along the local normal. Gaps,
/// hurdles and the creature's own quake phase lower and raise the surface here
/// too, so the marks follow the ground that is drawn.
pub(super) fn node_contact(
    nodes: &[Node],
    positions: &[[f32; 2]],
    creature: &Creature,
    config: &Config,
    out: &mut [bool],
) {
    out.fill(false);
    if !config.ground {
        return;
    }
    let hash = physics::quake_hash(creature.id);
    let amplitude =
        physics::terrain_amplitude(config.terrain) + config.quake * physics::quake_scale(hash);
    let phase = if config.quake > 0.0 {
        physics::quake_phase(hash)
    } else {
        0.0
    };
    for ((node, position), down) in nodes.iter().zip(positions).zip(out.iter_mut()) {
        let (height, slope) = physics::ground(
            position[0],
            amplitude,
            config.slope,
            config.gaps,
            config.hurdles,
            phase,
        );
        let floor = height + node.radius * (1.0 + slope * slope).sqrt();
        *down = position[1] <= floor + 0.002;
    }
}
/// Marks both ends of every bone in `broken`, the bits of the bones whose
/// joint the scoring engine found past its break angle in a recorded frame
/// (`replay_forces::Forces::broken`).
pub(super) fn broken_nodes(creature: &Creature, broken: u64, out: &mut [bool]) {
    out.fill(false);
    for (j, bone) in creature.bones.iter().enumerate().take(64) {
        if broken & (1 << j) != 0 {
            out[bone.a as usize] = true;
            out[bone.b as usize] = true;
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::test_support::test_creature;
    #[test]
    fn the_follow_camera_ignores_the_stride_and_keeps_up_with_the_walk() {
        // One node walking at 2 m/s, swinging 0.3 m back and forth once per
        // second.
        let rate = physics::rate() as f32;
        let frames: Vec<Vec<[f32; 2]>> = (0..600)
            .map(|i| {
                let t = i as f32 / rate;
                vec![[2.0 * t + 0.3 * (std::f32::consts::TAU * t).sin(), 0.5]]
            })
            .collect();
        let nodes = vec![Node {
            mass: 1.0,
            ..Node::default()
        }];
        let track = camera_track(&frames, &nodes);
        let margin = (CAMERA_WINDOW * rate) as usize;
        for (i, x) in track.iter().enumerate().skip(margin).take(600 - 2 * margin) {
            let walk = 2.0 * i as f32 / rate;
            assert!(
                (x - walk).abs() < 0.02,
                "frame {i}: camera {x}, walk {walk}"
            );
        }
    }
    #[test]
    fn a_preparing_replay_holds_the_first_pose_and_survives_playback() {
        let mut playback = Playback::preparing(test_creature(), Config::default());
        assert!(playback.preparing);
        assert_eq!(playback.frames.len(), 1);
        playback.advance();
        playback.reset();
        playback.seek(5);
        assert_eq!(playback.tick, 0);
    }
}
