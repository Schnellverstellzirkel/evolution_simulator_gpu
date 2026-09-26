//! Environment effects: world changes the player applies and can undo. Each
//! effect has a few levels around a calm default. The grip is the one effect
//! whose levels go both ways: rougher than the calm world as well as
//! slipperier. Changing a level changes the physics, so the worker re-tests
//! the archive's elites under the new rules instead of keeping their old
//! scores.
use crate::config::Config;

pub struct Effect {
    pub name: &'static str,
    /// What the world is like at each level, from level 0 upward.
    pub levels: &'static [&'static str],
    /// Level of the calm world. The buttons step away from it in both
    /// directions, so an effect with a calm level above zero can also be
    /// pushed past its default.
    pub calm: usize,
    /// Button text to raise and to lower the level.
    pub raise: &'static str,
    pub lower: &'static str,
    /// Why this effect pushes evolution somewhere interesting.
    pub why: &'static str,
    get: fn(&Config) -> usize,
    set: fn(&mut Config, usize),
}

impl Effect {
    pub fn level(&self, cfg: &Config) -> usize {
        (self.get)(cfg).min(self.levels.len() - 1)
    }
    pub fn set_level(&self, cfg: &mut Config, level: usize) {
        (self.set)(cfg, level.min(self.levels.len() - 1));
    }
}

/// Gravity at each level (m/s^2).
pub const GRAVITY: [f32; 4] = [9.8, 14.7, 19.6, 29.4];
/// Velocity kept per 1/60 s at each level: less means thicker air.
pub const AIR: [f32; 4] = [1.0, 0.995, 0.985, 0.96];
/// Ground friction multiplier at each level, from grippier than the calm
/// world to nearly frictionless ice.
pub const GRIP: [f32; 5] = [3.0, 1.5, 1.0, 0.6, 0.3];
/// Muscle energy store multiplier at each level, from the calm world down to
/// a harsh heat wave.
pub const HEAT: [f32; 4] = [1.0, 0.7, 0.5, 0.35];
/// Muscle energy recovery multiplier at each level, from the calm world down
/// to almost no recovery.
pub const DROUGHT: [f32; 4] = [1.0, 0.6, 0.3, 0.1];
/// Ground slope (rise over run) at each level, from flat to a steep climb.
pub const SLOPE: [f32; 5] = [0.0, 0.03, 0.08, 0.15, 0.25];
/// Horizontal wind acceleration (m/s²) at each level, from calm to a steady
/// headwind that opposes +x travel.
pub const WIND: [f32; 4] = [0.0, -1.0, -3.0, -6.0];
/// Mud sink depth (m) at each level, from dry ground to deep mud.
pub const MUD: [f32; 4] = [0.0, 0.02, 0.05, 0.10];
/// Pit opening width (m) at each level, from solid ground to chasms. The pit
/// spacing grows with the width (`physics::gap_spacing`).
pub const GAPS: [f32; 4] = [0.0, 0.35, 0.8, 1.5];
/// Hurdle height (m) at each level, from clear ground to steps tall enough
/// that climbing or leaping over them is the gait's main job.
pub const HURDLES: [f32; 4] = [0.0, 0.08, 0.20, 0.35];
/// Earthquake base bump height (m) at each level. Each creature jitters the
/// phase and height with its own deterministic stream.
pub const QUAKE: [f32; 4] = [0.0, 0.05, 0.12, 0.25];
/// Generations between automatic season steps at each seasons level:
/// Off, Slow, Normal, Fast. Level 0 leaves the world alone.
pub const SEASON_INTERVALS: [u32; 4] = [0, 20, 10, 5];
/// Effects the season rotation visits first, in this order. Every other table
/// effect follows in `EFFECTS` order, and the Seasons row itself never
/// rotates, so adding an effect to the table adds it to the seasons.
const SEASON_FIRST: [&str; 5] = ["Wind", "Ground", "Grip", "Mud", "Slope"];

fn nearest(table: &[f32], value: f32) -> usize {
    table
        .iter()
        .enumerate()
        .min_by(|a, b| (a.1 - value).abs().total_cmp(&(b.1 - value).abs()))
        .map_or(0, |(i, _)| i)
}

pub const EFFECTS: [Effect; 13] = [
    Effect {
        name: "Seasons",
        levels: &["Off", "Slow", "Normal", "Fast"],
        calm: 0,
        raise: "Faster seasons",
        lower: "Slower seasons",
        why: "The world advances one effect one level every 20, 10, or 5 generations, walking through wind, ground, grip, mud, slope, and every other effect before returning to calm. Gaits meet a changing world instead of one fixed challenge.",
        get: |c| usize::from(c.seasons),
        set: |c, level| c.seasons = level as u8,
    },
    Effect {
        name: "Ground",
        levels: &[
            "Flat",
            "Pebbles, 3 cm",
            "Rough, 8 cm",
            "Rocky, 15 cm",
            "Boulders, 25 cm",
        ],
        calm: 0,
        raise: "Roughen",
        lower: "Smooth",
        why: "Bumps catch dragged nodes and trip shuffling gaits. Lifted feet and real steps pay off.",
        get: |c| usize::from(c.terrain),
        set: |c, level| c.terrain = level as u8,
    },
    Effect {
        name: "Gravity",
        levels: &["Earth", "1.5 g", "2 g", "3 g"],
        calm: 0,
        raise: "Strengthen",
        lower: "Weaken",
        why: "Heavy bodies need firm steps and strong muscles. Large bodies suffer most, so compact ones gain.",
        get: |c| nearest(&GRAVITY, c.gravity),
        set: |c, level| c.gravity = GRAVITY[level],
    },
    Effect {
        name: "Air",
        levels: &["Thin", "Breezy", "Thick", "Syrup"],
        calm: 0,
        raise: "Thicken",
        lower: "Thin",
        why: "Drag slows every fast-moving node. Flailing wastes speed, and smooth, efficient strokes win.",
        get: |c| nearest(&AIR, c.air_retention),
        set: |c, level| c.air_retention = AIR[level],
    },
    Effect {
        name: "Grip",
        levels: &["Sandpaper", "Grippy", "Firm", "Wet", "Ice"],
        calm: 1,
        raise: "Make slippery",
        lower: "More grip",
        why: "Sliding feet stop working, so a body that scrapes along the ground loses its push. Rough ground holds planted feet firmly; smooth ground cannot push a slider at all.",
        get: |c| nearest(&GRIP, c.ground_friction),
        set: |c, level| c.ground_friction = GRIP[level],
    },
    Effect {
        name: "Heat wave",
        levels: &["Full", "Warm", "Hot", "Heat wave"],
        calm: 0,
        raise: "Heat up",
        lower: "Cool down",
        why: "A smaller energy store runs out sooner. Short strokes must count and the gait has to recover between them.",
        get: |c| nearest(&HEAT, c.muscle_energy),
        set: |c, level| c.muscle_energy = HEAT[level],
    },
    Effect {
        name: "Drought",
        levels: &["Normal", "Dry", "Parched", "Drought"],
        calm: 0,
        raise: "Dry out",
        lower: "Water",
        why: "Energy returns slowly, so bursts fail and steady, well-paced gaits win.",
        get: |c| nearest(&DROUGHT, c.muscle_recovery),
        set: |c, level| c.muscle_recovery = DROUGHT[level],
    },
    Effect {
        name: "Slope",
        levels: &["Flat", "3%", "8%", "15%", "25%"],
        calm: 0,
        raise: "Tilt uphill",
        lower: "Flatten",
        why: "A climb charges the body for every meter of height it gains, so long, heavy bodies pay and compact ones keep their speed.",
        get: |c| nearest(&SLOPE, c.slope),
        set: |c, level| c.slope = SLOPE[level],
    },
    Effect {
        name: "Wind",
        levels: &["Calm", "Breeze", "Strong", "Gale"],
        calm: 0,
        raise: "Headwind",
        lower: "Calm",
        why: "A steady headwind pushes every node back. Low, streamlined bodies waste less of each stroke and stop flailing.",
        get: |c| nearest(&WIND, c.wind),
        set: |c, level| c.wind = WIND[level],
    },
    Effect {
        name: "Mud",
        levels: &["Dry", "Damp", "Muddy", "Deep mud"],
        calm: 0,
        raise: "Add water",
        lower: "Drain",
        why: "Sunk feet drag through the mud, so every stroke pays for the ground it scrapes. Lifted feet and real steps come out ahead.",
        get: |c| nearest(&MUD, c.mud),
        set: |c, level| c.mud = MUD[level],
    },
    Effect {
        name: "Gaps",
        levels: &["Solid", "Narrow", "Wide", "Chasms"],
        calm: 0,
        raise: "Open gaps",
        lower: "Fill in",
        why: "Periodic pits remove the ground a shuffling gait leans on. Bridges, leaps, and long bodies that can span a gap win.",
        get: |c| nearest(&GAPS, c.gaps),
        set: |c, level| c.gaps = GAPS[level],
    },
    Effect {
        name: "Hurdles",
        levels: &["Clear", "Low", "High", "Walls"],
        calm: 0,
        raise: "Raise hurdles",
        lower: "Lower hurdles",
        why: "Periodic steps break a flat shuffle. A gait must lift over every step or leap it whole, so climbing charges energy the smooth world never asked for.",
        get: |c| nearest(&HURDLES, c.hurdles),
        set: |c, level| c.hurdles = HURDLES[level],
    },
    Effect {
        name: "Earthquake",
        levels: &["Still", "Tremors", "Quakes", "Big one"],
        calm: 0,
        raise: "Shake",
        lower: "Calm",
        why: "Every creature meets bumps with its own phase and height, so a gait cannot memorize one pattern. Robust gaits that handle any ground come out ahead.",
        get: |c| nearest(&QUAKE, c.quake),
        set: |c, level| c.quake = QUAKE[level],
    },
];

/// The deterministic season rotation: one entry per scheduled step, each
/// setting exactly one effect to one level. Every effect except Seasons is
/// visited once per lap, and each effect's run ends back at its calm level,
/// so the calm world returns after every cycle.
pub fn season_rotation() -> Vec<(usize, usize)> {
    let mut order: Vec<usize> = SEASON_FIRST
        .iter()
        .filter_map(|name| EFFECTS.iter().position(|effect| effect.name == *name))
        .collect();
    for (index, effect) in EFFECTS.iter().enumerate() {
        if effect.name != "Seasons" && !order.contains(&index) {
            order.push(index);
        }
    }
    let mut rotation = Vec::new();
    for index in order {
        let effect = &EFFECTS[index];
        let levels = effect.levels.len();
        for step in 1..=levels {
            rotation.push((index, (effect.calm + step) % levels));
        }
    }
    rotation
}

/// Applies rotation step `step` to `cfg`. Returns true when any physics field
/// changed.
pub fn apply_season_step(cfg: &mut Config, step: u16) -> bool {
    let rotation = season_rotation();
    let Some(&(index, level)) = rotation.get(usize::from(step) % rotation.len()) else {
        return false;
    };
    let effect = &EFFECTS[index];
    let before = effect.level(cfg);
    effect.set_level(cfg, level);
    effect.level(cfg) != before
}

/// Applies one season step when `generation` begins, if the seasons level is
/// on and the generation is a multiple of its interval. The schedule depends
/// only on `cfg.seasons` and `generation`, never on wall time, so a checkpoint
/// resumed mid-cycle continues at the same step. Returns true when the world
/// changed.
pub fn advance_seasons(cfg: &mut Config, generation: u32) -> bool {
    let Some(&interval) = SEASON_INTERVALS.get(usize::from(cfg.seasons)) else {
        return false;
    };
    if interval == 0 || generation == 0 || !generation.is_multiple_of(interval) {
        return false;
    }
    let changed = apply_season_step(cfg, cfg.season_step);
    cfg.season_step = cfg.season_step.wrapping_add(1);
    changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_level_round_trips_and_validates() {
        for effect in &EFFECTS {
            for level in 0..effect.levels.len() {
                let mut cfg = Config::default();
                effect.set_level(&mut cfg, level);
                assert_eq!(effect.level(&cfg), level, "{}", effect.name);
                cfg.validate().unwrap();
            }
        }
    }

    #[test]
    fn default_world_is_each_effects_calm_level() {
        let cfg = Config::default();
        for effect in &EFFECTS {
            assert_eq!(effect.level(&cfg), effect.calm, "{}", effect.name);
        }
    }

    #[test]
    fn grip_reaches_both_slippery_and_grippier_ground() {
        let grip = EFFECTS.iter().find(|e| e.name == "Grip").unwrap();
        let calm = Config::default().ground_friction;
        let mut cfg = Config::default();
        grip.set_level(&mut cfg, 0);
        assert!(
            cfg.ground_friction > calm,
            "level 0 must be grippier than the calm world"
        );
        grip.set_level(&mut cfg, grip.levels.len() - 1);
        assert!(
            cfg.ground_friction < calm,
            "the last level must be slipperier than the calm world"
        );
    }

    #[test]
    fn slope_only_climbs_and_wind_only_opposes_the_run() {
        assert!(
            SLOPE.windows(2).all(|pair| pair[1] > pair[0]),
            "slope levels must rise monotonically from flat"
        );
        assert!(
            WIND.windows(2).all(|pair| pair[1] < pair[0]),
            "wind levels must strengthen monotonically from calm"
        );
        assert_eq!(SLOPE[0], 0.0);
        assert_eq!(WIND[0], 0.0);
        let slope = EFFECTS.iter().find(|e| e.name == "Slope").unwrap();
        let wind = EFFECTS.iter().find(|e| e.name == "Wind").unwrap();
        let mut cfg = Config::default();
        slope.set_level(&mut cfg, slope.levels.len() - 1);
        assert!(cfg.slope > 0.0, "the steepest level must tilt uphill");
        let mut cfg = Config::default();
        wind.set_level(&mut cfg, wind.levels.len() - 1);
        assert!(cfg.wind < 0.0, "the strongest level must oppose +x travel");
    }

    #[test]
    fn mud_deepens_and_gaps_widen_monotonically() {
        assert!(
            MUD.windows(2).all(|pair| pair[1] > pair[0]),
            "mud levels must deepen from dry ground"
        );
        assert!(
            GAPS.windows(2).all(|pair| pair[1] > pair[0]),
            "gap levels must widen from solid ground"
        );
        assert_eq!(MUD[0], 0.0);
        assert_eq!(GAPS[0], 0.0);
        let mud = EFFECTS.iter().find(|e| e.name == "Mud").unwrap();
        let gaps = EFFECTS.iter().find(|e| e.name == "Gaps").unwrap();
        let mut cfg = Config::default();
        mud.set_level(&mut cfg, mud.levels.len() - 1);
        assert!(cfg.mud > 0.0, "deep mud must sink contacting nodes");
        let mut cfg = Config::default();
        gaps.set_level(&mut cfg, gaps.levels.len() - 1);
        assert!(cfg.gaps > 0.0, "chasms must open pits");
    }

    #[test]
    fn hurdles_rise_and_quake_roughens_monotonically() {
        assert!(
            HURDLES.windows(2).all(|pair| pair[1] > pair[0]),
            "hurdle levels must rise from clear ground"
        );
        assert!(
            QUAKE.windows(2).all(|pair| pair[1] > pair[0]),
            "quake levels must roughen from still ground"
        );
        assert_eq!(HURDLES[0], 0.0);
        assert_eq!(QUAKE[0], 0.0);
        let hurdles = EFFECTS.iter().find(|e| e.name == "Hurdles").unwrap();
        let quake = EFFECTS.iter().find(|e| e.name == "Earthquake").unwrap();
        let mut cfg = Config::default();
        hurdles.set_level(&mut cfg, hurdles.levels.len() - 1);
        assert!(cfg.hurdles > 0.0, "the tallest level must raise steps");
        let mut cfg = Config::default();
        quake.set_level(&mut cfg, quake.levels.len() - 1);
        assert!(cfg.quake > 0.0, "the strongest level must shake the ground");
    }

    #[test]
    fn heat_wave_and_drought_only_get_harsher() {
        let heat = EFFECTS.iter().find(|e| e.name == "Heat wave").unwrap();
        let drought = EFFECTS.iter().find(|e| e.name == "Drought").unwrap();
        let mut cfg = Config::default();
        heat.set_level(&mut cfg, heat.levels.len() - 1);
        assert!(cfg.muscle_energy < 1.0, "heat wave must shrink the store");
        let mut cfg = Config::default();
        drought.set_level(&mut cfg, drought.levels.len() - 1);
        assert!(cfg.muscle_recovery < 1.0, "drought must slow recovery");
    }

    fn with_seasons(seasons: u8) -> Config {
        Config {
            seasons,
            ..Config::default()
        }
    }

    #[test]
    fn seasons_step_on_the_generation_schedule() {
        let interval = SEASON_INTERVALS[2];
        let mut cfg = with_seasons(2);
        for generation in 1..interval {
            assert!(!advance_seasons(&mut cfg, generation));
        }
        assert_eq!(
            cfg,
            with_seasons(2),
            "nothing may change before the interval"
        );
        assert!(advance_seasons(&mut cfg, interval));
        assert_eq!(cfg.season_step, 1);
        assert_eq!(
            cfg.wind, WIND[1],
            "the first step is the wind's first level"
        );
        let mut off = with_seasons(0);
        for generation in (0..200).step_by(5) {
            assert!(!advance_seasons(&mut off, generation));
        }
        assert_eq!(off, with_seasons(0), "the off level never steps");
    }

    #[test]
    fn each_season_step_changes_one_effect_and_the_lap_returns_to_calm() {
        let interval = SEASON_INTERVALS[2];
        let rotation = season_rotation();
        assert!(rotation.len() > 1);
        let mut cfg = with_seasons(2);
        for step in 0..rotation.len() {
            let generation = (step as u32 + 1) * interval;
            let before = cfg.clone();
            assert!(advance_seasons(&mut cfg, generation));
            let changed = EFFECTS
                .iter()
                .filter(|effect| effect.level(&cfg) != effect.level(&before))
                .count();
            assert_eq!(changed, 1, "step {step} must touch exactly one effect");
        }
        for effect in &EFFECTS {
            if effect.name == "Seasons" {
                assert_eq!(effect.level(&cfg), 2, "the seasons setting must not rotate");
                continue;
            }
            assert_eq!(
                effect.level(&cfg),
                effect.calm,
                "{} must be calm after a full lap",
                effect.name
            );
        }
        assert_eq!(usize::from(cfg.season_step), rotation.len());
        let mut next_lap = cfg.clone();
        advance_seasons(&mut next_lap, (rotation.len() as u32 + 1) * interval);
        let mut first_lap = with_seasons(2);
        advance_seasons(&mut first_lap, interval);
        for effect in &EFFECTS {
            assert_eq!(
                effect.level(&next_lap),
                effect.level(&first_lap),
                "{} repeats the same lap",
                effect.name
            );
        }
    }

    #[test]
    fn the_season_lap_visits_every_effect_but_seasons() {
        let rotation = season_rotation();
        let seats = EFFECTS.iter().filter(|e| e.name != "Seasons").count();
        let mut visited: Vec<usize> = rotation.iter().map(|&(index, _)| index).collect();
        visited.sort_unstable();
        visited.dedup();
        assert_eq!(visited.len(), seats);
        for (index, effect) in EFFECTS.iter().enumerate() {
            let levels: Vec<usize> = rotation
                .iter()
                .filter(|&&(i, _)| i == index)
                .map(|&(_, level)| level)
                .collect();
            if effect.name == "Seasons" {
                assert!(levels.is_empty(), "the Seasons row must not rotate");
                continue;
            }
            assert_eq!(levels.len(), effect.levels.len(), "{}", effect.name);
            assert_eq!(*levels.last().unwrap(), effect.calm, "{}", effect.name);
        }
    }

    #[test]
    fn seasons_do_not_depend_on_wall_time() {
        let interval = SEASON_INTERVALS[1];
        let run = |pause: bool| {
            let mut cfg = with_seasons(1);
            for generation in 1..=interval * 6 {
                if pause {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                advance_seasons(&mut cfg, generation);
            }
            cfg
        };
        assert_eq!(run(false), run(true));
    }
}
