//! Environment effects: world changes the player applies and can undo. Each
//! effect has a few levels, from the calm world upward. Changing a level
//! changes the physics, so the worker re-tests the archive's elites under the
//! new rules instead of keeping their old scores.
use crate::config::Config;

pub struct Effect {
    pub name: &'static str,
    /// What the world is like at each level, from level 0 upward.
    pub levels: &'static [&'static str],
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
/// Ground friction multiplier at each level.
pub const GRIP: [f32; 4] = [1.5, 1.0, 0.6, 0.3];

fn nearest(table: &[f32], value: f32) -> usize {
    table
        .iter()
        .enumerate()
        .min_by(|a, b| (a.1 - value).abs().total_cmp(&(b.1 - value).abs()))
        .map_or(0, |(i, _)| i)
}

pub const EFFECTS: [Effect; 4] = [
    Effect {
        name: "Ground",
        levels: &[
            "Flat",
            "Pebbles, 3 cm",
            "Rough, 8 cm",
            "Rocky, 15 cm",
            "Boulders, 25 cm",
        ],
        raise: "Roughen",
        lower: "Smooth",
        why: "Bumps catch dragged nodes and trip shuffling gaits. Lifted feet and real steps pay off.",
        get: |c| usize::from(c.terrain),
        set: |c, level| c.terrain = level as u8,
    },
    Effect {
        name: "Gravity",
        levels: &["Earth", "1.5 g", "2 g", "3 g"],
        raise: "Strengthen",
        lower: "Weaken",
        why: "Heavy bodies need firm steps and strong muscles. Large bodies suffer most, so compact ones gain.",
        get: |c| nearest(&GRAVITY, c.gravity),
        set: |c, level| c.gravity = GRAVITY[level],
    },
    Effect {
        name: "Air",
        levels: &["Thin", "Breezy", "Thick", "Syrup"],
        raise: "Thicken",
        lower: "Thin",
        why: "Drag slows every fast-moving node. Flailing wastes speed, and smooth, efficient strokes win.",
        get: |c| nearest(&AIR, c.air_retention),
        set: |c, level| c.air_retention = AIR[level],
    },
    Effect {
        name: "Grip",
        levels: &["Grippy", "Firm", "Wet", "Ice"],
        raise: "Make slippery",
        lower: "Restore grip",
        why: "Sliding feet stop working. Gaits must press down and push back instead of scraping.",
        get: |c| nearest(&GRIP, c.ground_friction),
        set: |c, level| c.ground_friction = GRIP[level],
    },
];

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
    fn default_world_is_level_zero_everywhere() {
        let cfg = Config::default();
        for effect in &EFFECTS {
            assert_eq!(effect.level(&cfg), 0, "{}", effect.name);
        }
    }
}
