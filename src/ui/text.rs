//! Words and numbers for the player: species names with their gait words, and
//! short readable numbers, durations and file sizes. The tabs, cards and
//! dialogs call these helpers.

use crate::evolution::Creature;

/// Invented stems that start a species name. The body plan picks one.
const SPECIES_STEMS: [&str; 16] = [
    "Vex", "Tor", "Quil", "Nym", "Zeb", "Cro", "Fen", "Lum", "Tar", "Wisp", "Brak", "Ovi", "Pyr",
    "Sable", "Dro", "Ril",
];
/// Syllables between the stem and the size word.
const SPECIES_LINKS: [&str; 8] = ["a", "o", "i", "u", "e", "y", "ar", "en"];
/// A file size for people, in binary units: "512 B", "12 KiB", "3.4 MiB" or
/// "1.25 GiB".
pub(super) fn file_size(bytes: u64) -> String {
    const KIB: f64 = 1024.0;
    const MIB: f64 = KIB * 1024.0;
    const GIB: f64 = MIB * 1024.0;
    let bytes = bytes as f64;
    if bytes >= GIB {
        format!("{:.2} GiB", bytes / GIB)
    } else if bytes >= MIB {
        format!("{:.1} MiB", bytes / MIB)
    } else if bytes >= KIB {
        format!("{:.0} KiB", bytes / KIB)
    } else {
        format!("{bytes:.0} B")
    }
}
/// A species name such as "Vexapod Walker". The first word is a stem, a link
/// syllable and a size word. The body plan (`worker::body_plan`) picks the
/// stem and the link, and the bone count picks the size word. The second word
/// is the gait word of the muscles' rhythm. The name uses only creature data,
/// so every place that names a creature agrees without asking the worker.
pub(super) fn species_name(creature: &Creature) -> String {
    // The body plan decides the first word, so a creature keeps it through the
    // small mutations that tune lengths and rhythms. Only the gait word
    // follows the rhythm. A stem and a linking syllable give 128 first words
    // per size class.
    let plan = crate::worker::body_plan(creature);
    // Fold the high bits into the low bits, because the stem and the link read
    // the low bits.
    let mixed = plan ^ (plan >> 29) ^ (plan >> 47);
    let stem = SPECIES_STEMS[(mixed % SPECIES_STEMS.len() as u64) as usize];
    let link = SPECIES_LINKS[((mixed >> 8) % SPECIES_LINKS.len() as u64) as usize];
    // The size word grows with the bone count.
    let form = match creature.bones.len() {
        0..=2 => "ling",
        3..=4 => "pod",
        5..=7 => "form",
        8..=11 => "morph",
        _ => "titan",
    };
    format!("{stem}{link}{form} {}", gait_word(creature))
}
/// The gait word for the mean muscle period: "Crawler" under 0.5 cycles per
/// second, "Walker" under 1, "Trotter" under 2 and "Sprinter" above that. A
/// body with no muscles is a "Drifter".
fn gait_word(creature: &Creature) -> &'static str {
    if creature.muscles.is_empty() {
        return "Drifter";
    }
    let mean_period =
        creature.muscles.iter().map(|m| m.period).sum::<f32>() / creature.muscles.len() as f32;
    let hertz = 1.0 / mean_period.max(0.05);
    if hertz < 0.5 {
        "Crawler"
    } else if hertz < 1.0 {
        "Walker"
    } else if hertz < 2.0 {
        "Trotter"
    } else {
        "Sprinter"
    }
}
/// How long ago a file time was, such as "3 min ago". No time, or a time in
/// the future, gives "unknown time".
pub(super) fn ago(time: Option<std::time::SystemTime>) -> String {
    time.and_then(|t| t.elapsed().ok()).map_or_else(
        || "unknown time".to_owned(),
        |age| format!("{} ago", seconds_text(age.as_secs_f64())),
    )
}
/// A short duration for people: "8 s" up to 90 s, "3 min" up to 90 min, then
/// "2 h". It never shows less than "1 s". A negative or non-finite `seconds`
/// gives "a while".
pub(super) fn seconds_text(seconds: f64) -> String {
    if !seconds.is_finite() || seconds < 0.0 {
        "a while".to_owned()
    } else if seconds < 90.0 {
        format!("{:.0} s", seconds.max(1.0))
    } else if seconds < 90.0 * 60.0 {
        format!("{:.0} min", seconds / 60.0)
    } else {
        format!("{:.0} h", seconds / 3600.0)
    }
}
/// `n` with commas between thousands: 1,234,567.
pub(super) fn number(n: usize) -> String {
    let text = n.to_string();
    let mut out = String::new();
    for (i, c) in text.chars().enumerate() {
        if i > 0 && (text.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        evolution::{Bone, NodeGene},
        ui::test_support::test_creature,
    };
    #[test]
    fn body_plans_ignore_lengths_and_rhythms() {
        use crate::worker::body_plan;
        let creature = test_creature();
        let mut tuned = creature.clone();
        tuned.bones[0].rest_length = 0.9;
        tuned.muscles[0].period = 0.3;
        tuned.nodes[1].x = 0.7;
        assert_eq!(body_plan(&creature), body_plan(&tuned));
        let mut more = creature.clone();
        more.muscles.push(more.muscles[0]);
        assert_ne!(body_plan(&creature), body_plan(&more));
    }
    #[test]
    fn species_names_follow_the_body_plan() {
        let creature = test_creature();
        let name = species_name(&creature);
        let mut reversed = creature.clone();
        reversed.bones.reverse();
        assert_eq!(name, species_name(&reversed));
        let mut longer = creature.clone();
        longer.nodes.push(NodeGene {
            x: 1.5,
            y: 0.0,
            diameter: 0.2,
            friction: 0.8,
        });
        longer.bones.push(Bone::new(2, 3, 0.5));
        assert_ne!(name, species_name(&longer));
        let mut tuned = creature.clone();
        tuned.bones[1].rest_length = 0.8;
        tuned.muscles[0].phase = 0.4;
        assert_eq!(name, species_name(&tuned), "tuning keeps the name");
    }
}
