//! Writes an animated GIF of one creature's trial (a JSON file from
//! `filmstrip` with EVOLUTION_FILM_DUMP, or a creature exported from the
//! game), replayed under physics v2 (
//! the v2 prototype).
//!
//! Usage: cargo run --release --example creature_gif -- <creature.json> <out.gif> [from s] [seconds] [fps]
use evolution_simulator::{config::Config, evolution::Creature, ui};

fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let json: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let body = if json.get("creature").is_some() {
        json["creature"].clone()
    } else {
        json
    };
    let creature: Creature = serde_json::from_value(body)?;
    let out = args
        .get(2)
        .cloned()
        .unwrap_or_else(|| "creature.gif".into());
    let from: f32 = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(0.0);
    let seconds: f32 = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(8.0);
    let fps: f32 = args.get(5).and_then(|v| v.parse().ok()).unwrap_or(20.0);
    let cfg = Config {
        random_seed: false,
        duration: (from + seconds).max(20.0),
        ..Config::default()
    };
    let frames = ui::creature_gif(
        &creature,
        &cfg,
        from,
        seconds,
        fps,
        std::path::Path::new(&out),
    )?;
    let bytes = std::fs::metadata(&out)?.len();
    println!("{out}: {frames} frames, {} KB", bytes / 1024);
    Ok(())
}
