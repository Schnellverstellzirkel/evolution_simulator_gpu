//! Physics v2 on the GPU against the CPU prototype.
//!
//! Needs the RTX 4060 and is ignored by default. Run it with
//!
//!     cargo test --release --test physics2_gpu -- --ignored
//!
//! Trials are short, because contact sequences are chaotic: a footfall that
//! lands one step apart in the two engines sends the trials apart, and that
//! grows with time. `examples/p2_agreement.rs` reports the distributions over
//! long trials.
use evolution_simulator::{
    config::Config,
    engine::{self, Engine},
    evolution::{self, Creature},
    physics2,
};
use std::time::Duration;

#[test]
#[ignore = "needs the RTX 4060"]
fn the_v2_kernel_agrees_with_the_cpu_prototype() {
    agree(0.0, false);
}

#[test]
#[ignore = "needs the RTX 4060"]
fn the_v2_kernel_agrees_with_the_cpu_prototype_in_mud() {
    agree(0.06, false);
}

#[test]
#[ignore = "needs the RTX 4060"]
fn the_v2_kernel_agrees_with_the_cpu_prototype_with_tendons() {
    agree(0.0, true);
}

fn agree(mud: f32, tendons: bool) {
    assert!(physics2::enabled());
    let cfg = Config {
        population: 512,
        duration: 1.0,
        random_seed: false,
        screen: None,
        mud,
        ..Config::default()
    };
    let mut pop = evolution::create(&cfg).unwrap();
    if tendons {
        // A third of the muscles get a tendon, and a shorter longest length so
        // the body's own stretching engages it at once.
        for (i, m) in pop.muscles.iter_mut().enumerate() {
            if i % 3 == 0 {
                m.tendon = 0.2 + 0.2 * ((i / 3) % 4) as f32;
                m.long = m.short + 0.3 * (m.long - m.short);
            }
        }
    }
    // A body that falls over and one with a bone at the head.
    let chain = |nodes: &[[f32; 2]]| {
        let genes: Vec<evolution::NodeGene> = nodes
            .iter()
            .map(|p| evolution::NodeGene {
                x: p[0],
                y: p[1],
                diameter: 0.08,
                friction: 0.6,
            })
            .collect();
        let bones = (1..nodes.len())
            .map(|i| {
                let (a, b) = (nodes[i - 1], nodes[i]);
                evolution::Bone::new((i - 1) as u32, i as u32, (b[0] - a[0]).hypot(b[1] - a[1]))
            })
            .collect();
        Creature {
            nodes: genes,
            bones,
            muscles: Vec::new(),
            id: 7,
            mutability: 1.0,
        }
    };
    pop.push(chain(&[[0.0, 0.6], [0.0, 0.3], [0.3, 0.3], [0.6, 0.05]]));
    let cpu = physics2::evaluate(&pop, &cfg);
    // Both GPU backends: Vulkan first, then CUDA. The variable is read when
    // an engine opens; this test binary has one test, so nothing else reads
    // the environment meanwhile.
    for (setting, expect) in [("0", "Vulkan"), ("1", "CUDA")] {
        // SAFETY: see above.
        unsafe { std::env::set_var("EVOLUTION_CUDA", setting) };
        let mut gpu = engine::gpu_engine("RTX 4060", 16, 64).expect("GPU");
        let name = gpu.name();
        assert!(
            expect == "Vulkan" || name.contains(expect),
            "expected {expect}, opened {name}"
        );
        gpu.submit(pop.clone(), &cfg).unwrap();
        let done = loop {
            if let Some(done) = gpu.poll().unwrap() {
                break done;
            }
            gpu.wait(Duration::from_millis(20));
        };
        let mut worst = 0.0f32;
        let mut close = 0usize;
        for (a, b) in cpu.iter().zip(&done.results) {
            assert_eq!(
                a.fitness <= -1e19,
                b.fitness <= -1e19,
                "failed trials must agree"
            );
            let gap = (a.fitness - b.fitness).abs();
            worst = worst.max(gap);
            close += usize::from(gap <= 0.005);
        }
        eprintln!(
            "{name}: {} creatures, 1 s: worst distance gap {worst:.4} m, {close} within 5 mm",
            cpu.len()
        );
        assert!(
            close * 100 >= cpu.len() * 99,
            "{name}: only {close} of {} within 5 mm",
            cpu.len()
        );
    }
}
