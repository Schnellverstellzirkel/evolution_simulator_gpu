# Round 5, GPU domain: the design I back, my tracks, what would change my mind, the ceiling

## 1. The design

One maximal-coordinate CUDA source templated on W, each lane owning 4 nodes and up to 16 muscles, class = max(ceil(nodes / 4), ceil(muscles / 16)), W = 2 first, W = 4 second, W = 8 last, the lane-group kernel serving what is not yet covered and retired when W = 8 lands. Contact rows unrolled at a fixed MAXC with predication. Registers at or under 128 with at most 32 B of spill, 16 warps per SM, about 150 B of shared per lane. Muscle records float32, split 32 B per step and 24 B per substep, no gene quantization. Genes DMA'd per block into VRAM on a copy engine, a per-block prologue kernel unpacking 32 creatures per warp into lane records in VRAM, take-up from those records (ruling 2; my round-3 fallback is now the primary path and I accept it, since the divergent take-up was my own attack). Rulings 3 to 13 as written: VRAM tail and drain flag with the 2 s timeout, frozen ring constants, the three bucket rules, recovery by re-registered process memory, the physics spec with the red-team fixes and the ledger over the projection, the ladder with L2.5 as the target row, the growth-step rule as the tail candidate with W = 8 last, the nursery exemptions and R4's emitter-median floor, the host tax at 15 to 21% until the rows exist, the compile and stage-log items, the disk rules. In-kernel confirmation is out (ruling 1) and I accept it: the two-implementation argument is right, and the 0.3% was never load-bearing. I reject no ruling.

## 2. Tracks

Each is one coding agent, one to two weeks, its own worktree at /home/amipo/workspace/evolutionSimulator-<track> on branch claude/<track>, seeded only when the disk has 30 GB free.

G1, `stub-kernel`. Owns: examples/lane_stub.rs (new), shaders/lane_stub.cu (new), examples/warp_regs.rs (extended to the stub). Deliverable: the stub of round 3 section 2 (real state layout, shared tables, 16-muscle loop from split L2 records, 4-rod LDL with the W = 2 exchange, 8 tree solves and the 8x8 direct solve on synthetic data, 300 fake steps over 262k creatures) plus a 30-minute nsys warp-state sample of the lane-group kernel on save42 (share of stalls that are barrier and short-scoreboard). Gate: at most 128 registers with at most 32 B spill; at least 45% issue-active at 16 warps; the extrapolated rate at least 250 M creature-steps/s on the mean body at 2 substeps at the logged clock. Dependencies: none. One week. First.

G2, `lane-w2`. Owns: shaders/lane_creature.cu (new), src/lane_kernel.rs (new: classes, defines, packing to lane records for the interim CPU pack), src/cuda_engine.rs (class routing and the W = 2 kernel key), tests/lane_kernel.rs (new). Deliverable: the W = 2 class scoring bodies of up to 8 nodes and 32 muscles from physics's specification, the lane-group kernel keeping the rest, replays through the RECORD variant of the same source. Gate: p2_speed on save42's up-to-8-node subset at least 150 M creature-steps/s at 2 substeps; first_generation free propulsion at most 0.05 m median; physics_audit ledgers at rounding level; elite re-test ratio at 4x at least 0.95 median; a fixed-seed search_ab repeat bit-equal; the same 1,000 creatures scored alone and inside a 262k wave bit-equal. Dependencies: physics's spec track (the formulation as a document), G1. Two weeks. Second.

G3, `lane-cuts`. Owns: shaders/lane_creature.cu (the three cuts and the warm-started active set behind defines), examples/p2_speed.rs (per-cut reporting). Deliverable: waveform and drive target once per step, ledger only in flight, lagged contact matrix, each merged alone. Gate per cut: p2_speed up by its estimate within 3 points, size_report slip ratio and the elite ratio unchanged within 0.01. Dependencies: G2. One week, three merges. Third, in parallel with G4.

G4, `lane-w4-w8`. Owns: the G2 files, plus deletion of shaders/warp_creature.cu, src/warp_kernel.rs, shaders/physics2_creature.wgsl and src/vk_engine.rs at the end. Deliverable: W = 4, then W = 8, then the lane-group and Vulkan kernels removed. Gate: harmonic-mean rate on save42's full ring at least 200 M creature-steps/s at 2 substeps; the G2 audits on the 9-to-16 and 17-to-32 subsets; the growth-rule dump numbers (ga) read before W = 8 is scheduled, so W = 8 is skipped only if the owner adopts a hard 16-node cap. Dependencies: G2, ga's dump. Two weeks. Fourth.

G5, `prologue-unpack`. Owns: shaders/lane_unpack.cu (new), src/cuda_engine.rs (DMA per block, the prologue launch, records in VRAM), src/lane_kernel.rs (the record layout shared with the prologue). Deliverable: the per-block prologue kernel unpacking cpu's SoA gene arena into lane records at 32 creatures per warp with pinned rounding intrinsics, take-up from VRAM, the CPU pack deleted; the one-day measurement of ruling 2 (records from VRAM, records zero-copy, genes with in-kernel unpack) reported first. Gate: lane records bit-equal to the CPU packer on save42's ring (262k); p2_speed within 2% of the CPU-packed path; packing seconds zero in the stage log. Dependencies: cpu's arena format track, data's engine track (queues, tails, completion words). Two weeks. Fifth, and it can start after G2 with the lane-group kernel as its first consumer.

G6, `substep-kernel`. Owns: the 40 and 60 kernel lines (realized-work ledger over contacts and the projection, anchored friction with the store, the spin-adaptive substep rule) behind defines in the lane-group kernel now and in lane_creature.cu once G2 merges. Deliverable: the ladder L0 to L2.5 runnable on the current kernel through EVOLUTION_WARP_* overrides; physics owns the honesty runs and their bars. Gate: physics's 0.9 median with p10 reported at each rung; on a pass, the passing rung becomes the default and its define is removed. Dependencies: physics's spec. One week of kernel work. Runs in parallel with G1 from day one, because the ladder is the largest contested lever and needs no new kernel.

G7, `kernel-compile`. Owns: src/cuda_engine.rs (prefetch), src/warp_kernel.rs then src/lane_kernel.rs (the effects-as-uniforms variant for the fat-kernel measurement), src/ui.rs (the "compiling the new world" state only). Deliverable: the three standard classes compiled in parallel on a world change, neighbour-world prefetch of the 14 single-toggle worlds at idle priority, the visible compiling state, cache eviction at 200 files, the fat-kernel measurement. Gate: a cold world change idles the GPU at most 1.5 s and a warm one 0; the fat kernel adopted only if p2_speed is within 3% of the per-world kernel. Dependencies: none. One week. Any time; small merges.

The fine-fidelity persistent kernel for data's three reserved slots is the same source at the 4x fidelity key; it is data's track, not mine.

## 3. Three measurements that would change my mind

1. G1's issue-active at 16 warps. Under 35%, or a lane-group stall sample where barrier plus short-scoreboard is under 30% of stalls, means the per-lane layout gains less than 2x from issue, and I would go to W = 1 at 255 registers and 8 warps for small bodies, or, below 25%, keep the lane-group layout with physics's direct solve inside it (1.6x) and state the ceiling at 1 M/s.
2. os's rows 11 and 12. A register-only FMA kernel at full issue holding 2.0 GHz or less at the cap means 3 T is the budget, the 5 T column is deleted, every stall-removal lever is worth nothing, and 2 M/s on this laptop needs L2 or L2.5 plus the growth rule plus the full ladder, with no margin. Above 2.3 GHz the 5 T column stands.
3. G5's one-day test. If records from VRAM and zero-copy records differ by under 3% on p2_speed, the prologue kernel is unnecessary and zero-copy comes back with 2 GB of VRAM saved; if genes with an in-kernel unpack are within 3% of records from VRAM, the prologue kernel goes too. I expect neither, and the test decides.

## 4. The ceiling

Assumptions: the per-lane kernel at its round-3 counts (harmonic mean over the generation-51 mix with the growth rule holding the tail at 1%: 295 M at 3 T, 490 M at 5 T at 2 substeps); R4 firing for 50% of survivors, so 265 steps per creature; the host tax at 18% and device helpers at 1.5%, together 0.81; L2.5 at 1.6x; the generation-100 muscle slope at 0.75 (the middle of os's 0.65 to 0.82).

| generation, substeps | 3 T | 5 T |
|---|---:|---:|
| 51, 2 substeps | 0.90 M/s | 1.50 M/s |
| 51, L2.5 | 1.44 M/s | 2.40 M/s |
| 51, L2 | 1.67 M/s | 2.77 M/s |
| 100, 2 substeps | 0.68 M/s | 1.12 M/s |
| 100, L2.5 | 1.08 M/s | 1.80 M/s |
| 100, L2 | 1.25 M/s | 2.08 M/s |

Honest reading: at 3 T, 2 M/s is out of reach on this laptop in every row. At 5 T, 2 M/s at generation 51 needs L2.5 and everything else in the design; at generation 100 it needs L2, or L2.5 with the growth rule holding the muscle slope above 0.85, which the stage-log line will show within the first long run. Without any substep change the sustained rate is 0.9 to 1.5 M/s at generation 51 and 0.7 to 1.1 M/s at generation 100. Every number in the table moves by the three measurements above; none of them is arguable further.
