# Round 3, data flow: the engine and the ring at implementation level

Design target: the generation-51 population (8.7 nodes mean, p90 12; 19.2 muscles mean, p90 29; genes 1.35 KB mean, 2 KB at p90, 6 KB for the 9% tail), on the round 2 rulings.

## 1. The pipeline in one paragraph

The worker breeds a block into a pinned host arena (SoA genes, a 24 B header per creature, a sorted order array) and submits it: the engine appends creature references to per-class device queues and bumps their tails. Persistent per-class kernels take creatures through an atomic head, read the genes zero-copy over PCIe at take-up, unpack them in registers, run the trial against the block's bar table, compute the descriptor bins and the candidate bit at the end, and write the 80 B result into a pinned results ring. The last creature of a block flips a completion word. The worker polls those words, reads results in place, asks confirmations for record candidates, commits candidates in ring order, uploads the new occupant table, breeds the block again into the same arena, and submits it. Per creature the CPU runs only the breeder and the candidate commit.

## 2. Memory layout

Block arena (pinned, cuMemAllocHost, allocated once and reused; a block that outgrows its arena gets a larger one at its next breed):

- headers, 24 B per creature: node_start, bone_start, muscle_start (u32), node, bone and muscle counts and class (u8), id (u64).
- nodes 16 B, bones 28 B, muscles 52 B, today's gene structs, SoA by kind, contiguous per creature. Capacity 2x the running mean gene bytes of the last 10 blocks: 2.7 KB per creature at generation 51.
- order: per class, creature indices sorted by (exact muscle count, node count, depth); a par_sort of 100k keys, under 5 ms.
- a world tag and the block's bar table (ga's P1: [arena][niche][rung] f32, 10 x 1,440 x 3 x 4 B = 173 KB).

Results ring (pinned, DEVICEMAP, written by the kernel): 80 B per creature, indexed (block, index). The 7 free floats carry ml's rung distances as fp16 pairs, the niche in 6 bytes, and flags: candidate, record candidate, audit lane, rung stopped, world tag.

Device side: per class a queue of 8 B refs (block, index), 8 MB at 1M creatures; a head (atomic) and a tail (in a mapped pinned page the host writes, read with volatile loads); a drain flag; per block a remaining count and a completion word in a mapped page; the occupant table [arena][niche] f32, 58 KB, 4 versions rotating; bar tables, 173 KB per block in flight; replay frames 10 MB; kernels.

Budget at a 1 s ring (1M creatures, the cap): pinned genes 1.35 GB (2 GB at p90 bodies), results 80 MB. Host RSS with the CPU archives, cpu's versioned elite table and lineage: under 2.5 GB, from 5 to 9 GB today. VRAM: under 100 MB, from 5 to 7.5 GB. If zero-copy take-up measures too slow (section 6), a VRAM mirror of the arenas adds 1.35 to 2 GB.

PCIe at 2M/s: genes 2.7 GB/s up (4 GB/s at p90 bodies), tables under 5 MB/s, results 160 MB/s down. A PCIe 4.0 x8 link carries about 12 GB/s each way: 25 to 35% up, 1.5% down. gpu's 32 B muscle record would halve the up traffic; ga's call, not a gate.

## 3. Block, wave and ring sizing from the measured rate

The engine keeps an exponential mean of creatures per second over the last 10 blocks, and the worker the p95 of its per-block host chain (poll to submit). Then block = 50 ms of GPU work, floor 32k, cap 256k (100k at 2M/s, 32k at today's 167k/s), and ring depth = clamp(5 x p95 host chain, 0.3 s, 1 s), 6 to 20 blocks in flight. There are no waves: a block is one append to the class queues, and the persistent kernels see no boundary, so a tail exists only at drains (world change, class re-split, shutdown), about 20 ms each.

RING_SLOTS, RING_BLOCKS, WORK_UNIT and WAVE become these two functions, and the first merge changes only them: at today's rate the 786k ring becomes about 5 blocks of 32k, and a button press discards 1 s of work instead of 4.7 s (26% of a generation).

## 4. The spanning counter and the persistent kernels

Per class (W = 1 small, W = 2 main, lane-group 16 and 32 for the tail) one kernel is resident. Its loop, per lane or group:

```
if (!live) {
    got = atomicAdd(head, 1)                   // one lane, then broadcast
    while (got >= volatile_load(tail)) {       // starved: the host is late
        if (volatile_load(drain)) return       // block retires
        __nanosleep(2000)
    }
    ref = queue[got]; load header, genes (zero-copy), bars[ref.block]; unpack; live = true
}
step; on end: write result to results_ring[ref]; if (atomicSub(remaining[ref.block], 1) == 1)
    { __threadfence_system(); completion[ref.block] = world_tag; }
```

The host splits 92 of the 96 block slots (24 SMs x 4) among the classes by their share of lane-steps over the last 20 blocks and reserves 4 (section 7). A re-split is a drain and relaunch, one 20 ms tail, when a share moves more than 10 points or the world changes. Starvation is a spin on a mapped word (1 us per poll, only while the queue is empty), which the ring depth prevents; the stage log records starved seconds per block. Section 10 splits each class queue by muscle count.

## 5. Tables: bars, occupants, candidates, and what is deterministic

Two tables, two rules.

The bar table changes a creature's outcome (where a rung stops it), so it must be a function of the block, not of wall time. It is bred with the block, uploaded with the block, and indexed by block. Same rule as Config today.

The occupant table only filters. A creature is a candidate if it beats its arena cell's occupant in the table, the cell is empty, or it beats the island record (so confirmations get asked). Occupants only rise and any table the kernel can see is a version at or before the block's own absorption point, so the candidate set is a superset of the true entrants whatever version was read, and the CPU commit against the real archive decides. I checked the shortcut of letting the kernel atomicMax the table itself: protection rules and failed confirmations mean a device-inserted challenger might be refused by the CPU, and the table would then over-reject. It stays CPU-written after each commit.

Determinism rules:

1. Results are addressed by (block, index), never by completion order; which lane ran a creature is irrelevant.
2. Absorption, commit and breeding run in ring order on the worker; the archive a block is bred from is the state after the previous block's commit. cpu's versioned elite table lets breed(k+1) overlap absorb(k) without changing that.
3. Everything a trial reads is fixed at breed time: genes, Config, bar table, world tag. Nothing read at trial time varies with timing except the occupant table, whose effect is result-invariant by the superset argument.
4. Atomics only where order does not matter: the take-up head, the remaining count, the candidate append. No float atomics.
5. A world change is a player action; the set of creatures live at that instant is timing-dependent, and they enter no archive, which is today's rule at finer grain. Two runs of one seed with no player action produce one history.

## 6. Take-up cost and the register question

At take-up a lane reads the header, about 1.4 KB of genes (3 to 6 KB for the tail) and its bars over PCIe, then computes what Model::new computes (masses, slack lengths, strengths, joint ranges, start pose, breadth-first order, end lists): about 2,000 instructions and 40 to 50 independent 32 B loads at 1 to 2 us, pipelined, 10 to 30 us per creature against 300 to 1,200 steps at 5 to 15 us each, so 0.2 to 1%. If the unpack pushes the trial kernel over its register budget (gpu's concern), it moves to a per-block prologue kernel writing lane records to VRAM (150 MB per 100k block, 1.5 GB for 10) and take-up loads records instead. The gate is the same: lane records bit-equal to warp_kernel::fill_creature on save42's ring.

## 7. Replays, confirmations, and the reserved slots

Persistent kernels hold their slots, so 4 of 96 are reserved (4%): 3 for a persistent fine-fidelity kernel over a confirmation queue (confirmations are 1.6% of steps by ga's count; a 4x-rate trial is about 60 ms alone) and 1 for replays. A replay click launches one block on the free slot: 12 ms of recording plus a 100 KB readback, so 20 to 30 ms click-to-replay against up to 1.5 s today. When idle, the replay slot pre-records the newest confirmed island record, so the champion's replay is on the host before the click: about 50 records per generation x 15 ms in a slot the game reserves anyway.

## 8. The world-change path

Today blocks already on an engine enter no archive (786k creatures, 4.7 s). In the design the host stops appending, writes the new bar tables, sets the drain flags, and the kernels retire as their live creatures end (about 20 ms). The un-taken refs between head and tail are re-appended to the new world's queues, whose kernels the prefetch thread compiled in the background (a cold compile is 1 to 2 s, the one case where the GPU idles). Results of creatures live at the change carry the old world tag and are excluded at absorption; their blocks are retargeted as today. Loss per button press: the live set, under 10k creatures, plus 20 ms.

## 9. Saves, the worker, and 60 FPS

A save is unchanged (archives and search state, no ring). The worker clones the Experiment between blocks (tens of ms) and serializes on another thread; the ring keeps running because the arenas belong to the engine. Loading rebreeds the ring from the archives, as today. cpu's versioned elite table is rebuilt on load.

The worker's loop per block: poll completion words (1 ms), read results in place, verdict on candidates, ask confirmations, commit candidates in position order (cpu's 30 to 60 ms per 196k, unmeasured, 15 to 30 ms at 100k), upload the occupant table, plan and breed the next block into the arena (cpu's 3.5 core-seconds per 3M is 40 ms per 100k on 3 threads), submit. About 80 to 100 ms per 100 ms block on 2 to 3 threads, so the 5x rule gives a 0.5 s ring; cpu's overlap of breed and absorb halves the chain. Commands are read between blocks, so a button acts within 100 ms.

The UI never touches the RTX, takes the snapshot by a swap under a Mutex, sends changed cells only, and gets os's scheduling changes (rayon at 14 threads, nice 10, SCHED_BATCH, the UI thread on its own SMT sibling). Gate: frame time p99 under 16.7 ms and command latency p99 under 20 ms from the existing benchmark report at full rate.

## 10. The new idea: bucketed take-up queues keep warps uniform

In a per-lane kernel every loop bound is the warp's maximum. The breeder sorts a block by muscle count, so the first take-up gives uniform warps. But lanes finish at different times (falls at 30 steps, screens at 300, survivors at 1,200) and each takes its next creature from a global counter, so after a few take-ups a warp holds 32 creatures drawn from the whole block's distribution and runs at the max of 32 samples. At generation 51 (muscles p50 17, p90 29) that max is about the 97th percentile, about 32 muscles, against a mean of 19. Muscles are about half the step at 19 muscles (gpu, physics), so the warp's step costs about 0.5 + 0.5 x 32 / 19 = 1.34x the mean body's: a 25% loss no kernel count includes, because the counts assume the sorted warp.

The fix is in the queue: split each class queue into muscle buckets of width 4, give each resident warp a bucket for the launch (assigned in proportion to the buckets' lane-steps), and let a lane take only from its warp's bucket. The warp max is then within 4 of its mean: 0.5 + 0.5 x 23 / 19 = 1.1x. Expected gain: 1.15 to 1.25x on the per-lane kernel at generation 51, more as the spread grows. Cost: 8 heads per class, and a warp whose bucket empties retires until the next re-split. It applies to today's lane-group kernel too, where 4 groups per warp run the max rounds of 4 samples: a counter per (class, rounds) is a 20-line change, and p2_speed on save42.evo with and without it is the measurement, this week, before any per-lane kernel exists.

## 11. Per-lever table

Multipliers on creatures per second at the generation-51 mix, from 43 to 45M creature-steps/s and 480 steps. The kernel and steps rows are the chair's; mine are the data-path rows.

| lever | owner | at 3T | at 5T | note |
|---|---|---:|---:|---|
| per-lane kernel, 2 substeps, harmonic mean over the mix | gpu, physics | 300M | 450M | chair's range |
| bucketed take-up queues (section 10) | data | 1.15x | 1.15x | counts as instructions, so it holds at the cap |
| spanning counter, no wave tails | data | 1.02x | 1.02x | tails are 2% per wave today, 5% at 130 ms waves |
| reserved slots for confirmations and replays | data | 0.96x | 0.96x | today's replay slot costs the same when it runs |
| take-up unpack over PCIe | data | 0.99x | 0.99x | 0.2 to 1% |
| pack, staging copy and page faults removed | data, cpu | 1.0x | 1.0x | hidden by overlap today; its value is the host tax row |
| host tax at the cap (cpu at 2 to 3 cores) | cpu, os | 0.93x | 0.93x | 5 to 8% by cpu's revised slope; 1.0x while clock-bound |
| kernel rate after the data path | | 320M | 480M | |
| steps per creature, R1 to R3 | ga, ml | 300 | 300 | conservative |
| creatures per second, 2 substeps | | 1.07M | 1.6M | row 1 |
| 1 substep (if it passes), kernel 550 to 800M | physics, owner | 1.95M | 2.85M | row 3 |
| R4 at ml's rate on top, 235 steps | ga, ml | x1.28 | x1.28 | rows 2 and 4 |

Where my design lands: row 1 (1.0 to 1.6M/s) with 2 substeps and R1 to R3, row 3 (2M/s at 3T, 2.9M/s at 5T) if 1 substep passes. The data path is not the wall on any row: with cpu's breeder at 3.5 core-seconds per 3M and the commit under 60 ms per block the host sustains about 4M/s, and PCIe about 8M/s at 1.35 KB per creature. If 1 substep fails or the kernel lands below 300M, the design does not change; the ring shrinks to 32k blocks.

Across generations the arenas grow with gene bytes, the block count follows the rate and the bucket split follows the histogram, so growth to p90 12 nodes and 29 muscles changes numbers, not code. ga's archive-following body cap keeps the 9% tail from taking a quarter of the lane-steps; the stage log makes that tail visible per block.

## 12. What stays out

Recipes on the RTX: insurance, built only if breed_bench exceeds 8,000 weighted cycles or plan_offspring 2,000. The structural port: withdrawn. The Radeon in the per-creature path: closed. A VRAM genome ring: unnecessary while zero-copy take-up costs under 1%. Device-side archive insertion: refused by section 5. CUDA events and graphs: replaced by the completion words.

## 13. Tracks in order, each with a gate, the game playable at every merge

1. Ring and block sizing by rate and latency; starved seconds and lane-steps per class in the stage log. Days. Gate: rate unchanged within 3% at 3M, world-change loss under 1 s of work, replay p95 under 1 s.
2. Bucketed take-up counters in the lane-group kernel (one per class and rounds). Days. Gate: p2_speed on save42.evo up at least 5%, results bit-equal per creature.
3. Genes written once into pinned arenas (cpu's P1), the pack deleted, unpack at take-up. Two weeks. Gate: lane records bit-equal to fill_creature on save42's ring; packing seconds zero; page faults near zero; RSS under 3 GB.
4. Persistent per-class kernels: spanning counter, completion words, results in pinned memory, reserved slots, drain path. Two weeks. Gate: two_continuous_runs_of_one_seed_agree at 3M for 20 generations; replay p95 under 50 ms; world-change loss under 10k creatures; no tails in the nsys timeline.
5. Bar tables per block, the candidate bit and occupant table, the CPU commit on candidates. One week. Gate: identical history to the full commit over 20 generations; archive seconds per generation under 0.2 s on save42.evo.
6. Delta snapshot and os's scheduling. Days. Gate: section 9.
7. The per-lane kernel plugs into the same queues and buckets (gpu's track). Gate: gpu's stub numbers, then the harmonic-mean rate on save42.

Each merge leaves cargo run --release the better game: cheaper button presses first, then the pack gone, then persistent kernels, then the kernel itself.

## 14. Engine API

As the worker sees it (ring.rs keeps its flights and ring-order absorption; scheduler.rs shrinks to one engine plus the CPU failover implementing the same trait over the same arenas):

```
trait Engine {
    fn open(world: &Config) -> Result<Self>;                 // kernels compile in the background
    fn configure(&mut self, blocks: usize, capacity: usize); // pinned arenas, results ring
    fn arena(&mut self, block: usize) -> &mut BlockArena;    // the breeder fills it
    fn submit(&mut self, block: usize) -> Result<()>;        // append refs, bump tails
    fn poll(&mut self, timeout: Duration) -> Vec<BlockDone>; // completion words
    fn results(&self, block: usize) -> &[Result];            // in place
    fn occupants(&mut self, table: &[f32]) -> Result<()>;    // new version, side stream
    fn confirm(&mut self, block: usize, members: &[usize]) -> Result<Ticket>;
    fn replay(&mut self, creature: &Creature, cfg: &Config) -> Result<Frames>;
    fn retarget(&mut self, world: &Config) -> Result<Retargeted>; // drain, requeue, relaunch
    fn rate(&self) -> Rate;                                  // EMA, lane-steps per class, starved seconds
}
```

As the kernel sees it: per class and bucket a queue of 8 B refs with a head, a tail and a drain flag; per block an arena base, a bar table, a remaining count and a completion word; the occupant table; the results ring; one Params per world. Everything is a pointer passed at launch, once.
