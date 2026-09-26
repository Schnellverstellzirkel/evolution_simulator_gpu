# Fixed-seed search benchmark

This benchmark records fixed-seed, equal-evaluation-budget runs of the MAP-Elites search. It can evaluate through Vulkan (the default) or the production CPU SIMD engine (`--cpu`). The search objective, physics, trial duration, mutation settings, population, and evaluation budget remain unchanged by the benchmark backend.

## Method

- Seeds: 38, 39, 40, 41, 42, with random seeding disabled.
- Population: 1,000 creatures; 300 generations; 300,000 evaluations per seed.
- Trial duration: 18 seconds on flat ground.
- The recorded runs below used the NVIDIA GeForce RTX 4060 Laptop GPU; CPU mode is available for reproducible search comparisons without Vulkan initialization.
- The behavior archive has 192 measured behavior cells. QD score and coverage count only these cells.
- The reserve variant adds up to 64 topology entries. When entries are available, ten percent of structural-emitter offspring try a reserve parent. Each entry receives at least eight selected offspring before it can be evicted to make room for another topology; the behavior archive can absorb it earlier if a behavior elite matches or beats its distance. A new reserve entry must have higher distance than the weakest eligible entry.
- The reserve admits a changed topology from the structural or novelty emitter when its trial is valid. Descendants can improve an existing reserve entry. No body-complexity reward or triangle penalty is used.

Generation 0 is the initial population; the 300-generation candidate budget includes it. The reported candidate budget is exactly `population × generations`, with one standard trial per candidate slot. The default GPU path retains scheduler contender-check trials (`EVOLUTION_ROBUST_TRIALS`, default 2), while CPU mode runs one standard trial per candidate. `Experiment::archive_batch` may additionally replay contender candidates on the CPU validation path before admitting them. Check and verifier trials are outside the candidate count and vary with contenders. Search and archive wall time includes that work. CPU and GPU runs therefore have equal candidate budgets but can differ in total physics calls and scores; compare algorithm variants on the same backend with the same trial settings. Benchmark time also includes lineage measurement and file output. Each seed summary includes the highest-scoring 50 archive elites, with distance, node and muscle counts, total rest length of bones, and starting-pose height. The same top-50 list is included in each generation's `curve.csv` row as JSON.

The baseline results are in [`search-baseline/`](search-baseline/); the final reserve run is in [`search-morphology-reserve/`](search-morphology-reserve/). Each seed directory contains a per-generation curve, a summary, emitter and parent-topology statistics, archive genealogy, innovation records, and a full creature snapshot for each all-time record.

## Results

| Metric | Behavior-only | Morphology reserve |
| --- | ---: | ---: |
| Mean best distance across seeds | 112.00 m | 112.83 m |
| Median of the five seed bests | 108.89 m | 103.04 m |
| Mean behavior-archive median | 46.21 m | 49.05 m |
| Mean QD score | 8,531 | 9,115 |
| Mean behavior coverage | 98.02% | 97.50% |
| Seeds reaching 100 m | 3 / 5 | 3 / 5 |
| Seeds reaching 150 m | 2 / 5 | 1 / 5 |
| Mean distinct behavior-archive topologies | 31.6 | 36.6 |
| Reserve topologies at generation 300 | 0 | 64 per seed |
| Mean first-time structural topology admissions | 136.6 | 522.2 |
| Mean three-generation innovation survival | 91.7% | 82.6% |
| Mean search wall time per seed | 11.33 s | 10.59 s |

The paired best-distance changes were mixed: seed 38 gained 87.42 m, seed 39 lost 99.35 m, seed 40 lost 5.85 m, seed 41 gained 17.85 m, and seed 42 gained 4.08 m. Mean best distance was nearly unchanged, and the median seed best fell from 108.89 m to 103.04 m; the mean behavior-archive median and QD score increased. The reserve admitted about 3.8 times as many first-time structural topology families and finished with 64 distinct node-indexed topologies per seed. The archive graph count is exact by node labels and sorted undirected edge multiset; graph-isomorphic bodies with different node labels count separately.

The wall-time means are not evidence of a speedup. The runs used the desktop GPU, per-seed timings varied, and body complexity changes GPU evaluation cost. The comparable claim from this run is that the reserve retained more structural alternatives at the same evaluation budget, with a higher QD score and behavior-archive median. Best-distance gains were not consistent across seeds. These results predate later physics changes (including bone mass, pull-only muscles, and replay-based archive validation); treat them as historical only, and rerun comparisons under the current physics before drawing search conclusions.

Small triangles remain eligible on fitness alone. The all-time record holder was a three-node, three-muscle triangle in three of five behavior-only seeds and two of five reserve seeds. The reserve therefore expands topology retention without requiring larger bodies to win.

## Reproduce

Use new empty output directories because the command does not overwrite benchmark data. CPU mode initializes no GPU or Vulkan device. `EVOLUTION_CPU_THREADS` controls the CPU evaluation pool; `RAYON_NUM_THREADS` controls archive and other global Rayon work. Both are capped at eight threads by the runner:

```bash
EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 RAYON_NUM_THREADS=8 nice -n 10 cargo run --release -- search-benchmark \
  --variant behavior-only \
  --seeds 38,39,40,41,42 \
  --population 1000 \
  --generations 300 \
  --duration 18 \
  --output-dir /tmp/search-behavior-only

EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 RAYON_NUM_THREADS=8 nice -n 10 cargo run --release -- search-benchmark \
  --variant morphology-reserve \
  --seeds 38,39,40,41,42 \
  --population 1000 \
  --generations 300 \
  --duration 18 \
  --output-dir /tmp/search-morphology-reserve

EVOLUTION_DEVICES=primary RAYON_NUM_THREADS=8 EVOLUTION_CPU_THREADS=6 nice -n 10 cargo run --release -- search-benchmark \
  --cpu \
  --variant behavior-only \
  --seeds 38,39,40,41,42 \
  --population 1000 \
  --generations 300 \
  --duration 18 \
  --output-dir /tmp/search-cpu-behavior-only

EVOLUTION_DEVICES=primary RAYON_NUM_THREADS=8 EVOLUTION_CPU_THREADS=6 nice -n 10 cargo run --release -- search-benchmark \
  --cpu \
  --variant morphology-reserve \
  --seeds 38,39,40,41,42 \
  --population 1000 \
  --generations 300 \
  --duration 18 \
  --output-dir /tmp/search-cpu-morphology-reserve

nice -n 10 cargo run --release -- analyze runs/latest.evo \
  --output /tmp/search-analysis.json \
  --champion /tmp/champion.json
```

For a new search-algorithm change, use paired runs with the same seed list, population, generation count, configuration, backend, and trial settings on the baseline and changed revisions. Check out each revision in turn, save output to distinct empty directories, and compare per-seed best distance and QD score alongside the top-50 body-size lists; do not compare wall times across machines or backends. CPU random streams and breeding are seed-controlled and per-creature, so Rayon scheduling does not affect candidate generation. CPU/GPU floating-point behavior can still differ slightly, and runs on different backends are not an exact paired physics comparison. Candidate budgets remain equal even if contender counts, and therefore check-trial and archive-verification replay counts, differ between variants.

For a quick CPU smoke run before a full comparison:

```bash
EVOLUTION_DEVICES=primary RAYON_NUM_THREADS=2 EVOLUTION_CPU_THREADS=2 nice -n 10 cargo run --release -- search-benchmark \
  --cpu --seeds 38,39,40,41,42 --population 8 --generations 2 --duration 1 \
  --output-dir /tmp/search-cpu-smoke
```

The command writes `metadata.json`, one `curve.csv` and seed summary directory per seed, and an aggregate `summary.json`. Checkpoint analysis reports behavior-archive and reserve morphology separately and exports the current champion when requested.
