# Fixed-seed search benchmark

This benchmark records a behavior-only baseline and a morphology-reserve variant of the same MAP-Elites search. The search objective, physics, trial duration, mutation settings, population, and evaluation budget were held constant.

## Method

- Seeds: 38, 39, 40, 41, 42, with random seeding disabled.
- Population: 1,000 creatures; 300 generations; 300,000 evaluations per seed.
- Trial duration: 18 seconds on flat ground.
- Device: NVIDIA GeForce RTX 4060 Laptop GPU; 16 logical CPUs; throughput evaluation mode.
- The behavior archive has 192 measured behavior cells. QD score and coverage count only these cells.
- The reserve variant adds up to 64 topology entries. When entries are available, ten percent of structural-emitter offspring try a reserve parent. Each entry receives at least eight selected offspring before it can be evicted to make room for another topology; the behavior archive can absorb it earlier if a behavior elite matches or beats its distance. A new reserve entry must have higher distance than the weakest eligible entry.
- The reserve admits a changed topology from the structural or novelty emitter when its trial is valid. Descendants can improve an existing reserve entry. No body-complexity reward or triangle penalty is used.

Generation 0 is the initial population; the 300-generation budget includes it. Search time includes GPU evaluation, archive updates, and offspring creation. Benchmark time also includes lineage measurement and file output.

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

The wall-time means are not evidence of a speedup. The runs used the desktop GPU, per-seed timings varied, and body complexity changes GPU evaluation cost. The comparable claim from this run is that the reserve retained more structural alternatives at the same evaluation budget, with a higher QD score and behavior-archive median. Best-distance gains were not consistent across seeds.

Small triangles remain eligible on fitness alone. The all-time record holder was a three-node, three-muscle triangle in three of five behavior-only seeds and two of five reserve seeds. The reserve therefore expands topology retention without requiring larger bodies to win.

## Reproduce

Use new empty output directories because the command does not overwrite benchmark data:

```bash
cargo run --release -- search-benchmark \
  --variant behavior-only \
  --seeds 38,39,40,41,42 \
  --population 1000 \
  --generations 300 \
  --duration 18 \
  --output-dir /tmp/search-behavior-only

cargo run --release -- search-benchmark \
  --variant morphology-reserve \
  --seeds 38,39,40,41,42 \
  --population 1000 \
  --generations 300 \
  --duration 18 \
  --output-dir /tmp/search-morphology-reserve

cargo run --release -- analyze runs/latest.evo \
  --output /tmp/search-analysis.json \
  --champion /tmp/champion.json
```

The command writes `metadata.json`, one `curve.csv` and seed summary directory per seed, and an aggregate `summary.json`. Checkpoint analysis reports behavior-archive and reserve morphology separately and exports the current champion when requested.
