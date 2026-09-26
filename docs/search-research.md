# Faster and better creature evolution: research notes

September 2026. The search is MAP-Elites (Mouret and Clune, 2015) with CMA, structural, novelty, and random-immigrant emitters over four island archives. This note reviews the published research that applies to it, measures where the current search spends its evaluations, and ranks changes by expected payoff. "Faster" means more distance per evaluation, and also more evaluations per GPU-second. "Better" means complex bodies that are also fast, rather than a search that settles on 3–6-node bodies.

## How the numbers here were measured

All measurements use the CPU SIMD engine (`cpu_engine::evaluate`) on a 4-core AVX-512 container, driving the generational loop (`archive_batch` → `prepare_next_batch`) with the game's defaults: 60 s trials, 60 Hz physics, the perturbed second trial, 4 islands, all emitters. Each run has 4,000 creatures and 61 generations (244,000 evaluations) with seeds 38, 39, and 40. The throwaway probe harness and its temporary switches were not committed.

The game runs 3 million creatures in steady state on the GPU. At that population each archive cell gets far more offspring per generation, so per-offspring insertion rates are lower than here. The waste measured below is therefore a lower bound for the game. Fall rates and mutation effects depend on the operators, not on the population, and should carry over.

## Ranked recommendations

| # | Change | Why | Evidence | Effort |
|---|---|---|---|---|
| 1 | Stop simulating a creature once it falls | Fitness is frozen at the fall, but the trial runs on | 53% of offspring fall, 65% of those within 2 s; 43–55% of simulated time is after a fall | Medium (GPU lane compaction) |
| 2 | Run the perturbed trial only for archive contenders | Score is min(A, B) ≤ A, so a creature whose first trial cannot enter its niche never needs B | 18% of creatures qualify on trial A on average, falling to 11–16% late in a run | Small |
| 3 | Stop funding random immigrants; shrink novelty jumps | Immigrants and scale-0.75 mutants almost never enter the archive | Immigrants: 0.03–0.06% insertion; novelty children keep a median 6–20% of the parent's distance | Small |
| 4 | Make structural mutations near-neutral | New parts start fully active, so a new body starts at a few percent of its parent's distance | Median distance retained after a split: 1–2%; 68–85% of split children fall | Small–medium |
| 5 | Tune new bodies before judging them, and compare bodies at equal tuning age | Complex bodies lose to tuned simple ones before their controllers adapt | 256 evaluations lift a new limb from 29% to 78% of its parent, but the parent reaches 120% with the same budget | Medium |
| 6 | One body clock per creature | Independent periods drift apart; gaits are not periodic | A/B: +32% best distance on every seed, and 7+-node bodies make up 10–38 of the top 50 instead of 1–3 | Small |
| 7 | Fewer, coarser archive cells, with body size as an axis | More offspring per cell; large bodies keep their own lineages | A/B: dropping the bounce axis gave +27%; swapping it for body size gave +24% (5 seeds) and faster large bodies | Small |
| 8 | Spend more on the best elites (CMA-MAE-style thresholds, curiosity-based parent choice) | A 256-evaluation hill climb improves top elites by a median 10–22% under the game's own fitness | Measured below | Medium |
| 9 | Short trials first, long trials for survivors (successive halving) | 10 s ranks predict 60 s ranks (Spearman 0.89–0.94) | Measured below | Medium |
| 10 | Repeated, linked body parts and limb-local controllers | Generative encodings evolve larger, faster bodies than direct encodings in the literature | Literature | Large |

Items 1, 2, and 9 change cost per evaluation; the others change distance per evaluation. The two multiply.

## 1. Where evaluations go today

### About half of all offspring fall, almost immediately

From generation 10 on, 53% of creatures fell on average (33–64% by generation): the head dropped below the neck base. The median fall time was 1.3 s into a 60 s trial; 65% of falls happened in the first 2 s and 81% in the first 5 s. The physics then keeps simulating the limp body for the rest of the trial, and the perturbed second trial does the same. From 43% to 55% of all simulated time in trial A comes after the creature has already fallen. The GPU kernel zeroes muscle drive after a fall (`shaders/physics_creature.wgsl`, `metrics.fall_time > 0.0`) but still runs every step.

The score of a fallen creature is fixed at the fall (`fall_x`), so stopping early loses nothing on fitness. Only the behavior descriptors change: contact, height, bounce, cadence, and feet are averaged over the whole trial, including the time spent lying down. Averaging them over the time before the fall would be a more honest descriptor anyway.

### The second trial matters for about one creature in seven

Fitness is the lower of trial A and a trial B with nodes shifted up to 2 cm and friction varied by 10% (`src/scheduler.rs:292`). Because min(A, B) ≤ A, a creature whose trial-A score does not beat the elite of its niche (in its island or the global archive) cannot enter the archive whatever B scores. Across the seven runs that logged it, 18% of creatures qualified on trial A on average (10–36% by generation), falling to 11–16% over the last 10 generations. The other 82% of B trials could not change any archive decision. The game breeds 750 times more creatures per generation into a similar number of cells, so its qualifying share is likely much smaller.

The two trials agree well: their Spearman rank correlation was 0.82–0.90.

### Emitter yield

Share of offspring that entered an archive (new cell or improvement), generations 10–60:

| Emitter | Seed 38 | Seed 39 | Seed 40 | Share of offspring |
|---|---:|---:|---:|---:|
| Diagonal CMA-ES | 12.5% | 11.8% | 10.6% | ~30% |
| Morphology (structural) | 8.7% | 8.1% | 7.5% | ~30% |
| Novelty | 2.3% | 2.2% | 2.0% | ~25% |
| Random immigrant | 0.06% | 0.06% | 0.03% | ~14.5% |

Random immigrants made 9–17 archive entries out of about 29,000 attempts per seed. They cannot compete with tuned elites in the same archive (see ALPS below for how to make immigrants useful). The bandit keeps shares near the prior because half of each prior is a fixed floor (`src/qd.rs:835`), so immigrants never drop below about 7.5% of the budget.

### Ten seconds predict sixty

For the same creatures, the rank correlation between distance after 10 s and after 60 s (trial A) was 0.89–0.94 from generation 10 on. Keeping only the top 20% by 10 s distance retained 92–100% of the creatures that were in the top 5% at 60 s; keeping the top 50% retained 99–100%. Archive insertions are less predictable, because many are low-fitness cells being filled: the top 50% at 10 s contained 78–87% of them. By 10 s, 40–53% of creatures had already fallen, yet those contributed only 12–25% of insertions, all with a fitness that was already final.

### Elites are far from a local optimum

A plain (1+32) hill climber, 8 rounds (256 evaluations) of small mutations on each of the 40 best elites, improved their unperturbed-trial distance by a median 17–23% in all three seeds. Scored with the game's own fitness (the lower of the normal and perturbed trials), the tuned elites were still better by a median 10%, 13%, and 22%, and 27, 38, and 37 of the 40 improved. The search spreads its effort across roughly 4,000 cells and leaves easy gains at the top.

### Structural mutations destroy gaits

For the 150 best elites of each seed, each operator was applied 8 times and the child was evaluated for 60 s:

| Operator | Median distance kept | Children ≥ 90% of parent | Children that fall |
|---|---:|---:|---:|
| `split_bone` | 1–2% | 0.1–0.7% | 68–85% |
| `split_bone`, neutral* | 1–6% | 1–5% | 45–73% |
| `duplicate_mirrored_node` | 2–14% | 0.2–0.5% | 46–77% |
| `duplicate_mirrored_node`, neutral* | 13–19% | 3–5% | 12–31% |
| `duplicate_limb` | 1–12% | 1–2% | 43–55% |
| `duplicate_limb`, neutral* | 13–37% | 4% | 13–26% |
| `sync_rhythm` | 6–74% | 10–29% | 18–41% |
| `phase_shift_group` | 99% | 77–82% | 9–14% |
| local mutation, scale 0.035 (after each structural op) | 97% | 75–76% | 6–10% |
| local mutation, scale 0.12 (CMA fallback) | 85–87% | 40–42% | 12–19% |
| local mutation, scale 0.75 (95% of novelty offspring) | 6–20% | 1–2% | 31–43% |

\* Neutral: the new joint starts rigid (zero range) and every muscle the parent did not have, including the ones `repair` adds to close the motor ring, starts with zero stroke (`short = long`, which gives zero motor force).

Every structural operator starts a new body at a small fraction of its parent's distance. Neutralizing the new parts roughly halves the falls for duplications, but a rigid split still breaks most gaits. A lighter mid node did not help, so mass is not the cause. The likely causes are the new node's contact sphere in the middle of a bone that used to clear the ground, and the compliance of a joint "locked" by two projection passes; this was not isolated further.

Two more properties of the operators matter:
- There is no operator that removes a node or a muscle, or adds a muscle between existing bones (the legacy `mutate` had these; the emitters do not use it). Bodies can only grow, and a bad growth step can only be undone by picking the parent again.
- `split_bone` creates its second half with `Bone::new` (`src/evolution.rs:1348`), so the new joint has the full ±120° range and no muscle across it: a rigid bone becomes a free hinge.

### New bodies cannot catch up at equal budgets

Starting from the top-40 elites, one structural child each, then 256 evaluations of the same hill climber on the child, and separately on the unchanged parent:

| Child | Before tuning | After 256 evaluations | Parent after the same 256 | Tuned child beats tuned parent |
|---|---:|---:|---:|---:|
| `duplicate_limb`, neutral | 3–39% | 57–86% | 117–123% | 0–2 of 40 |
| `duplicate_mirrored_node`, neutral | 12–17% | 50–68% | 117–123% | 0–1 of 40 |
| `split_bone` | 0–1% | 5–16% | 117–123% | 0 of 40 |

This is the "fragile co-adaptation" of body and controller that Cheney et al. (2018) and Mertan and Cheney (2024, 2025) describe: a morphology change breaks a controller tuned for the old body, so the new body is judged before its controller has adapted, and selection keeps the simple, well-tuned body.

### Complex bodies stay rare and slow

After 61 generations, the 50 best elites had 3–7 nodes in every seed. Bodies with 8 or more nodes reached at most 29–40 m, against 56–61 m for the best. Of 3,800–4,400 filled cells, 75–85 held bodies with 9 or more nodes. Mean offspring size grew only from about 4.9 to 5.4–5.8 nodes.

### Muscles do not share a rhythm

Among the 50 best elites, the number of distinct muscle periods divided by the number of muscles was 0.98–1.0: almost every muscle runs at its own period. `local_mutation` perturbs each period independently (`src/evolution.rs:1266`), so a creature synced by `sync_rhythm` is desynced by its next mutation. With independent periods a gait never exactly repeats, which makes coordination of many legs hard to evolve and makes short trials less predictive.

## 2. A/B tests of five cheap changes

Each variant changes one thing and runs the same budget as the baseline (4,000 creatures, 61 generations, 60 s trials, seeds 38–40; seeds 41 and 42 were added for the body-size axis). Distances are the archive's best after the last generation.

| Variant | Best distance, seeds 38 / 39 / 40 (m) | Mean | Mean best at generation 30 | Best body with 8+ nodes (mean) | Top-50 bodies with 7+ nodes |
|---|---|---:|---:|---:|---|
| Baseline | 56.4 / 61.2 / 56.0 | 57.8 m | 49.4 m | 34.0 m | 1 / 3 / 1 |
| One body clock | 74.8 / 85.1 / 68.9 | 76.3 m | 54.1 m | 62.2 m | 23 / 10 / 38 |
| Body size replaces the bounce axis | 71.9 / 90.1 / 76.7 | 79.6 m | 66.9 m | 48.6 m | 4 / 2 / 4 |
| Bounce axis removed (control) | 81.5 / 72.7 / 66.3 | 73.5 m | 57.1 m | 37.0 m | 0 / 3 / 16 |
| No random immigrants | 69.6 / 68.7 / 50.0 | 62.8 m | 44.4 m | 37.9 m | 0 / 0 / 5 |
| CMA initial σ 0.03 instead of 0.12 | 43.4 / 30.0 / 30.1 | 34.5 m | 27.1 m | 18.2 m | 0 / 9 / 16 |

- **One body clock** (every muscle shares one period, mutated as a whole; phases stay free) improved every seed, by 32% on average. It is the only change that made complex bodies competitive: the fastest body with 8 or more nodes went from 34 m to 62 m, and 10–38 of the 50 best elites had 7 or more nodes, against 1–3 in the baseline.
- **Body size as an archive axis** (bins for 3, 4, 5, 6–7, 8–10, and 11+ nodes, replacing vertical bounce) gained 24% over five seeds (75.8 m against 61.1 m; better in 4 of 5). The control that simply drops the bounce axis gained 27% over three seeds, so most of the gain comes from having fewer cells, each getting more offspring, not from body-size diversity. The body-size axis still produced faster large bodies than the control (48.6 m against 37.0 m for 8+ nodes). The champion itself was a 3-node body in 2 of 3 seeds, so this axis alone does not make complex bodies win.
- **No random immigrants** gained 9% on average but lost on one seed: inconclusive at three seeds. The insertion data (0.03–0.06% per immigrant) remains the stronger argument.
- **A smaller initial CMA step** everywhere made search worse by 40% and made 69% of offspring fall. Shrinking all dimensions is wrong; only the badly scaled ones (node position, period) should shrink.

The seed-to-seed spread is large (the baseline alone ranged 55–77 m over five seeds), so differences under about 15% need more seeds.

## 3. Faster: ideas in detail

### F1. Stop simulating fallen creatures

Fitness is final at the fall, so stopping is lossless for the score. On the GPU, a lane that returns early does not free its warp while other lanes still run, so the saving needs compaction: at a dispatch boundary (every 64 steps, about 1 s), a small kernel writes the indices of live creatures and the next dispatches run only those (indirect dispatch), or the host runs the trial in two phases and re-packs survivors after the first 5–10 s. With most falls in the first 2 s, one compaction after about 2–5 s captures most of the saving. Computing descriptors up to the fall (and marking fallen creatures as such) keeps "lying down" from filling behavior cells.

Related work: early termination on falling is standard in RL locomotion benchmarks, and Arza, Le Goff, and Hart (2024) show a problem-independent early-stopping rule for direct policy search that saves up to 75% of computation.

### F2. Evaluate the perturbed trial only for contenders

Run trial A for everyone. Offer each result to the archive with a flag: if A beats (or opens) the niche in the creature's island or the global archive, queue trial B and decide with min(A, B); otherwise reject immediately. This is exact for archive decisions. The only change is CMA ranking for rejected samples, which would use A instead of min(A, B); ranking every sample by A keeps it consistent. Racing algorithms (Heidrich-Meisner and Igel, 2009) generalize this: spend extra trials only where they can change a decision.

### F3. Successive halving over trial length

Evaluate every creature for 10 s, then continue only the creatures that could still matter: all that have not fallen, or the top half of those. The data above says the top half at 10 s holds essentially all eventual top-5% creatures. This is Hyperband/successive halving (Li et al., 2018) applied to trial length. Because fatigue (15 J capacity, 0.25/s recovery) only bites in long trials, keep the full 60 s for anything that can enter the archive. Idea F9 (one body clock) makes short trials more predictive, because a periodic gait's first cycles represent the rest.

Cost estimate in units of one 60 s trial per creature. Today: 2.0 (A and B). Stopping at the fall cuts trial A to about 0.5 (49% of its simulated time is after a fall), and running B only for the 18% of contenders adds about 0.18: about 0.7, or 2.9 times the evaluations per GPU-second, with no change to any archive decision. A 10 s first stage that continues only the better half of the creatures still standing brings this to about 0.5 (4 times), at the cost of skipping some low-fitness cell fills. These are estimates from the measured fractions, not timed implementations.

### F4. Reallocate immigrants and novelty

Random immigrants are 14.5% of the budget and yield almost nothing. Options, from simplest:
- Set the immigrant share to near zero once the archive has, say, 100 cells, and drop the prior floor for emitters whose reward stays near zero.
- Make immigrants useful the ALPS way (Hornby, 2006): random creatures enter a separate young layer and compete only with creatures of similar age (generations since their lineage started), moving up as they age. Hornby shows this keeps an EA from converging prematurely because fresh lineages get time to improve before facing old ones.

Novelty offspring use mutation scale 0.75 (2.25 in 5% of cases, `src/evolution.rs:1048`), which keeps a median 6–20% of the parent's distance and makes 31–43% fall. Novelty search needs offspring that behave differently, not broken ones. A smaller scale (0.12–0.25), or novelty-driven parent choice with ordinary mutations, is likely to produce more new cells per evaluation.

### F5. Spend more on the best

MAP-Elites spreads effort over thousands of cells. That is good for diversity but slow for the single fastest creature: the hill-climb result shows large easy gains left at the top. CMA-MAE (Fontaine and Nikolaidis, 2023) replaces "improve on the cell's elite" with "improve on the cell's threshold", where the threshold rises by a learning rate α toward each accepted score. With α = 1 it is CMA-ME; with α = 0 it is CMA-ES on the objective. The authors show that it addresses CMA-ME's habit of "prematurely abandoning the objective in favor of exploration" and report state-of-the-art QD performance. The CMA ranking here (`src/storage.rs:480`) puts every new cell (1e6 + score) ahead of any improvement, which is the CMA-ME behavior CMA-MAE was designed to fix.

Two simpler options:
- Choose parents by "curiosity" (Cully and Demiris, 2018): each elite's score rises when its offspring enter the archive and falls when they do not, so the search learns which parents are productive. Their selection mechanism outperformed every other variant they tested.
- Dedicate a fixed share (for example 10%) to an optimizing emitter that samples parents only from the top 1% of elites by distance and uses small mutations. JEDi (Templier et al., 2024) formalizes this: it uses the behavior map only to find promising regions and then spends evaluations on ES there, and it beats both QD and ES when the goal is the highest fitness.

### F6. Rescale and complete the CMA emitter

The CMA emitter samples in coordinates normalized to fixed ranges (`src/qd.rs` `parameters`), starting with σ = 0.12 (`src/qd.rs:887`). In physical units that initial step is a standard deviation of 0.96 m in node x, 0.48 m in node y, 0.12 m in node diameter (the whole allowed range is 0.06–0.12 m), 0.24 m in bone length, 1.1 s in muscle period, and 14 in stiffness. The emitter's fallback mutation at scale 0.12 uses 1.2 cm, 1 cm, 0.3 cm, 0.4 cm, 0.024 s, and 1.2% respectively. CMA's step-size control shrinks σ over updates, but each new CMA emitter (there are up to 96, restarted when stale) starts large again, and the CMA-ME ranking rewards the large steps that land in new cells. Suggestions:
- Do not simply shrink σ: a uniform σ = 0.03 did 40% worse than the default in the A/B test above. Instead, initialize per-dimension variances from sensible physical steps (a few cm, a few hundredths of a second), and normalize by the configured bounds instead of fixed ranges (diameter currently maps 0.01–1.0 m).
- Parameterize the pose by bone angles instead of node x/y plus rest length; `repair` re-derives node positions from bone directions and clamps rest length to ±25% of node spacing, so x, y, and rest length are partly redundant.
- Include the joint ranges and touchdown reset phases, which CMA currently leaves at the template's values.

### F7. Directional variation between same-plan elites

Iso+LineDD (Vassiliades and Mouret, 2018) mutates x_i by adding isotropic noise plus a random multiple of (x_j − x_i), where x_j is another elite. Elites of different cells often share much of their genome, and on the paper's benchmarks (a toy function, a redundant arm, and a hexapod) this operator sped MAP-Elites up substantially, in one case reaching in 10,000 evaluations what line variation needed 60,000 for. The code already groups elites by body plan for crossover (`src/storage.rs:856`); the same groups can drive this operator. Hutchinson et al. (2026) add discrete gene-level crossover between elites to QD mutation operators and report higher QD score, coverage, and maximum fitness on three locomotion tasks, especially late in a run.

### F8. Revive self-adaptive mutation

`local_mutation` mutates each creature's `mutability` gene (`src/evolution.rs:1280`) but never uses it; only the legacy `mutate` did. Either use it (multiply the scale by it, so lineages near a cliff can learn small steps) or remove it.

### F9. One body clock

The A/B test above supports this directly: a shared period gained 32% and made complex bodies competitive. Give each creature one base frequency and give each muscle a phase and, optionally, a small integer frequency multiple. This removes one dimension per muscle, guarantees a periodic gait, makes limb duplication with half-period offsets meaningful, and supports F3. Central pattern generators with a shared clock or coupled phases are the standard model for animal and robot locomotion (Ijspeert, 2008), and phase-coupled oscillators beat independent sine waves on evolved 2D creatures (Veenstra et al., 2023). The touchdown reset already present is a phase-resetting reflex in that model.

## 4. Better: complex bodies that win

### B1. Add structure that starts neutral

NEAT (Stanley and Miikkulainen, 2002) adds a node by splitting a connection so the network's function is initially unchanged, and new connections start with small weights. The equivalent here:
- a split's new joint starts rigid (zero range) and opens by later mutation;
- every new muscle, including the ones `repair` adds to close the motor ring, starts with zero stroke (`short = long`) and copies an existing muscle's period;
- a duplicated limb copies its source's muscles in phase (not antiphase) or with zero stroke, so the gait is initially unchanged and later mutations can shift it.

The measurements show this helps duplications a lot (falls from 43–77% to 12–31%), but not splits; a split that keeps the new node out of ground contact until its joint opens, or a leaf-growth operator in place of mid-bone splits, may be needed. Adding shrink operators (remove a leaf node, merge a bone, remove a muscle) and an "add muscle between existing bones" operator makes the body-plan search reversible.

### B2. Tune a new body before judging it

A body is only as good as its controller. The literature consistently finds that giving each new body a learning period before selection helps. Luo et al. (2022) show that an infant learning period "can greatly increase task performance and reduce the number of generations required to reach a certain fitness level". Lamarckian inheritance, where the learned controller is written back into the genome, helps further: newborn robots "have a higher fitness because their inherited brains match their bodies better" (Luo et al., 2023), and inheriting optimized parental controllers bootstraps infant learning (Jelisavcic et al., 2019). Learning need not start from the parent alone: de Bruin et al. (2026) start each robot's controller learning from optimized parameters of peers, preferably morphologically similar ones, and this "clearly outperforms learning from scratch under equivalent computational budgets". Two cautions: short learning budgets "systematically underestimate true potential and bias selection towards fast learners" (Song et al., 2026, whose AdaControl allocates "minimally sufficient" learning and cuts computation by up to 80% compared with exhaustive learning), and 256 evaluations were not enough here for a new body to catch its tuned parent.

On this GPU, a nursery is cheap: a new topology gets its own small CMA-ES over controller parameters (phases, strokes, stiffness, reset phases) for a few hundred evaluations, runs as one batch of same-plan creatures, and only its best result is offered to the archive. That fits the existing per-topology CMA emitters.

### B3. Compare bodies at equal tuning age

Cheney et al. (2018) show that "morphological innovation protection" (lowering selection pressure on recently changed bodies until their controllers readapt) is what lets morphology and control be co-optimized at scale. They implement it with age-fitness Pareto selection where age resets on a body change. ALPS layers do the same by age. Here, the current protection only stops a different topology from replacing a protected elite for 3 generations; it does not help a new body enter a cell, nor ensure it gets offspring.

A concrete design: each topology carries an age (evaluations spent on it since it appeared). A new topology may enter a cell if it beats the elite's score at the same age (record each lineage's best score after N evaluations), or it lives in a young archive whose cells are compared only among young topologies, with graduation to the main archive when old enough. Mertan and Cheney (2025) find that current algorithms "regularly undervalue the fitness of individuals with newly mutated bodies" and get stuck one mutation from better bodies; age matching targets exactly that.

### B4. Put body size or limb count in the archive

The archive axes are all behavioral: contact, cadence, bounce, height, and feet (`src/qd.rs:190`). A 12-node body therefore competes in the same cell as a tuned triangle with the same gait. Lehman and Stanley (2011) get many functional morphologies in one run by applying novelty to morphology and local competition among similar morphologies; Nordmoen et al. (2021) find that MAP-Elites evolves both the highest-performing and the most morphologically diverse modular robots, and that its lineages pass through more diverse and higher-performing stepping stones than objective-based search. Options: replace one behavior axis with a body-size axis (tested above: most of its gain came from the coarser grid), or keep a second archive keyed by morphology (node-count bins × limb count × symmetry) that shares parents with the behavior archive. Morphological niches alone are not enough, though. Mertan and Cheney (2025, GECCO) show that MAP-Elites with morphological descriptors still suffers from fragile co-adaptation: the body mutation that moves a creature into a new niche breaks its controller, so it rarely beats the elite already there. Their fix, periodically giving creatures a controller distilled to work across many bodies, "increases the success of body mutations and the number of migrations". So a body-size axis works best together with B1–B3 and B6. If more axes are wanted than a grid can hold, Dominated Novelty Search (Bahlous-Boldi et al., 2025) implements local competition as a fitness transformation with no grid bounds and outperforms grid-based QD on standard benchmarks. The current 64-entry topology reserve is a weak form of a morphology archive: it keys on the exact labeled graph (isomorphic bodies count separately) and evicts entries after 8 offspring, far too few to tune a new body.

### B5. Encode repetition and symmetry

Direct encodings make each new part an independent set of genes that must be tuned separately. Generative encodings reuse genes: Sims (1994) used a directed graph with recursive parts; Hornby and Pollack (2002) used L-systems and found their generative representation "rapidly produces robots with significantly greater fitness" than a direct one, by reusing assemblies of parts; CPPN encodings produce regular, symmetric soft robots (Cheney et al., 2013). The one comparison on 2D virtual creatures is a caution against network encodings: Veenstra and Glette (2020) found that a direct encoding and an L-system "generated more fit solutions" than CPPN and cellular encodings, with the L-system making larger jumps across body space; and Veenstra, Olsen, and Glette (2022) found that the encoding "accounted for a larger performance discrepancy" than MAP-Elites versus a standard EA. That points to L-system-like repetition rather than CPPNs. For this 2D world, a segment gene with a repeat count (a spine segment with a pair of legs, repeated N times), linked copies (a mutation to a leg gene changes every copy), and a phase offset per copy (a traveling wave) would let the search add a whole coordinated leg pair in one step.

### B6. Controllers that belong to limbs

Mertan and Cheney (2023) show that modular controllers (each body part runs its own copy of a small controller) lose less performance after a body mutation than one global controller. Neural Graph Evolution (Wang et al., 2019) shares a graph network controller between parent and child bodies so a new body starts with a working controller. On 2D virtual creatures, Veenstra, Szorkovszky, and Glette (2023) found that phase-coupled oscillators, where a limb's oscillator is modulated by the one above it, "gives significantly better performance than a simple wave" with both a direct and an indirect encoding. Here, each leaf limb could carry its own oscillator module (phase relative to the body clock, stroke, reset reflex), so a duplicated limb brings a working controller with it.

### B7. Make the world ask for complexity

Auerbach and Bongard (2014) find that selection for locomotion drives morphological complexity up over time, and that when complexity has a cost, more complex bodies evolve in more complex environments. On flat ground a 4–6-node body may simply be the best design. The "Roughen the ground" levels, steps, slopes, or a curriculum that raises difficulty as the archive stalls (POET: Wang et al., 2019) give extra legs a reason to exist. Evaluating trial B on different terrain (rather than a 2 cm pose shift) would also make robustness mean something. Bongard (2011) points to a related lever: robots that change body form during their lifetime early in evolution (from snake-like to legged) evolved gaits for the legged form faster, and more robust ones.

### B8. Crossover across body plans

Crossover needs identical body plans today (20% of structural and novelty offspring). NEAT's historical markings let two different-but-related bodies line up their shared parts. With innovation ids on nodes, bones, and muscles, a child can take a limb from one parent and the rest from the other.

### B9. Periodic extinctions per island

Lehman and Miikkulainen (2015) find that mass extinctions in divergent search select for evolvable lineages and lead to better walking gaits. With four islands, occasionally clearing a random region of one island's archive would be cheap to test.

### B10. Lucky elites

Fitness is noisy (trial A vs B rank correlation 0.82–0.90). Min-of-two is a pessimistic estimate, but an elite that got lucky on both trials still blocks its cell forever. Deep-Grid MAP-Elites (Flageat and Cully, 2020) keeps several candidates per cell so a lucky one can be displaced; re-evaluating elites with fresh perturbations when they are selected as parents is a simpler alternative.

## 5. Small issues found

- The island-count override reads the environment variable `EVOLUTION_island_count()` (`src/storage.rs:172`), a search-and-replace leftover; `EVOLUTION_ISLANDS` (documented in the island commit) is never read.
- `mutability` is mutated but unused (F8).
- CMA's normalized ranges ignore the configured body bounds, and CMA leaves joint ranges, sensors, and reset phases fixed (F6).
- The README still describes 18 s trials and a 192-cell archive; the defaults are now 60 s trials, 3 million creatures, and 8,640 cells.

## 6. How to test these

Use `search-benchmark` with fixed seeds, the same evaluation budget, and at least 5 seeds per variant (seed-to-seed spread here was ±10–15%). Report best distance against evaluations and against GPU-seconds separately, and add body-complexity metrics: best distance among bodies with at least 8 nodes, and the node-count distribution of the top 50 elites. Turn one change on at a time, starting with the lossless cost changes (F1, F2), which make every later experiment cheaper.

### Search A/B harness

`examples/search_ab.rs` is the committed, CPU-only harness for quick paired checks. It runs the game's production loop on fixed seeds (`cpu_engine::evaluate` for every creature, then `archive_batch` and `prepare_next_batch`, the same path `src/worker.rs` uses), so a search change guarded by an environment flag is measured with the same command before and after. It prints one row per seed and generation, then a per-seed summary (archive best distance, QD score, archive cells, and the node-count, total-bone-length, and mass mix of the generation's top 50) and a paired mean/median across seeds. Wall time goes to stderr, so stdout is deterministic and can be diffed between runs.

Arguments are generations, population, trial seconds, and a comma-separated seed list; the defaults are 2, 64, 1.0, and 38,39. `--tag NAME`, or a leading positional tag, labels the run.

    nice -n 19 env CARGO_BUILD_JOBS=4 EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 \
        cargo run --release --example search_ab -- 2 64 0.5 38,39

One tiny run finishes in under a second:

    search_ab untagged: 2 generations, population 64, 0.50 s trials, seeds 38,39
    untagged seed generation best_m qd_score cells
    untagged 38 0 0.10 0.11 10
    untagged 38 1 0.38 0.92 33
    untagged seed 38 summary: best 0.38 m, qd 0.92, cells 33
    untagged seed 38 top-50 node mix: 3x4 4x10 5x30 6x6
    untagged seed 38 top-50: median length 1.22 m, median mass 2.24 kg, longest bone 2.00 m
    untagged 39 0 0.32 0.50 16
    untagged 39 1 0.56 3.22 36
    untagged seed 39 summary: best 0.56 m, qd 3.22, cells 36
    untagged seed 39 top-50 node mix: 4x17 5x29 6x4
    untagged seed 39 top-50: median length 1.29 m, median mass 2.18 kg, longest bone 1.95 m
    paired across 2 seeds: best distance mean 0.47 m, median 0.56 m; qd score mean 2.07, median 3.22

## 7. Reaching a kilometer in a minute

Goal: evolved runners that cover 1 km in 60 s (16.7 m/s), at least 10× the 57.8 m best of the original search, without relying on solver errors. "Verified" below means the worst of 9 trials at 240 Hz with 8 bone and 4 velocity passes: one from the evolved pose and eight from poses perturbed by up to 2 cm, with node grip varied by ±10%.

### The old world could not allow it

- Nodes were capped at 5 m/s, so no creature could exceed 300 m.
- Air kept 98.5% of velocity per 1/60 s, a drag of 0.9 × speed per second. At 16.7 m/s that is about 15 m/s², more than ground grip (friction 1.5 × g ≈ 15 m/s²) can supply.
- A momentum ledger of a fast gait with a much weaker drag (0.998) still showed drag removing about 110 of the 125 kg·m/s that ground contact added over a trial.
- Muscles (2 m/s, 5 N), bones (2 m), and rhythms (0.5 s minimum) were all slower than the gaits a kilometer needs.

### Size sets the speed limit

Relaxing one limit at a time and re-tuning a champion with CMA-ES (300 iterations of 64 samples) showed which limits bind once they are no longer tiny. Node speed, bone spin, and muscle force did not: relaxing them left a tuned gait's distance unchanged. Removing air drag gave +24%, doubling ground grip +22% on top of that, and doubling muscle speed +12%. What mattered most was scale: doubling gravity gave +40–50%, and doubling the bone limit about +33%, in line with Froude scaling (running speed ∝ √(g·L)). With Earth gravity, a kilometer in a minute needs bodies about 10 m across; small bodies would need about 7 g. The new defaults keep g = 9.8 and ground grip 1.5, and allow large bodies: bones up to 10 m, muscles up to 5 m that change length at up to 24 m/s with up to 100 N, nodes up to 60 m/s, bones turning up to 40 rad/s, 0.2 s rhythms, 120 J of muscle energy recovering at half the deficit per second, and no air drag.

Bodies start at about 0.25 m per bone, so the search must grow them 30–40×. Local mutation alone rarely did: long runs plateaued at 300–450 m with 2–4 m bones, while a standalone CMA-ES from a scaled-up champion reached about 960 m. Two changes let evolution grow bodies:

- **Whole-body rescaling** as a structural mutation: all lengths × s (0.75–1.5), rhythm × √s, stiffness ÷ √s, so the gait roughly carries over.
- **Log-scale height axis** from 15 cm up to 60% of the bone limit, so small and giant bodies occupy different archive cells instead of all giants sharing the top cell.

Together: 304 → 856 m on seed 38 and 1,229 m on seed 40 at 241 generations (4,000 creatures per generation).

### A local optimizer in physical units

The existing CMA emitter works in normalized [0, 1] coordinates with σ = 0.12. That is about 1 m for node positions and 1.2 s for the rhythm period, so it acts as a large-step explorer. It found most new niches early in a run: 116 new niches in the first generation, against 8 when it was replaced by a fine-grained optimizer, which cut the gen-40 best from 270 m to 25 m. It stays for exploration.

A second role was added: each island runs a separable CMA-ES (Ros and Hansen, 2008) on its fastest elite's body plan. It works in physical units: node positions, bone lengths, and muscle lengths scaled to the body's size; log period and log stiffness; joint ranges; organ mass and position; and touchdown reset phases. It ranks samples by distance alone. Alone, this optimizer lifts a 230 m chain to 515 m in 300 iterations of 64 samples. Inside evolution it needed three fixes before it helped:

1. **Rank all samples on the same terms.** Only archive contenders got the perturbed fine check, so the samples that looked best were exactly the ones scored by min(A, B), and the rest by A alone. Every optimizer sample is now checked.
2. **Do not jump to every new record.** Recentering on each new island best, often a lucky evaluation, kept resetting its progress. Step-length control was also misled by repair and body limits moving samples: σ collapsed to 0.09 or grew to 10. A sticky optimizer per island and body plan fixed it (seed 39: 455 → 958 m at 241 generations). So did shrinking σ whenever the median sample falls below a quarter of the best.
3. **Start at σ = 0.5** of the base steps, since evolved gaits are fragile.

### Independent islands

Four islands exchanging their best 10% every 5 generations converged on one design within about 15 generations; every island's optimizer then polished the same local peak (seed 38: 855 m at 481 generations, flat from generation 240). Migrating every 25 generations lets each island settle on its own design and optimize it. The best island wins:

| Migration interval | Seed 38 best at 241 generations |
|---|---|
| 5 generations | 847 m |
| 25 generations | 1,556 m (1,562 m verified at 240 Hz) |
| never | 1,243 m |

### Glitch guard: joints that break

With the new limits, strong muscles can force a joint of a small body through its limits and round like a wheel. In the existing joint test, a random creature spun a joint through a full turn and jammed 1.5 rad outside its range. A joint forced more than 0.5 rad past its range now breaks, which ends the trial like a fall: the CPU engine, GPU kernel, and replay all apply it. The fastest evolved runners keep their joints within 0.06 rad of their ranges, so the rule costs them nothing.

### Result

Final runs used the committed code and default physics: 4,000 creatures per generation for 481 generations (about 1.9 million evaluations), seeds 38–41. The 12 fastest creatures of each run were run again from 9 starts at 240 Hz; the median and worst columns give the best such creature (they may be different creatures).

| Seed | Recorded best | Best median of 9 starts | Best worst of 9 starts | Best at 244,000 evaluations |
|---|---|---|---|---|
| 38 | 838 m | 834 m | 774 m | 647 m |
| 39 | 1,021 m | 972 m | 860 m | 569 m |
| 40 | 1,131 m | 1,103 m | 995 m | 437 m |
| 41 | 1,176 m | 1,133 m | 1,092 m | 567 m |

In two of four runs the best runner's median exceeds 1 km, and seed 41's best covers at least 1,092 m from every start. The other two stalled on designs that top out near 840 and 970 m; seed 38 set no new record in its last 200 generations. At the budget of the original measurements (4,000 creatures × 61 generations) the best distances are 437–647 m, 7.6–11× the 57.8 m the original search reached, although the physics differs.

Open issues:

- **Reliability.** Which design an island settles on early decides most of the outcome. Spending the same budget on 12,000 creatures per generation reached 584–618 m after 30 generations (two seeds) but was not run to completion. The game's populations are 25–750× larger than these CPU runs.
- **Giants.** The fastest runners are 20–25 m long with bones at the 10 m limit. Under Earth gravity that is what a kilometer in a minute takes in this physics; smaller runners would need stronger gravity, stiffer tendon-like muscles, or better controllers.
- **Chaos.** Many fast gaits fail from some perturbed starts even though they pass the single perturbed check. Re-checking top elites with fresh perturbations would favor steadier gaits.
- **GPU.** The kernel change compiles, but these runs used the CPU engine; CPU/GPU agreement was not rerun.

## 8. Bounded elite refresh with fresh perturbations (items 83 and 67)

The idea (note B10): fitness is noisy, so an elite that got lucky on both of its
trials holds its cell forever. Re-evaluate a rotating subset of archive elites
now and then and keep the lower score.

The item 83 implementation lives in `Experiment::refresh_elites`
(`src/storage.rs`). It runs once per generation from `push_archive_stats`, so it
covers the generational loop (`archive_batch`) and the steady-state boundary
(`finish_steady_generation`) alike. Every `EVOLUTION_ELITE_REFRESH` generations
(unset or `0` disables it, which is the default) it picks at most four elites
through a deterministic rotating window over the archive sorted by creature id,
runs one standard trial each on the CPU engine (`cpu_engine::evaluate` with
`fidelity: None`, the archive-admission configuration), and lowers the stored
fitness of any creature the trial scores lower. It lowers the entry in the
global archive and in any island archive that holds the same creature id. Cells,
creatures, descriptors, protection, and the admission rules never change;
`QdArchive::lower_fitness` (`src/qd.rs`) adjusts `qd_score` and invalidates the
cached behavior scores. The batch is bounded at four creatures per cycle, so the
refresh cannot stall the worker, and the whole feature is off by default.

Item 83 alone measured byte-identical paired runs. The reason holds today: since
the archive-admission check (7f3f3a3), every stored score already folds in the
exact-pose standard CPU trial, so re-running that same deterministic trial can
never score lower. Item 67 extends the refresh to score a fresh deterministic
perturbation of the elite instead (`storage::perturb_elite`). It mirrors the
contender check in `scheduler::perturb`: node x and y move by up to 2 cm and
grip varies by +-10%, seeded from the creature id alone, so each elite always
gets the same fresh pose. The trial still runs at the standard configuration,
and the archive still keeps the lower score. The refresh can now catch an elite
whose exact-pose score is a pose accident.

Regression coverage is in `tests/search_improvements.rs`. The synthetic tests
show a lucky score 500 m above its fresh perturbed trial dropping to that trial,
a stable score surviving, the cell and descriptor staying, and the rotating
window staying bounded. One test searches the random population for a creature
whose fresh perturbed standard trial is lower than its admitted exact trial and
shows the refresh replacing the stored score with the lower one.

Measured lowering counts (section 10 lists the full runs): at
`EVOLUTION_ELITE_REFRESH=1` the refresh lowered 2,999 archive entries over 600
cycles (10 seeds times 60 generations), about five per cycle. At interval 2 it
lowered 1,373 over 300 cycles, about 4.6 per cycle. A lowering count includes
each archive that holds the creature (the global archive plus island archives),
so one creature can contribute more than one. Nearly every elite the window
reaches is pose-fragile at standard fidelity on this harness.

Conclusion: the fresh perturbation catches lucky elites, and the reported best
distance falls by about 18 percent under interval 1 because short-budget
champions are exact-pose accidents. The search metrics do not reliably improve
at either budget. At 60 generations interval 1 is slightly negative (best mean
20.86 m against 25.64 m, QD mean 2,341 against 2,617, 3/10 best wins), while
interval 2 is slightly positive (best mean 32.28 m, QD mean 2,953, 6/10 QD wins)
but one seed supplies a +53.4 m gain and a +5,859 QD gain, so that mean is not a
reliable effect. The feature stays behind `EVOLUTION_ELITE_REFRESH`, default off,
as a correctness improvement rather than a measured search gain. The harness
admits exact-pose CPU trials only, while the real game also folds in its fine
perturbed contender check, so the refresh's reach in the game may be smaller.
The game is the place to re-measure.

## 9. Near-neutral structural mutations (item 66)

The recipe the measurements in section 1 and B1 support: when a structural
mutation adds a part, start its muscles weak so the parent's gait survives while
the new part waits for mutation to tune it.

Implementation, behind `EVOLUTION_NEUTRAL_SPLITS` (unset, empty, `0`, `false`,
`off`, or `no` disables it, the default): every muscle a mutation adds starts
neutral, which sets `short = long` (zero stroke, so the motor drive is zero) and
stiffness 5.0. The flag is read once per breeding batch. The paths that add
muscles are `added_muscle` in `duplicate_mirrored_node` (`src/evolution.rs`), the
copied limb muscles in `duplicate_limb`, and the ring muscles `repair_with` adds
for every bone that lacks one, including the second half of a `split_bone`. New
joints keep their full range; only muscles are neutralized. With the flag off
the code path is unchanged, and the harness smoke run (2 generations,
population 64, 0.5 s trials, seeds 38 and 39) is byte-identical before and
after the change.

A unit test in `src/evolution.rs`
(`neutral_structural_mutations_start_their_new_muscles_passive`) runs split,
duplicated-node, and duplicated-limb mutations over 64 random bodies in both
modes. It matches new muscles to the parent by their attachment points and
asserts that every new muscle is passive with the flag on and keeps a stroke
with the flag off. A split child's first-trial distance is deliberately not
asserted: section 1 measured splits at 1 to 6 percent of the parent even with
neutral parts, so the weaker claim about the new muscles is what the evidence
supports.

Conclusion: no measurable help at either budget. Under the flag the best
distance and QD score are slightly lower at both budgets (60 generations: best
mean 21.27 m against 25.64 m with 3/10 wins, QD mean 2,343 against 2,617 with
4/10 wins), and the differences sit inside the seed spread. The flag stays
default off. This implementation is also less neutral than the section 1 probe:
only the muscles an operator or repair adds are neutralized, while a parent
muscle that gets re-anchored onto a split's new half keeps its stroke. The
likely remaining causes of broken splits are the new middle node in ground
contact and the free new joint, so a leaf-growth operator or a rigid new joint
is the next thing to try here.

## 10. Measurements for items 66 and 67

All runs are short-budget and CPU-only: population 1024, 5 s trials, fixed
seeds, `cpu_engine::evaluate` for every creature through the production
`archive_batch` and `prepare_next_batch` path. They ran on the workstation with
`nice -n 15` and half the machine (`EVOLUTION_DEVICES=primary
EVOLUTION_CPU_THREADS=6`). Every variant had the same evaluation budget; only
the environment flag differed. The seed spread is large, so the paired per-seed
differences and the win counts matter more than the means.

    nice -n 15 env CARGO_BUILD_JOBS=6 EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 \
        cargo run --release --example search_ab -- --tag baseline 60 1024 5.0 38,39,40,41,42,43,44,45,46,47
    nice -n 15 env CARGO_BUILD_JOBS=6 EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 \
        EVOLUTION_NEUTRAL_SPLITS=1 cargo run --release --example search_ab -- --tag neutral 60 1024 5.0 38,39,40,41,42,43,44,45,46,47
    nice -n 15 env CARGO_BUILD_JOBS=6 EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 \
        EVOLUTION_ELITE_REFRESH=1 cargo run --release --example search_ab -- --tag refresh-1 60 1024 5.0 38,39,40,41,42,43,44,45,46,47
    nice -n 15 env CARGO_BUILD_JOBS=6 EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 \
        EVOLUTION_ELITE_REFRESH=2 cargo run --release --example search_ab -- --tag refresh-2 60 1024 5.0 38,39,40,41,42,43,44,45,46,47

### 25 generations, population 1024, 5 s trials, seeds 38 to 47

Per-seed archive best after the last generation (m):

| Variant | 38 | 39 | 40 | 41 | 42 | 43 | 44 | 45 | 46 | 47 | best mean | best median | QD mean | QD median | cells mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 16.71 | 7.68 | 13.56 | 9.17 | 28.77 | 10.13 | 12.88 | 17.39 | 6.74 | 14.87 | 13.79 | 13.22 | 930 | 968 | 751 |
| neutral-splits | 8.96 | 19.27 | 15.68 | 7.45 | 25.63 | 10.81 | 9.52 | 10.11 | 7.64 | 7.89 | 12.30 | 9.81 | 1016 | 1010 | 742 |
| refresh-1 | 8.95 | 7.33 | 16.97 | 7.36 | 12.15 | 18.08 | 7.78 | 12.50 | 10.32 | 11.44 | 11.29 | 10.88 | 923 | 921 | 756 |
| refresh-2 | 10.76 | 10.95 | 9.77 | 10.09 | 16.57 | 11.82 | 18.22 | 10.16 | 7.80 | 6.44 | 11.26 | 10.46 | 945 | 892 | 745 |

Paired differences against the baseline (variant minus baseline):

| Variant | 38 | 39 | 40 | 41 | 42 | 43 | 44 | 45 | 46 | 47 | best mean diff | QD mean diff | cells mean diff | best wins | QD wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| neutral-splits | -7.75 | +11.59 | +2.12 | -1.72 | -3.14 | +0.68 | -3.36 | -7.28 | +0.90 | -6.98 | -1.49 | +86 | -9.3 | 4/10 | 5/10 |
| refresh-1 | -7.76 | -0.35 | +3.41 | -1.81 | -16.62 | +7.95 | -5.10 | -4.89 | +3.58 | -3.43 | -2.50 | -7 | +5.2 | 3/10 | 4/10 |
| refresh-2 | -5.95 | +3.27 | -3.79 | +0.92 | -12.20 | +1.69 | +5.34 | -7.23 | +1.06 | -8.43 | -2.53 | +15 | -6.2 | 5/10 | 6/10 |

Top-50 body-size mix summed over the 10 seeds (node count x bodies):

- baseline: 5x70 6x136 7x97 8x110 9x39 10x26 11x20 12x1 13x1
- neutral-splits: 4x10 5x125 6x141 7x116 8x86 9x19 10x2 11x1
- refresh-1: 5x36 6x100 7x182 8x129 9x46 10x7
- refresh-2: 4x2 5x33 6x190 7x120 8x84 9x66 10x4 11x1

### 60 generations, population 1024, 5 s trials, seeds 38 to 47

Per-seed archive best after the last generation (m):

| Variant | 38 | 39 | 40 | 41 | 42 | 43 | 44 | 45 | 46 | 47 | best mean | best median | QD mean | QD median | cells mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 33.82 | 10.31 | 13.63 | 28.83 | 48.71 | 23.06 | 21.81 | 26.84 | 16.79 | 32.61 | 25.64 | 24.95 | 2617 | 2455 | 969 |
| neutral-splits | 27.58 | 25.37 | 18.90 | 13.65 | 34.57 | 28.22 | 12.90 | 21.16 | 16.77 | 13.57 | 21.27 | 20.03 | 2343 | 2152 | 968 |
| refresh-1 | 10.35 | 19.23 | 22.27 | 13.23 | 37.52 | 20.41 | 16.91 | 22.22 | 23.32 | 23.14 | 20.86 | 21.31 | 2341 | 2291 | 966 |
| refresh-2 | 38.07 | 26.97 | 18.59 | 82.20 | 32.76 | 17.67 | 21.78 | 17.92 | 50.08 | 16.81 | 32.28 | 24.38 | 2953 | 2367 | 960 |

Paired differences against the baseline (variant minus baseline):

| Variant | 38 | 39 | 40 | 41 | 42 | 43 | 44 | 45 | 46 | 47 | best mean diff | QD mean diff | cells mean diff | best wins | QD wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| neutral-splits | -6.24 | +15.06 | +5.27 | -15.18 | -14.14 | +5.16 | -8.91 | -5.68 | -0.02 | -19.04 | -4.37 | -274 | -1.4 | 3/10 | 4/10 |
| refresh-1 | -23.47 | +8.92 | +8.64 | -15.60 | -11.19 | -2.65 | -4.90 | -4.62 | +6.53 | -9.47 | -4.78 | -276 | -2.6 | 3/10 | 4/10 |
| refresh-2 | +4.25 | +16.66 | +4.96 | +53.37 | -15.95 | -5.39 | -0.03 | -8.92 | +33.29 | -15.80 | +6.64 | +336 | -9.5 | 5/10 | 6/10 |

Top-50 body-size mix summed over the 10 seeds (node count x bodies):

- baseline: 5x20 6x172 7x21 8x144 9x39 10x53 11x1 13x30 14x16 15x1 16x3
- neutral-splits: 5x75 6x148 7x112 8x81 9x11 10x19 11x5 12x10 13x14 14x24 15x1
- refresh-1: 5x6 6x83 7x135 8x94 9x105 10x60 11x11 13x1 14x1 15x4
- refresh-2: 5x2 6x69 7x150 8x153 9x50 10x68 11x6 13x1 14x1

How to read these: a "win" is one seed where the variant beat the baseline on
that metric. The refresh variants lower stored elite scores, so their reported
best can fall without any gait getting slower; that is the point of the
correction. The neutral-splits flag changes which offspring get evaluated, so
its comparison is a genuine search A/B. Neither variant shows a gain that
survives the seed spread.

## Sources

- Arza, Le Goff, Hart (2024). [Generalized Early Stopping in Evolutionary Direct Policy Search](https://arxiv.org/abs/2308.03574). ACM TELO.
- Auerbach, Bongard (2014). [Environmental Influence on the Evolution of Morphological Complexity in Machines](https://journals.plos.org/ploscompbiol/article?id=10.1371%2Fjournal.pcbi.1003399). PLOS Computational Biology.
- Bahlous-Boldi, Faldor, Grillotti, Janmohamed, Coiffard, Spector, Cully (2025). [Dominated Novelty Search: Rethinking Local Competition in Quality-Diversity](https://arxiv.org/abs/2502.00593). GECCO.
- Bongard (2011). [Morphological change in machines accelerates the evolution of robust behavior](https://www.pnas.org/doi/10.1073/pnas.1015390108). PNAS.
- Cheney, Bongard, SunSpiral, Lipson (2018). [Scalable co-optimization of morphology and control in embodied machines](https://royalsocietypublishing.org/doi/10.1098/rsif.2017.0937). J. R. Soc. Interface.
- Cheney, MacCurdy, Clune, Lipson (2013). [Unshackling Evolution: Evolving Soft Robots with Multiple Materials and a Powerful Generative Encoding](https://jeffclune.com/publications/2013_Softbots_GECCO.pdf). GECCO.
- Cully, Demiris (2018). [Quality and Diversity Optimization: A Unifying Modular Framework](https://arxiv.org/abs/1708.09251). IEEE TEVC.
- de Bruin, Glette, Ellefsen, Nadizar, Medvet (2026). [Social Learning Strategies for Evolved Virtual Soft Robots](https://arxiv.org/abs/2604.12482).
- Flageat, Cully (2020). [Fast and stable MAP-Elites in noisy domains using deep grids](https://arxiv.org/abs/2006.14253). ALIFE.
- Fontaine, Nikolaidis (2023). [Covariance Matrix Adaptation MAP-Annealing](https://arxiv.org/abs/2205.10752). GECCO.
- Heidrich-Meisner, Igel (2009). [Hoeffding and Bernstein races for selecting policies in evolutionary direct policy search](https://dl.acm.org/doi/10.1145/1553374.1553426). ICML.
- Hornby (2006). [ALPS: the age-layered population structure for reducing the problem of premature convergence](https://dl.acm.org/doi/10.1145/1143997.1144142). GECCO.
- Hornby, Pollack (2002). [Creating High-Level Components with a Generative Representation for Body-Brain Evolution](https://direct.mit.edu/artl/article-abstract/8/3/223/2398/Creating-High-Level-Components-with-a-Generative). Artificial Life.
- Hutchinson, Herrmann, Smith (2026). [Discrete Gene Crossover Accelerates Solution Discovery in Quality-Diversity Algorithms](https://arxiv.org/abs/2602.13730).
- Ijspeert (2008). [Central pattern generators for locomotion control in animals and robots: A review](https://doi.org/10.1016/j.neunet.2008.03.014). Neural Networks.
- Jelisavcic, Glette, Haasdijk, Eiben (2019). [Lamarckian Evolution of Simulated Modular Robots](https://pmc.ncbi.nlm.nih.gov/articles/PMC7805734/). Frontiers in Robotics and AI.
- Lehman, Miikkulainen (2015). [Extinction Events Can Accelerate Evolution](https://journals.plos.org/plosone/article?id=10.1371%2Fjournal.pone.0132886). PLOS ONE.
- Lehman, Stanley (2011). [Evolving a diversity of virtual creatures through novelty search and local competition](https://dl.acm.org/doi/10.1145/2001576.2001606). GECCO.
- Li, Jamieson, DeSalvo, Rostamizadeh, Talwalkar (2018). [Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization](https://arxiv.org/abs/1603.06560). JMLR.
- Luo, Stuurman, Tomczak, Ellers, Eiben (2022). [The Effects of Learning in Morphologically Evolving Robot Systems](https://arxiv.org/abs/2111.09851). Frontiers in Robotics and AI.
- Luo, Miras, Tomczak, Eiben (2023). [Enhancing robot evolution through Lamarckian principles](https://www.nature.com/articles/s41598-023-48338-4). Scientific Reports.
- Mertan, Cheney (2023). [Modular Controllers Facilitate the Co-Optimization of Morphology and Control in Soft Robots](https://arxiv.org/abs/2306.09358). GECCO.
- Mertan, Cheney (2024). [Investigating Premature Convergence in Co-optimization of Morphology and Control in Evolved Virtual Soft Robots](https://arxiv.org/abs/2402.09231). EuroGP.
- Mertan, Cheney (2025). [Controller Distillation Reduces Fragile Brain-Body Co-Adaptation and Enables Migrations in MAP-Elites](https://arxiv.org/abs/2504.06523). GECCO.
- Mertan, Cheney (2025). [Evolutionary Brain-Body Co-Optimization Consistently Fails to Select for Morphological Potential](https://arxiv.org/abs/2508.17464). Artificial Life (accepted).
- Mouret, Clune (2015). [Illuminating search spaces by mapping elites](https://arxiv.org/abs/1504.04909).
- Nordmoen, Veenstra, Ellefsen, Glette (2021). [MAP-Elites Enables Powerful Stepping Stones and Diversity for Modular Robotics](https://arxiv.org/abs/2012.04375). Frontiers in Robotics and AI.
- Ros, Hansen (2008). [A Simple Modification in CMA-ES Achieving Linear Time and Space Complexity](https://doi.org/10.1007/978-3-540-87700-4_30). PPSN X.
- Sims (1994). [Evolving Virtual Creatures](https://www.karlsims.com/papers/siggraph94.pdf). SIGGRAPH.
- Song, Yang, Xu, Wen, Peng, Li, Zhou, Yao (2026). [Shaping the Evolutionary Dynamics of Robot Morphology via Adaptive Control Learning](https://arxiv.org/abs/2608.23100).
- Stanley, Miikkulainen (2002). [Evolving Neural Networks through Augmenting Topologies](https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf). Evolutionary Computation.
- Templier, Grillotti, Rachelson, Wilson, Cully (2024). [Quality with Just Enough Diversity in Evolutionary Policy Search](https://arxiv.org/abs/2405.04308).
- Vassiliades, Mouret (2018). [Discovering the Elite Hypervolume by Leveraging Interspecies Correlation](https://arxiv.org/abs/1804.03906). GECCO.
- Veenstra, Glette (2020). [How Different Encodings Affect Performance and Diversification when Evolving the Morphology and Control of 2D Virtual Creatures](https://www.mn.uio.no/ifi/english/people/aca/kyrrehg/publications/veenstra-alife2020.pdf). ALIFE.
- Veenstra, Olsen, Glette (2022). [Effects of encodings and quality-diversity on evolving 2D virtual creatures](https://doi.org/10.1145/3520304.3529053). GECCO Companion.
- Veenstra, Szorkovszky, Glette (2023). [Decentralized Control and Morphological Evolution of 2D Virtual Creatures](https://direct.mit.edu/isal/proceedings/isal2023/35/108/116873). ALIFE.
- Wang, Lehman, Clune, Stanley (2019). [Paired Open-Ended Trailblazer (POET)](https://arxiv.org/abs/1901.01753).
- Wang, Zhou, Fidler, Ba (2019). [Neural Graph Evolution: Towards Efficient Automatic Robot Design](http://www.cs.toronto.edu/~henryzhou/NGE/nge.pdf). ICLR.
- Further reading: Wang et al. (2025). [Embodied Co-Design for Rapidly Evolving Agents: Taxonomy, Frontiers, and Challenges](https://arxiv.org/abs/2512.04770), a survey of more than 100 recent co-design studies.
