# UX audit and overhaul proposal

Date: 2026-09-28. Code at e161f51. The UI code is unchanged since b3f30d1.

This is a user experience audit, not a visual design review. It asks what the player comes to do, how many steps each task takes today, and where the screen hides, repeats or mislabels what the player needs. The owner's brief applies throughout: "you're making a game, not a knobfest". Fewer settings, sensible defaults, and nothing a player must configure are part of the result. The product rules in AGENTS.md also hold: distance is the only score, the game runs 3M creatures with 60 s trials, there are no mutation controls, and environment effects are buttons.

Method. I read `src/ui.rs` (4,626 lines), `src/worker.rs`, `src/environment.rs`, the storage paths the UI calls, README.md and AGENTS.md. I ran the release-fast game in the smoke harness once per tab, in a 1440 by 900 window, and let it capture after 8 s. The "fresh" screenshots start a new game with 4,000 creatures and 2 s trials and evolve for 8 s. The "loaded" screenshots open a save made by a 40-generation headless run (4,000 creatures, 60 s trials, rough ground and low hurdles). The harness cannot click, so flows that need clicks are traced in code. Line numbers refer to `src/ui.rs` unless another file is named.

Screenshots, all in [ux-audit/](ux-audit/):

| File | What it shows |
| --- | --- |
| [fresh-overview.png](ux-audit/fresh-overview.png) | Overview at generation 26 of a new game |
| [fresh-archive-cards.png](ux-audit/fresh-archive-cards.png) | Behavior archive, cards |
| [fresh-archive-map.png](ux-audit/fresh-archive-map.png) | Behavior archive, map, while evolving |
| [fresh-history.png](ux-audit/fresh-history.png) | History and statistics |
| [fresh-race.png](ux-audit/fresh-race.png) | Race at generation 36 |
| [fresh-lineage.png](ux-audit/fresh-lineage.png) | Lineage at generation 33 |
| [loaded-overview.png](ux-audit/loaded-overview.png) | Overview after opening a save |
| [loaded-archive-map.png](ux-audit/loaded-archive-map.png) | Archive map after opening a save, paused |

## 1. Who plays and what they come to do

The player is the owner and people like them. They enjoy watching a simulation discover movement, they poke the world to see what evolution does next, and they want to show a strange walker to someone else. They are not tuning a search algorithm. The search internals (MAP-Elites, emitters, islands, contender checks) exist so that the player gets good and varied creatures. The player does not need to operate them.

A session is long and mostly spent waiting. A 3M generation takes about 15 s at 200,000 creatures/s, and more than a minute late in a long session: AGENTS.md records 38,615 creatures/s at generation 70. The player looks at the screen far more often than they click. So the main screen has to reward watching, and every change the player did not cause (a new record, a season step, a re-test) has to explain itself.

The recurring tasks, in the order a session meets them:

1. Start or resume a run. Once per session.
2. Watch evolution happen. Is it running, how far along is this generation, is it getting better? All the time.
3. Find and watch the best or the most interesting creatures. Many times per session.
4. Change the world and see the effect. Several times per session.
5. Compare creatures: the champion against a challenger, before and after a world change, a creature against its ancestors.
6. Save and share: save the run, export a GIF or JSON of a creature, take a screenshot.

## 2. Current flows and friction

### 2.1 Start or resume a run

Current flow for a new run:

1. Launch. `App::new` (line 1156) sends `Command::New` with the default config, so the worker builds a 3M population before the player has chosen anything.
2. The viewport says "Preparing your first population…".
3. Click "Evolve continuously" in the left panel.

Current flow to resume:

1. Launch. A 3M population is built anyway.
2. Click Open in the top bar. A window opens with a text field that holds `runs/experiment.evo` (`file`, line 1284).
3. Type the path of the save. Autosaves are called `runs/seed-<seed>-auto.evo`, which only README mentions.
4. Click "Open experiment". The worker loads and stays paused (`worker.rs` line 432).
5. Notice "Loaded /home/…/world-4k.evo" in the status bar ([loaded-overview.png](ux-audit/loaded-overview.png)) and click "Evolve continuously".

Friction:

- Launch always builds a new 3M population, even when the player wants to resume. It costs seconds and gigabytes before the first choice.
- There is no list of saves. Open is a free-text path, and the player must remember file names.
- Nothing guards progress. Save overwrites without asking. Open and window close do not warn about an unsaved run (`on_exit`, line 3462, only prints benchmark frame times). The New dialog offers "Save current", Open does not.
- The New dialog says "Create N creatures using the settings in the sidebar" (`dialogs`, line 3297). The sidebar's world effects carry into the new run silently, and the seed hides in Advanced > Randomness.
- Three run buttons compete: "Evolve continuously", "One generation" and "Guided step" (lines 1373 to 1409). At 3M one generation takes 15 to 80 s. Guided step pauses between evaluation, archive insertion and breeding, which is a debugging tool. Both are greyed out while evolution runs ([fresh-overview.png](ux-audit/fresh-overview.png)), so most of the time they are dead space.
- Space does not do what README says. README: "Space pauses or resumes evolution". The key handler (line 3595) toggles the replay whenever a replay exists, and a replay exists from the first frame because New and Load always send a preview (`worker.rs` lines 336 and 428). In practice Space never reaches evolution. The help overlay describes the real behavior, README does not.

### 2.2 Watch evolution happen

Current flow: the Overview tab shows four tiles, the replay viewport, a lineage strip and a trend chart. The left panel shows a stage label and a progress bar. The top bar shows "GEN 026", the status bar "Evolving · generation 26" and creatures/s ([fresh-overview.png](ux-audit/fresh-overview.png)).

Friction:

- The main viewport does not show evolution. It replays creature 0 of the first population and never changes on its own. The worker sets a preview only on New, Load and an explicit Preview request (`worker.rs` lines 336, 428, 454). In [fresh-overview.png](ux-audit/fresh-overview.png) the best distance is 4.21 m at generation 26, and the viewport shows "#1 · 3 nodes / 2 bones / 1 muscles · replay distance -0.1 m": a random first-generation body that moves backwards. The label says "LIVE CREATURE" (line 1726) although it is a recorded replay.
- Two of the four headline tiles are research or machine numbers: "QD SCORE" and "EVALUATIONS / SEC" (`metrics`, line 2052). Creatures per second shows three times: the tile, the status bar and the Performance line. After a load the tile says 2,775 while the status bar says 0 creatures/s, because nothing runs ([loaded-overview.png](ux-audit/loaded-overview.png)). "NICHES" is jargon.
- The score has several names. The tile says "BEST GAIT SCORE", the chart axis "Gait score (m)", cards and lineage just "m", README and the owner say distance. A player cannot tell that these are one number.
- Progress reads as a contradiction. The bar can read "4,000 / 4,000 evaluated · 2,083 in checks" (line 1415) while the stage label says "Archive updated" and the status line says "Evolving". In a steady run generations overlap, so the stage label means nothing to a player. No time per generation or time left is shown anywhere.
- The trend chart gets about 100 px when there is no lineage, and its legend covers the newest generations ([fresh-overview.png](ux-audit/fresh-overview.png)). When a lineage exists, the strip takes that space and the chart is pushed off the screen entirely ([loaded-overview.png](ux-audit/loaded-overview.png)). The Overview has no scroll area, so the chart cannot be reached.
- New records are not announced. They appear only on the History tab, in two lists.
- The top of the left panel is a slogan ("Let life find a way. More kinds of life. Better walkers.", line 1369) above the controls.

### 2.3 Find and watch the best or most interesting creatures

Current paths to the champion. None is on the Overview:

1. Behavior archive > Cards > card #1.
2. Behavior archive > Map > pick the height and feet slice that holds it > hottest cell.
3. History > records timeline > newest button.
4. History > hall of fame > top Replay.
5. History > Best creature thumbnail.
6. Race > lane #1, if the race was started recently (see 2.5).

Friction:

- Every path except Race jumps to the Overview tab after the click (lines 2456, 2564, 2660, 2820, 3228). Browsing the archive means tab 2, scroll, click, watch, tab 2, find the place again.
- The archive tab speaks research ([fresh-archive-cards.png](ux-audit/fresh-archive-cards.png)). The header says "Search archive" while the tab says "Behavior archive". Below it: "991 niches · 64 topology reserves · QD score 694.70", "Archive axes:", and "Next batch: Diagonal CMA-ES 35% · Morphology 35% · Novelty 30% · Immigrant 0%" (`population`, lines 2254 to 2362). The card tooltip lists "Mutability" (a gene the search does not use, AGENTS item 70), "form", "bob", "visits" and "Emitter: Diagonal CMA-ES" (line 2419). The Lineage tab calls the same CMA step "fine-tuned" (`storage.rs` `describe_change`, line 202).
- Cards do not show what makes a creature interesting. They are static poses sorted by distance, so the first rows are near copies of one body plan: cards #1, #2, #3, #6, #7, #8 and #11 look alike. The archive holds 991 kinds of movement, and the first row of the card view shows mostly one.
- Species names do not group species. Those seven look-alike cards carry six different names (Rilform, Lumform, Nymform, Oviform, Sableform and Wispform Walker). `species_name` (line 642) hashes exact bone lengths and muscle periods, so any small mutation renames the creature. The player cannot use names to recognize a kind.
- Cards cannot be filtered by feet, body size or gait, or grouped. The only search box searches settings.
- The map is empty while evolution runs. After 8 s of a fresh game it still showed "Mapping archive pages… 0 / 1025" ([fresh-archive-map.png](ux-audit/fresh-archive-map.png)). The UI fills the map by requesting the archive 120 cards at a time, one page per worker publish (at most one every 200 ms), and starts over whenever the archive size changes or a new generation begins (`absorb_archive_page`, line 2473, and `population`, line 2369). At 4,000 creatures it had not placed a single cell after 8 s. I did not measure it at 3M, where generations are slower but the archive changes with every absorbed unit.
- When the map does fill (paused, [loaded-archive-map.png](ux-audit/loaded-archive-map.png)), it opens on the slice of the smallest bodies with one foot: 18 cells from -0.03 to 1.31 m while the champion reaches 8.03 m. The color scale is recomputed per slice, so 1.3 m is drawn in the hottest red. The other 29 slices sit behind two dropdowns.
- The end-of-trial text is wrong for two of three endings. The viewport always prints "Fell over at X s: head below its neck" (line 1945), but the same marker covers a broken joint and the 8 g head-shake limit (`Playback::fall`).
- One creature shows two distances. The viewport shows the replay distance, and the archive keeps the worse of the standard trial and the fine check. The explanation is a tooltip on the viewport header. The Race makes it visible: lane #4, archive score 0.25 m, leads the standings at 0.71 m ([fresh-race.png](ux-audit/fresh-race.png)).
- Replays run in different worlds depending on where they start. The hall of fame (line 2658) and the lineage (lines 2753 and 2818) replay in the current world. The records timeline (line 2559) and the History thumbnails (line 3227) replay in the world of their generation. After a world change a record holder replayed from the hall of fame falls short of its record, and nothing on screen says why. The viewport header never names the world.
- Clicking an ancestor in the lineage replays it, but the "selected" mark stays on the descendant (`paint_lineage_tile`, line 968: `current` is always the first tile).
- There are two Pause buttons with different meanings: evolution in the left panel, the replay under the viewport. "Single tick" (line 2034) is a debugging control.

### 2.4 Change the world and see the effect

Current flow: the left panel lists 13 effects as "Name: level" with a raise and a lower button each, then a Catastrophe row with Meteor strike, Extinction and Undo (`control_contents`, lines 1454 to 1550). The reason for each effect is in a tooltip. A click applies at once and re-tests the archive.

Friction:

- 26 effect buttons carry 25 different labels: Roughen and Smooth, Strengthen and Weaken, Thicken and Thin, Make slippery and More grip, Heat up and Cool down, Dry out and Water, and so on. "Calm" is the lower button of both Wind and Earthquake. The player reads each pair to learn which way is harder.
- At 1440 by 900 only 9 of the 13 effects fit ([fresh-overview.png](ux-audit/fresh-overview.png)). Mud, Gaps, Hurdles, Earthquake, the catastrophes and Advanced are below the fold of a scrolling panel. The loaded save runs with low hurdles, and nothing on screen says so without scrolling ([loaded-overview.png](ux-audit/loaded-overview.png)).
- One level per click. Flat to Boulders takes four clicks, and each click is a world change: the archive empties and every elite is queued for a re-test (`storage.rs` `reset_search_context`, line 2012).
- The panel shows the new level at once, but a change during a generation is only queued (`storage.rs` line 1921 stores it as `pending`). The status line says "Settings applied or queued for the next generation" (`worker.rs` line 377). The player cannot tell whether the world has changed yet.
- A world change empties the archive. The Behavior archive shows few cards until re-tests land, the Race cannot build, and the best distance drops. None of this is explained on screen, and the chart has no marker for the change (AGENTS item 43). The data exists: every history row carries its config.
- "Every change can be undone" (line 1448) is half true. Lowering the level restores the rules, but the old archive does not come back. The elites are re-tested again.
- Nothing summarizes how the world differs from calm, and nothing returns it to calm in one click. "Reset settings" at the bottom resets every config field and then waits for "Apply settings".
- Seasons change the world every 20, 10 or 5 generations. The panel updates silently. Nothing says which effect changed or when the next step comes.
- The viewport draws the ground from the replay's own config. That is correct for the replay, but after a change the player keeps seeing the old ground until they pick another creature, so the new world has no preview.
- Catastrophe explanations live in tooltips. The number of fossils Undo would return is only in its tooltip.

### 2.5 Compare

- The Race is built once and never refreshed. `maybe_build_race` (line 2664) takes the top five from the first archive page it sees. In [fresh-race.png](ux-audit/fresh-race.png), at generation 36, the lanes still show "best 0.34 m", "0.33 m", "0.27 m": the generation-0 record was 0.34 m ([fresh-history.png](ux-audit/fresh-history.png), same seed). The header still says "The fastest archived creatures run their trials side by side". The player must know to press "New race".
- The Race runs the archive's top five and nothing else. The player cannot add the creature they are watching, a record holder or an ancestor.
- There is no before and after view for a world change.
- The Lineage tab is empty by default. At generation 33 it says "No recorded ancestors for this creature yet" ([fresh-lineage.png](ux-audit/fresh-lineage.png)), because the selected creature is still the random first one (see 2.2). Once filled, it is a list: comparing an ancestor with its descendant means clicking each and watching them one at a time on another tab.
- The History tab shows the best creature per generation three times: the records timeline, the hall of fame directly below it with the same six entries ([fresh-history.png](ux-audit/fresh-history.png)), and the Best creature thumbnail. The "Worst" curve is the worst archive elite, near -1 m, which is not a creature the player cares about. The body-type strip has no legend. The heading says "Generation archive" under a tab called "History & statistics". "Percentile curves" offers 29 checkboxes (P0 to P100).

### 2.6 Save and share

- Save opens the path dialog. The worker saves inside its command loop (`worker.rs` line 407), so evolution stops while it writes, and at 3M a checkpoint is 1.1 to 1.4 GB. There is no progress indicator. The dialog closes at once, and "Saved …" appears in the status bar later.
- Creature export (Export JSON, GIF, Open creature) sits in the left panel's bottom row next to Save preset, Load preset and Reset settings (lines 1684 to 1720), far from the replay it exports.
- Screenshot is in the top bar and Export CSV in the History header. File actions live in four places.
- The status bar has two message channels: the worker status on the left and a local message line below it. The local message is set in about ten places and never cleared.
- Exported files are named `creature-<id>-<millis>.gif`. The species name and distance would make them recognizable.

### 2.7 Settings a player can touch today

"Advanced controls" opens a search box and four sections: Randomness (new seed on creation, seed), Performance and checkpoints (Maximum throughput, autosave interval), Display (UI scale, sort animation speed, dark theme, show help), Debug (histogram minimum, maximum and bins per meter, performance details, GPU budget MiB, RAM budget MiB). Changes set a dirty flag that needs "Apply settings". The bottom row adds Save preset, Load preset and Reset settings. Elsewhere: 29 percentile checkboxes, two map dropdowns, a playback speed slider and a follow checkbox.

That is about 20 settings and 29 checkboxes. Most have one correct automatic value. Maximum throughput is already chosen from the population size, the budgets can come from the machine, the histogram can fit its data, and the sort animation speed has no reason to change. This is the knobfest the owner wants gone.

## 3. Information architecture proposal

Principles:

- The main screen is a theater. It shows the current champion walking, always.
- One headline number: best distance. Everything else is one click away or gone.
- The world panel shows state. The player clicks the level they want.
- Every change the player did not cause explains itself in an event feed, with the next action attached (replay, undo).
- No setting a player must configure. Machine limits are automatic. Diagnostics live in one drawer that is closed by default.

### Main screen

- Top bar: one Evolve and Pause button with its state, "Generation 42 · 63% · about 9 s left", "Best 12.4 m", the save state ("Saved 4 min ago" or "Not saved"), a File menu (New, Open, Save, Save as, Export) and a Help button.
- Left: the world panel. Each effect is one row of level buttons with the current level lit and calm marked. Rows away from calm stand out and sort to the top. A "Calm world" button heads the panel. Seasons show the next step. Catastrophes sit in their own group with Undo and its fossil count.
- Center: the theater. Its header says what is playing: "Champion" (the default, follows every new record) or "Watching: Vexpod Walker, rank 17" with a "Back to champion" button. The header also names the world the replay runs in. Under it: the time scrubber, play and speed, and the creature's actions: Race it, Family tree, Export GIF, Export JSON.
- Right or bottom: the event feed and a best-distance chart with a marker for every record and every world change. The chart keeps a fixed height.

### Tabs, organized around the tasks

1. Watch. Today's Overview, as above.
2. Explore. Today's Behavior archive. Cards and map, with filters for feet, body size and gait, and cards grouped by body plan. A click plays the creature in a player docked on the same tab, so browsing never leaves the tab.
3. Race. Lanes the player picks: the champion by default, plus anything sent with "Race it". The current top five stay as a one-click default and refresh when the player opens the tab.
4. History. One chart with world-change bands and record markers, one records list with replay buttons (the hall of fame and the records timeline merged), the body-type mix with a legend, and the distance histogram with an automatic range.

Lineage becomes a panel opened from any creature's "Family tree" action instead of a top-level tab.

### What moves or goes

- To the Diagnostics drawer: QD score, niche and reserve counts, emitter shares, creatures/s per engine, GPU and memory figures, stage label, "in checks", Guided step, One generation, Single tick.
- Automatic, no control: Maximum throughput, GPU and RAM budgets, histogram range and bins, sort animation speed, the percentile choice (best and median stay), the Apply flow.
- Kept as a choice, but only where it is needed: the seed (in the New dialog, prefilled at random), autosave on or off (in the File menu), dark theme and UI scale (follow the system, with a menu item).
- Removed: Save preset and Load preset (the world is set with buttons and the rest is fixed), Reset settings, the Advanced search box, the slogan, "Mutability" and "visits" in tooltips.

## 4. Ranked changes

### Quick wins (small, highest value first)

1. Theater follows the champion. After each generation, or when a record lands, the Overview replay switches to the current best elite unless the player has pinned a creature. The header says "Champion" or "Watching …" with "Back to champion". Fixes: the main view shows a random first-generation creature for the whole session, and the Lineage tab stays empty. Size: small. The worker already picks the best elite on Load (`worker.rs` line 423).
2. Correct the end-of-trial text. Say "fell over", "broke a joint" or "shook its head too hard", from the replay result. Fixes: two of three endings are mislabeled. Size: small.
3. One name for the score. Use "distance" everywhere ("Best distance", axis "Distance (m)"). Drop the QD score and creatures/s tiles from the main screen. Fixes: four names for one number, and research metrics as headlines. Size: small.
4. Effects as level buttons. Show each effect as a row of its levels, the current one lit and calm marked, one click per target level, plus "Calm world". Rows away from calm sort to the top so the world is readable without scrolling. Fixes: 25 labels, up to four clicks and four re-tests per change, and hidden active effects. Size: small to medium. Effects stay buttons.
5. Show pending world changes. Mark a changed effect "from next generation" until the worker applies it. Fixes: the panel claims a world the creatures are not in yet. Size: small, because the worker knows `pending`.
6. Replay in the right world, and say which. Every replay path uses the world the creature was scored in, and the theater header names that world. Fixes: hall of fame and lineage replays that miss their record for no visible reason. Size: small.
7. Clear the knobs. Remove Guided step, One generation, Single tick, sort animation speed, histogram controls, budgets, Maximum throughput, presets, Reset settings and the Apply flow from the player's view. Keep a Diagnostics drawer for the developer. Fixes: about 20 settings and 29 checkboxes in a game that needs almost none. Size: small.
8. Fresh race. Rebuild the top-five race whenever the player opens the Race tab, and rank the standings by the same distance the lanes show as their score. Fixes: a race of generation-0 creatures at generation 36, and a rank #4 that wins. Size: small.
9. Keep the chart on screen. Give the Overview chart a fixed height and fold the lineage strip into the Family tree panel. Fixes: the chart disappears whenever a lineage exists. Size: small.
10. Creature actions under the theater. Move Export GIF, Export JSON and Open creature next to the replay, and name files after the species and distance. Fixes: export lives in the settings area, and files are unrecognizable. Size: small.
11. One Space meaning, shown on the buttons. Space pauses and resumes evolution as README says, and K or a click on the viewport pauses the replay. Show each shortcut on its button. Fixes: Space never reaches evolution. Size: small.
12. Seasons say what comes next. "Next season step at generation 60: Wind to Breeze." Fixes: silent world changes. Size: small.
13. Messages expire. Merge the local message line into the status line and clear it after a few seconds. Fixes: two channels and stale messages. Size: small.

### Structural changes (highest value first)

1. Event feed. One list of what happened: records ("Gen 41: new record 12.3 m, Vexpod Walker [Replay]"), world changes ("Ground to Rough 8 cm: re-testing 1,240 elites… done, best 10.1 m to 7.4 m"), season steps, catastrophes with Undo, saves and errors. Fixes: every jump in the numbers currently goes unexplained. Size: medium.
2. Chart markers. Records and world changes on the best-distance chart, taken from the per-generation config in history. Fixes: AGENTS item 43 and the unexplained drops. Size: small to medium.
3. Resume flow. The Open dialog lists saves in `runs/` with generation, best distance, date and size, autosaves included, newest first. Launch offers "Resume latest", "Open" and "New" before any population is built. Fixes: typed paths, and a wasted 3M population on every launch. Size: medium.
4. Progress you can trust. "Saved 4 min ago" or "Not saved" in the top bar, a confirmation before New, Open, overwrite and close with unsaved generations, and a visible "Saving 1.2 GB…" state. Fixes: silent overwrites, lost runs and a save with no feedback. Size: medium.
5. Explore without tab jumps. A docked player on the Explore tab, filters for feet, body size and gait, and grouping by body plan. Fixes: the tab-2, click, tab-1 loop and a card grid whose first rows repeat one body plan. Size: medium to large.
6. A map that works while evolving. Keep the last complete map and update it in place instead of clearing it every generation, or have the worker send the compact cell table (niche, distance, rank) instead of 120 full creatures per page. Open on the best cell of every contact and cadence pair across all heights and feet, with height and feet as optional filters and one fixed color scale. Fixes: an empty map during play and a misleading default slice. Size: medium.
7. Species names that mean a kind. Name by body plan (node, bone and muscle counts and the bone tree), not by exact lengths and periods, so small mutations keep the name. Fixes: look-alike creatures with different names. Size: small to medium; it changes the names in existing saves.
8. Race from anywhere. "Race it" on any creature adds it to the race next to the champion. Fixes: comparison is limited to the top five. Size: medium.
9. History consolidation. One records list (hall of fame and timeline merged), world bands on the chart, best and median curves only, a legend for body types, and the histogram with an automatic range. Fixes: three copies of the best creature per generation and 29 checkboxes. Size: medium.
10. Stall hint. When the best has not moved for many generations, the feed suggests an environment effect (AGENTS item 102). Fixes: the player does not know when to intervene. Size: small once the feed exists.

## 5. What to measure afterwards

- Clicks and seconds from launch to watching the current champion. Today at least two clicks and a tab switch; without clicks the player never sees it. Target: zero clicks.
- Clicks to resume the latest save. Today three clicks, a typed path and a wasted population. Target: one click.
- Clicks to set one effect to a chosen level. Today up to four, each a re-test. Target: one.
- Active effects visible without scrolling at 1440 by 900. Today 9 of 13 rows fit. Target: every effect away from calm.
- Visible settings in the default UI. Today about 20 plus 29 checkboxes. Target: five or fewer, none required.
- Research words on the main screen: QD, niche, emitter, CMA, topology reserve, gait score, mutability. Today seven. Target: none outside Diagnostics.
- Map fill time while evolving at 3M. Today it did not fill at 4,000 creatures within 8 s. Target: a complete map at all times, updated in place.
- Explained changes. Run a scripted session with the owner and one new player, thinking aloud: start, watch five generations, change the ground to Rocky, find the new best, race it against the old champion, export a GIF, save, quit, resume. Note every "why did that change?" and "where is …?". Repeat after the changes.
- Use of each path. An opt-in local log (for example `EVOLUTION_UI_LOG=path`, one CSV row per event) of replays by source (champion, card, map, record, race, lineage), tab visits, effect clicks and exports. A path nobody uses after a week of play is a candidate for removal.
- Lost runs. Closes with unsaved generations and no confirmation should be zero.
- No cost to evolution. The 3M GUI rate, frame p95 and control latency (`EVOLUTION_BENCH_SETTINGS_PROBE=1`) stay within run-to-run noise, because following the champion, the feed and the map must not slow the worker.

## 6. After the overhaul

Branch `claude/ux`, merged into main as it went (last UX commit 80565ac). Every quick win and structural change from section 4 landed, one commit each. The owner's readability feedback then added three more commits: contrast in both themes, bigger type, and alignment. Two items from the plan changed on the coordinator's instructions. Launch still starts playing at once, with no chooser. The opt-in usage log from section 5 was dropped, because it would be a setting.

Screenshots after the overhaul were taken CPU-only (4,000 creatures, 2 s trials) while the RTX was busy. The 3M benchmark below ran on the RTX.

| File | What it shows |
| --- | --- |
| [after-overview.png](ux-audit/after-overview.png) | Overview: tiles, the champion replay, chart and event feed |
| [after-overview-dark.png](ux-audit/after-overview-dark.png) | The same in the dark theme |
| [after-overview-1920.png](ux-audit/after-overview-1920.png) | The same laid out 1920 px wide (UI scale 0.75) |
| [after-archive-cards.png](ux-audit/after-archive-cards.png) | Ways of moving: filters, cards, docked replay |
| [after-archive-map.png](ux-audit/after-archive-map.png) | The map, filled while evolving |
| [after-history.png](ux-audit/after-history.png) | History: chart and one records list |
| [after-race.png](ux-audit/after-race.png) | Race, scaled to the farthest finish |
| [after-lineage.png](ux-audit/after-lineage.png) | Lineage with species names |

### Section 5 counts, before and after

| Measure | Before | After |
| --- | --- | --- |
| Clicks from launch to watching the champion | 2 or more and a tab switch; never without clicks | 0: the Overview shows it after generation 1 |
| Clicks to resume the latest save | 3, a typed path and a wasted 3M population | 2: File, then Open on the newest row |
| Clicks to set one effect to a chosen level | up to 4, each a re-test | 1 |
| Effects visible without scrolling at 1440x900 | 9 of 13 | 13 of 13, plus seasons and catastrophes |
| Visible settings in the default UI | about 20, plus 29 percentile checkboxes | 4, none required: autosave (File), dark theme and UI scale (View), seed (New dialog) |
| Research words on the main screen (QD, niche, emitter, CMA, topology reserve, gait score, mutability) | 7 | 0; they live in the closed Diagnostics drawer |
| Map cells placed after 8 s of evolving (4,000 creatures) | 0 ("0 / 1025") | the full table, 43 to 44 cells |

The resume count stays at 2 clicks because the owner wants no chooser at launch.

### Cost to evolution

The 3M GUI benchmark ran on the RTX under the GPU lock. Setup: a fresh 3M population, 20 s trials, one warm-up generation and two measured ones, `EVOLUTION_BENCH_SETTINGS_PROBE=1`, autosave off, `EVOLUTION_DEVICES=primary`, `RAYON_NUM_THREADS=8`. The Overview tab was on screen. Runs were interleaved, two per binary. The before binary is e161f51. The after binary is main at f371e70, which carries the UX work and also main's other changes since e161f51 (GPU authority, packing, the second screening rung, small saves). So the rate column bounds the combined effect and does not isolate the UI. The frame times are the UI thread's own.

| Binary | Run | End to end | Frame p95 | FPS | Control p95 / p99 | Settings median / max |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| e161f51 | 1 | 346,246/s | 9.59 ms | 118.5 | 723 / 829 ms | 117 / 133 ms |
| after | 1 | 356,337/s | 9.63 ms | 118.5 | 381 / 551 ms | 145 / 372 ms |
| e161f51 | 2 | 331,553/s | 9.56 ms | 118.4 | 1,417 / 1,925 ms | 198 / 278 ms |
| after | 2 | 367,805/s | 9.80 ms | 118.5 | 398 / 538 ms | 17 / 41 ms |

Evolution is not slower. The after binary ran 3 to 11% faster end to end, and its control latency was lower. With two runs each, a 10% spread between single runs and main's other changes in the mix, the benchmark shows no cost from the UI; it does not show a UI gain. The UI thread's frame p95 rose by 0.04 to 0.24 ms (9.6 to 9.8 ms), and both binaries held the 120 FPS cap.

### Known gaps

- The Generation tile can read "100% done" while the last contender checks still run. The completed count reaches the population before the checks finish.
- The event feed and the chart read the history rows, so a world change shows up in them only when the first generation in the new world ends. The world panel shows it at once.
- The scripted think-aloud session from section 5 has not happened. It needs the owner.
