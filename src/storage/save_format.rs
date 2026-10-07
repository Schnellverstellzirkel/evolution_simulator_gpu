use super::*;

// The file starts with the magic, then a small uncompressed header
// (`SaveHeader`), so the game can turn down a save it cannot use before it
// reads gigabytes. The body keeps only the archives and the search state
// (`SmallSave`). A loaded game breeds its population from the archives again
// Any other magic is an older format
// and is turned down.

const MAGIC: &[u8; 8] = b"EVORUST8";

/// What a save holds: the archives and the search state, without the
/// population, its scores, or anything bred for the generation in progress.
/// It holds the islands and the nurseries of new bodies, and none of the
/// nurseries of reshaped bodies: they refill from the bodies the islands
/// turn away, and a save stays as small as it was.
#[derive(Serialize)]
struct SmallSave<'a> {
    config: &'a Config,
    pending: &'a Option<Config>,
    generation: u32,
    history: &'a [Stats],
    archive: &'a QdArchive,
    emitter_stats: &'a [EmitterStats; qd::EMITTER_COUNT],
    cma_emitters: Vec<&'a CmaEmitter>,
    qd_version: u32,
    breed_round: u64,
    islands: &'a [QdArchive],
    lineage: SavedLineage<'a>,
    island_progress: &'a [(f32, u32)],
    reseed: &'a Reseed,
    ring: RingShape,
    audit: &'a crate::rungs::Audit,
}
#[derive(Deserialize)]
struct SmallLoad {
    config: Config,
    pending: Option<Config>,
    generation: u32,
    history: Vec<Stats>,
    archive: QdArchive,
    emitter_stats: [EmitterStats; qd::EMITTER_COUNT],
    cma_emitters: Vec<CmaEmitter>,
    qd_version: u32,
    breed_round: u64,
    islands: Vec<QdArchive>,
    lineage: KeyMap<Ancestor>,
    island_progress: Vec<(f32, u32)>,
    reseed: Reseed,
    ring: RingShape,
    audit: crate::rungs::Audit,
}

/// The lineage a save keeps, written in the layout of `HashMap<u64,
/// Ancestor>`. Every living elite keeps its own record, because breeding
/// reads its distance and early-rung features, but not its creature: the
/// archive holds that, and loading puts it back. The ancestors of the global
/// archive's elites and of each island's fastest few keep their creatures
/// back to `ANCESTRY_DEPTH`, which is as far as anything shows them. The
/// ancestors of the other island elites are left out.
struct SavedLineage<'a> {
    lineage: &'a KeyMap<Ancestor>,
    /// Which records to write, and whether each writes its creature.
    keep: HashMap<u64, bool>,
}
impl<'a> SavedLineage<'a> {
    /// The lineage of the global archive and of the first `held` archives
    /// of the experiment, the ones a save holds.
    fn of(e: &'a Experiment, held: usize) -> Self {
        let archives = || std::iter::once(&e.archive).chain(e.islands[..held].iter());
        let mut keep: HashMap<u64, bool> = HashMap::new();
        for elite in archives().flat_map(|archive| &archive.entries) {
            keep.insert(elite.creature.id, false);
        }
        let mut shown: Vec<u64> = e.archive.entries.iter().map(|x| x.creature.id).collect();
        for island in e.islands.iter().take(island_count()) {
            let mut fastest: Vec<&qd::Elite> = island.entries.iter().collect();
            fastest.sort_by(|a, b| b.fitness.total_cmp(&a.fitness));
            shown.extend(fastest.iter().take(ISLAND_LEADERS).map(|x| x.creature.id));
        }
        for start in shown {
            let mut current = e.lineage.get(&start).and_then(|a| a.parent);
            for _ in 0..ANCESTRY_DEPTH {
                let Some(id) = current else { break };
                let Some(ancestor) = e.lineage.get(&id) else {
                    break;
                };
                if keep.get(&id) == Some(&true) {
                    // The rest of this chain is already kept.
                    break;
                }
                // A living elite keeps its record without its creature.
                keep.entry(id).or_insert(true);
                current = ancestor.parent;
            }
        }
        Self {
            lineage: &e.lineage,
            keep,
        }
    }
}
impl Serialize for SavedLineage<'_> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        /// `Ancestor`, field for field.
        #[derive(Serialize)]
        struct Record<'b> {
            parent: Option<u64>,
            creature: &'b StoredCreature,
            fitness: f32,
            generation: u32,
            change: &'b str,
            rung: &'b [u16; 2 * crate::rungs::FEATURES],
        }
        let none = StoredCreature::default();
        let records: Vec<(&u64, Record)> = self
            .keep
            .iter()
            .filter_map(|(id, &with_creature)| {
                let a = self.lineage.get(id)?;
                Some((
                    id,
                    Record {
                        parent: a.parent,
                        creature: if with_creature { &a.creature } else { &none },
                        fitness: a.fitness,
                        generation: a.generation,
                        change: &a.change,
                        rung: &a.rung,
                    },
                ))
            })
            .collect();
        serializer.collect_map(records)
    }
}
impl<'a> SmallSave<'a> {
    fn of(e: &'a Experiment) -> Self {
        // The archives the save holds: each island and its nursery of new
        // bodies.
        let held = e.islands.len().min(island_count() * 2);
        let islands = &e.islands[..held];
        Self {
            config: &e.config,
            pending: &e.pending,
            generation: e.generation,
            history: &e.history,
            archive: &e.archive,
            emitter_stats: &e.emitter_stats,
            cma_emitters: e.cma_emitters.iter().filter(|c| c.island < held).collect(),
            qd_version: e.qd_version,
            breed_round: e.breed_round,
            islands,
            lineage: SavedLineage::of(e, held),
            island_progress: &e.island_progress,
            reseed: &e.reseed,
            ring: e.ring,
            audit: &e.rungs,
        }
    }
}
impl SmallLoad {
    /// The game the save describes, at the start of its saved generation.
    /// Its ring is bred from the archives, as the game would have bred it;
    /// without elites it starts with new random bodies.
    /// `saved_version` is the version the file's header names: an older one
    /// that still loads (`qd::loadable`) gets its archives re-binned.
    fn into_experiment(self, saved_version: u32) -> Result<Experiment> {
        let mut e = Experiment::empty(self.config);
        e.pending = self.pending;
        e.generation = self.generation;
        e.history = repair_history(self.history, self.generation);
        e.archive = self.archive;
        e.archive.set_global(true);
        e.emitter_stats = self.emitter_stats;
        e.cma_emitters = self.cma_emitters;
        e.qd_version = self.qd_version;
        e.breed_round = self.breed_round;
        e.islands = self.islands;
        e.lineage = self.lineage;
        e.island_progress = self.island_progress;
        // A save holds each island and its nursery of new bodies, and the
        // nurseries of reshaped bodies start empty (`SmallSave`).
        if e.islands.len() == island_count() * 2 {
            e.islands.resize_with(arena_count(), new_reshaped_nursery);
            if e.island_progress.len() == island_count() * 2 {
                e.island_progress
                    .resize(arena_count(), (f32::NEG_INFINITY, e.generation));
            }
        }
        e.reseed = self.reseed;
        e.rungs = self.audit;
        ensure!(
            self.ring.block > 0 && self.ring.blocks > 0,
            "Invalid ring shape"
        );
        // The ring keeps the shape it was saved with, so a resumed game
        // continues the search the uninterrupted one would have run.
        e.ring = self.ring;
        ensure!(
            e.island_progress.len() <= arena_count()
                && (e.island_progress.is_empty() || e.island_progress.len() == e.islands.len())
                && e.island_progress.iter().all(|&(fitness, generation)| {
                    (fitness.is_finite() || fitness == f32::NEG_INFINITY)
                        && generation <= e.generation
                }),
            "Invalid checkpoint optimizer progress"
        );
        ensure!(
            (e.islands.is_empty() || e.islands.len() == arena_count())
                && e.reseed.fits(island_count())
                && e.cma_emitters.iter().all(|c| c.island < arena_count()),
            "Invalid island state"
        );
        let refinable =
            std::iter::once(&mut e.archive).chain(e.islands.iter_mut().take(island_count()));
        if saved_version == qd::VERSION {
            for archive in refinable {
                archive.rebuild_indices();
                archive.derive_refined();
            }
            for nursery in e.islands.iter_mut().skip(island_count()) {
                nursery.rebuild_indices();
            }
        } else {
            // The archives were saved under another layout: each elite moves
            // to its cell now. A version 54 archive with elites in body
            // classes was refined. Optimizers keep their body plan's state;
            // the CMA emitters of single cells start over.
            for archive in refinable {
                if saved_version >= 54 {
                    archive.rebuild_indices();
                    archive.derive_refined();
                }
                archive.rebin();
            }
            for nursery in e.islands.iter_mut().skip(island_count()) {
                nursery.rebin();
            }
            e.cma_emitters.retain(CmaEmitter::optimizing);
        }
        // The global archive is always refined: it never breeds, so it has no
        // climb to protect. The islands are refined at the next generation
        // boundary if they are old enough.
        if !e.archive.refined() {
            e.archive.set_refined(true);
            e.archive.rebin();
        }
        // The lineage records of living elites were saved without a creature.
        for elite in std::iter::once(&e.archive)
            .chain(&e.islands)
            .flat_map(|archive| &archive.entries)
        {
            if let Some(record) = e.lineage.get_mut(&elite.creature.id)
                && record.creature.is_empty()
            {
                record.creature = elite.creature.clone();
            }
        }
        // The screen bar is not saved: the resumed generation runs every
        // trial in full until it has set a new one.
        e.config.screen = e.next_screen(e.config.duration);
        // The rungs' rules are not saved either: they are the window's fit.
        e.config.rungs = e.rungs.fit(e.global_stalled());
        let shared = Arc::new(e.config.clone());
        let elites =
            e.archive.entries.len() + e.islands.iter().map(|i| i.entries.len()).sum::<usize>();
        for (first, count) in e.ring.ranges(e.ring.len(e.config.population)) {
            let block = if elites == 0 && e.reseed.is_empty() {
                // Nothing to breed from: the new bodies of a new game.
                Block {
                    first,
                    population: Arc::new(evolution::random_block(&e.config, first, count)),
                    births: vec![Birth::RANDOM; count],
                    config: Arc::clone(&shared),
                    wild_bars: Arc::default(),
                }
            } else {
                // Plans sample every island's archive; an empty one breeds
                // new random bodies.
                Block {
                    config: Arc::clone(&shared),
                    ..e.breed_block(first, count, Arc::default())
                }
            };
            e.blocks.push(block);
        }
        e.validate()?;
        Ok(e)
    }
}

/// Autosaves kept in `dir`: the newest `keep` `seed-*-auto.evo` files stay,
/// older ones are deleted, and so are `.evo.tmp` files that an interrupted
/// save left behind more than ten minutes ago. Files the player saved under
/// other names are never touched. Returns how many files were removed.
pub fn rotate_autosaves(dir: &Path, keep: usize) -> usize {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    let now = std::time::SystemTime::now();
    let mut autosaves = Vec::new();
    let mut removed = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        let Ok(modified) = entry.metadata().and_then(|m| m.modified()) else {
            continue;
        };
        if name.ends_with(".evo.tmp") {
            let stale = now
                .duration_since(modified)
                .is_ok_and(|age| age.as_secs() > 600);
            if stale && std::fs::remove_file(&path).is_ok() {
                removed += 1;
            }
        } else if name.starts_with("seed-") && name.ends_with("-auto.evo") {
            autosaves.push((modified, path));
        }
    }
    autosaves.sort_by_key(|entry| std::cmp::Reverse(entry.0));
    for (_, path) in autosaves.into_iter().skip(keep) {
        if std::fs::remove_file(&path).is_ok() {
            removed += 1;
        }
    }
    removed
}
/// What a save's header says, read without decoding the save.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaveHeader {
    pub qd_version: u32,
    pub generation: u32,
    pub population: u64,
}
impl SaveHeader {
    const BYTES: usize = 16;
    fn of(experiment: &Experiment) -> Self {
        Self {
            qd_version: experiment.qd_version,
            generation: experiment.generation,
            population: experiment.config.population as u64,
        }
    }
    fn to_bytes(self) -> [u8; Self::BYTES] {
        let mut bytes = [0; Self::BYTES];
        bytes[..4].copy_from_slice(&self.qd_version.to_le_bytes());
        bytes[4..8].copy_from_slice(&self.generation.to_le_bytes());
        bytes[8..].copy_from_slice(&self.population.to_le_bytes());
        bytes
    }
    fn from_bytes(bytes: [u8; Self::BYTES]) -> Self {
        let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        Self {
            qd_version: word(0),
            generation: word(4),
            population: u64::from_le_bytes(bytes[8..].try_into().unwrap()),
        }
    }
}

/// Progress of a load or save that runs on another thread: file bytes read
/// or written so far, the file size when known, and a flag that stops it.
#[derive(Default)]
pub struct Progress {
    pub done: std::sync::atomic::AtomicU64,
    pub total: std::sync::atomic::AtomicU64,
    pub cancel: std::sync::atomic::AtomicBool,
}

/// A file that counts its bytes into a `Progress` and fails once cancelled.
struct Counted<'a, T> {
    inner: T,
    progress: Option<&'a Progress>,
}
impl<T> Counted<'_, T> {
    fn count(&self, bytes: usize) -> std::io::Result<()> {
        use std::sync::atomic::Ordering::Relaxed;
        if let Some(progress) = self.progress {
            if progress.cancel.load(Relaxed) {
                return Err(std::io::Error::other("cancelled"));
            }
            progress.done.fetch_add(bytes as u64, Relaxed);
        }
        Ok(())
    }
}
impl<T: Read> Read for Counted<'_, T> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.count(n)?;
        Ok(n)
    }
}
impl<T: Write> Write for Counted<'_, T> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.count(n)?;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Reads a save's header. Saves from before the header are from older game
/// versions, whose creatures were scored under other physics.
pub fn peek(path: &Path) -> Result<SaveHeader> {
    let mut file = File::open(path).with_context(|| format!("Cannot open {}", path.display()))?;
    let mut magic = [0; 8];
    file.read_exact(&mut magic)
        .with_context(|| format!("{} is not a save of this game", path.display()))?;
    reject_other_formats(path, &magic)?;
    let mut header = [0; SaveHeader::BYTES];
    file.read_exact(&mut header)
        .with_context(|| format!("{} is cut short", path.display()))?;
    Ok(SaveHeader::from_bytes(header))
}

/// Turns down a file that is not a save in the current format.
fn reject_other_formats(path: &Path, magic: &[u8; 8]) -> Result<()> {
    if magic == MAGIC {
        return Ok(());
    }
    ensure!(
        magic.starts_with(b"EVORUST"),
        "{} is not a save of this game",
        path.display()
    );
    anyhow::bail!(
        "{} was saved by an older version of the game, under older physics. It cannot be loaded; start a new population instead.",
        path.display()
    )
}

/// The header of a save the game can load now, or a message saying why not.
pub fn check(path: &Path) -> Result<SaveHeader> {
    let header = peek(path)?;
    ensure_current_version(path, &header)?;
    Ok(header)
}

fn ensure_current_version(path: &Path, header: &SaveHeader) -> Result<()> {
    ensure!(
        qd::loadable(header.qd_version),
        "{} was saved under physics version {}, and this game uses version {}. Its scores no longer hold, so it cannot be loaded; start a new population instead.",
        path.display(),
        header.qd_version,
        qd::VERSION
    );
    Ok(())
}

pub fn save(path: &Path, experiment: &Experiment) -> Result<()> {
    save_with_progress(path, experiment, None)
}

/// Saves through a temporary file that is renamed only once complete; a
/// failed or cancelled save removes it.
pub fn save_with_progress(
    path: &Path,
    experiment: &Experiment,
    progress: Option<&Progress>,
) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let tmp = path.with_extension("evo.tmp");
    let written = (|| -> Result<()> {
        let file = File::create(&tmp)?;
        let mut out = BufWriter::new(Counted {
            inner: file,
            progress,
        });
        out.write_all(MAGIC)?;
        out.write_all(&SaveHeader::of(experiment).to_bytes())?;
        let mut encoder = zstd::stream::write::Encoder::new(out, 3)?;
        encoder.include_checksum(true)?;
        // The same creature sits in the global archive and in an island, far
        // apart in the stream: matching over 128 MB made a save 28% smaller.
        encoder.long_distance_matching(true)?;
        encoder.window_log(27)?;
        // bincode writes field by field; a buffer turns each write into a
        // copy instead of a call into the compressor.
        let mut buffered = BufWriter::with_capacity(1 << 20, encoder);
        bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .serialize_into(&mut buffered, &SmallSave::of(experiment))?;
        bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .serialize_into(&mut buffered, &experiment.last_migration)?;
        let encoder = buffered.into_inner().map_err(|error| error.into_error())?;
        let mut out = encoder.finish()?;
        out.flush()?;
        out.get_ref().inner.sync_all()?;
        Ok(())
    })();
    if let Err(error) = written {
        let _ = std::fs::remove_file(&tmp);
        return Err(error);
    }
    std::fs::rename(&tmp, path)?;
    // Unix permits opening directories to persist the rename. Windows rejects
    // File::open on a directory; the checkpoint file itself was synced above.
    #[cfg(unix)]
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        File::open(parent)?.sync_all()?;
    }
    Ok(())
}
/// What the start of a checkpoint says about it: enough to list a save
/// without loading its population.
pub struct SaveSummary {
    pub generation: u32,
    pub config: Config,
}
/// Reads the settings and generation at the start of a checkpoint in the
/// current format. The payload begins with them, so only a few kilobytes are
/// decompressed. Other formats, other physics versions and unreadable files
/// give None.
pub fn summary(path: &Path) -> Option<SaveSummary> {
    #[derive(Deserialize)]
    struct Head {
        config: Config,
        _pending: Option<Config>,
        generation: u32,
    }
    let mut file = BufReader::new(File::open(path).ok()?);
    let mut magic = [0; 8];
    file.read_exact(&mut magic).ok()?;
    if &magic != MAGIC {
        return None;
    }
    let mut header = [0; SaveHeader::BYTES];
    file.read_exact(&mut header).ok()?;
    if !qd::loadable(SaveHeader::from_bytes(header).qd_version) {
        return None;
    }
    let decoder = zstd::stream::read::Decoder::new(file).ok()?;
    let head: Head = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(1 << 20)
        .deserialize_from(decoder)
        .ok()?;
    Some(SaveSummary {
        generation: head.generation,
        config: head.config,
    })
}
pub fn load(path: &Path) -> Result<Experiment> {
    load_with_progress(path, None)
}

/// `load`, counting the file bytes read into `progress`.
/// One row per generation, in order, up to `generation`: a skipped
/// generation gets a copy of the row before it, and a repeated one is dropped.
fn repair_history(history: Vec<Stats>, generation: u32) -> Vec<Stats> {
    let mut out: Vec<Stats> = Vec::with_capacity(history.len());
    for stats in history {
        if stats.generation < out.len() as u32 || stats.generation > generation {
            continue;
        }
        while (out.len() as u32) < stats.generation {
            let mut fill = out.last().unwrap_or(&stats).clone();
            fill.generation = out.len() as u32;
            out.push(fill);
        }
        out.push(stats);
    }
    while !out.is_empty() && (out.len() as u32) < generation {
        let mut fill = out[out.len() - 1].clone();
        fill.generation = out.len() as u32;
        out.push(fill);
    }
    out
}

pub fn load_with_progress(path: &Path, progress: Option<&Progress>) -> Result<Experiment> {
    load_from(path, progress, false, None)
}

/// `load` for a diagnostic that continues a big save on a small machine
/// (`examples/search_ab.rs --load`): the game runs `population` creatures per
/// generation, so its ring is bred for that and holds no more.
#[doc(hidden)]
pub fn load_for_population(path: &Path, population: usize) -> Result<Experiment> {
    load_from(path, None, false, Some(population))
}

/// `load` for diagnostics that only breed from the archives
/// (`examples/breed_bench.rs`): a save of an older physics version loads
/// too, with the scores it measured then.
#[doc(hidden)]
pub fn load_any_version(path: &Path) -> Result<Experiment> {
    load_from(path, None, true, None)
}

/// `load` for diagnostics that only read the archives
/// (`examples/island_report.rs`): the ring is one block of 64 creatures, not
/// the saved ring, so a 3M save loads in a few hundred MB.
#[doc(hidden)]
pub fn load_archives(path: &Path) -> Result<Experiment> {
    load_from(path, None, false, Some(64))
}

fn load_from(
    path: &Path,
    progress: Option<&Progress>,
    any_version: bool,
    population: Option<usize>,
) -> Result<Experiment> {
    let file = File::open(path).context("Cannot open checkpoint")?;
    if let Some(progress) = progress {
        progress.total.store(
            file.metadata().map_or(0, |m| m.len()),
            std::sync::atomic::Ordering::Relaxed,
        );
    }
    let mut file = BufReader::new(Counted {
        inner: file,
        progress,
    });
    let mut magic = [0; 8];
    file.read_exact(&mut magic)
        .with_context(|| format!("{} is not a save of this game", path.display()))?;
    reject_other_formats(path, &magic)?;
    let mut header = [0; SaveHeader::BYTES];
    file.read_exact(&mut header)
        .with_context(|| format!("{} is cut short", path.display()))?;
    let saved_version = SaveHeader::from_bytes(header).qd_version;
    if !any_version {
        ensure_current_version(path, &SaveHeader::from_bytes(header))?;
    }
    // bincode reads field by field; a buffer turns each read into a copy
    // instead of a call into the decompressor (18 s to 5 s at 3M).
    let mut decoder =
        BufReader::with_capacity(1 << 20, zstd::stream::read::Decoder::with_buffer(file)?);
    let mut small: SmallLoad = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(24 * 1024 * 1024 * 1024)
        .deserialize_from(&mut decoder)?;
    let migration: Option<(u32, Vec<(usize, usize)>)> = bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .with_limit(16 + 16 * island_count() as u64)
        .deserialize_from(&mut decoder)?;
    let mut trailing = [0u8; 1];
    ensure!(
        decoder.read(&mut trailing)? == 0,
        "Unexpected trailing checkpoint data"
    );
    small.qd_version = qd::VERSION;
    if let Some(population) = population {
        small.config.population = population;
    }
    let mut experiment = small.into_experiment(saved_version)?;
    experiment.last_migration = migration.filter(|(generation, exchange)| {
        *generation <= experiment.generation && exchange.len() == island_count()
    });
    Ok(experiment)
}

pub fn export_csv(path: &Path, history: &[Stats]) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    let mut w = csv::Writer::from_path(path)?;
    w.write_record([
        "generation",
        "population",
        "best_m",
        "median_m",
        "worst_m",
        "mean_m",
        "failed",
        "evaluation_seconds",
        "seed",
        "archive_cells",
        "qd_score",
        "archive_coverage",
    ])?;
    for s in history {
        w.serialize((
            s.generation,
            s.population,
            s.best,
            s.median,
            s.worst,
            s.mean,
            s.failed,
            s.seconds,
            s.config.seed,
            s.archive_cells,
            s.qd_score,
            s.archive_coverage,
        ))?;
    }
    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod peek_tests {
    use super::*;

    #[test]
    fn peek_reads_the_generation_and_world_of_a_save() {
        let config = Config {
            population: 64,
            random_seed: false,
            terrain: 2,
            ..Config::default()
        };
        let mut experiment = Experiment::new(config).unwrap();
        experiment.generation = 7;
        let dir = std::env::temp_dir().join(format!("evo-peek-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("peek.evo");
        save(&path, &experiment).unwrap();
        let found = summary(&path).expect("a current checkpoint can be peeked");
        assert_eq!(found.generation, 7);
        assert_eq!(found.config.terrain, 2);
        assert_eq!(found.config.population, 64);
        std::fs::write(&path, b"not a checkpoint").unwrap();
        assert!(summary(&path).is_none());
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
#[cfg(test)]
mod migration_tests {
    use super::*;

    /// Deterministic made-up results: a distance and a behavior from each
    /// creature's id.
    fn synthetic(pop: &Population, _: &Config) -> Result<Vec<EvaluationMetrics>> {
        Ok(pop
            .genomes
            .iter()
            .map(|g| {
                let h = g.id.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 20;
                EvaluationMetrics {
                    fitness: 1.0 + (h % 1000) as f32 * 0.02,
                    behavior: qd::TrialMetrics {
                        ground_contact: ((h >> 10) % 6) as f32 / 6.0 + 0.05,
                        gait_frequency: ((h >> 13) % 8) as f32 * 0.75 + 0.1,
                        mean_height: ((h >> 16) % 6) as f32 * 0.3 + 0.05,
                        feet: ((h >> 19) % 5) as f32,
                        ..Default::default()
                    },
                    ..Default::default()
                }
            })
            .collect())
    }

    #[test]
    fn the_header_turns_down_old_saves_before_reading_them() {
        let config = Config {
            population: 8,
            random_seed: false,
            ..Config::default()
        };
        let mut experiment = Experiment::new(config).unwrap();
        let dir = std::env::temp_dir();
        let current = dir.join(format!("evolution-header-{}.evo", std::process::id()));
        save(&current, &experiment).unwrap();
        let header = check(&current).unwrap();
        assert_eq!(
            header,
            SaveHeader {
                qd_version: qd::VERSION,
                generation: experiment.generation,
                population: 8,
            }
        );
        assert_eq!(load(&current).unwrap().ring_len(), 8);

        // Saved under other physics: the header says so.
        experiment.qd_version = qd::OLDEST_LOADABLE - 1;
        save(&current, &experiment).unwrap();
        let error = check(&current).unwrap_err().to_string();
        assert!(error.contains("physics version"), "{error}");

        // Before the header: only the magic is read.
        let old = dir.join(format!("evolution-header-v6-{}.evo", std::process::id()));
        let mut bytes = b"EVORUST6".to_vec();
        bytes.extend([0u8; 64]);
        std::fs::write(&old, bytes).unwrap();
        let error = check(&old).unwrap_err().to_string();
        assert!(error.contains("older version"), "{error}");
        std::fs::write(&old, b"not a save").unwrap();
        assert!(check(&old).is_err());
        let _ = std::fs::remove_file(current);
        let _ = std::fs::remove_file(old);
    }

    #[test]
    fn a_cancelled_save_leaves_no_file_behind() {
        let config = Config {
            population: 8,
            random_seed: false,
            ..Config::default()
        };
        let experiment = Experiment::new(config).unwrap();
        let path =
            std::env::temp_dir().join(format!("evolution-cancel-{}.evo", std::process::id()));
        let progress = Progress::default();
        progress
            .cancel
            .store(true, std::sync::atomic::Ordering::Relaxed);
        assert!(save_with_progress(&path, &experiment, Some(&progress)).is_err());
        assert!(!path.exists());
        assert!(!path.with_extension("evo.tmp").exists());
    }

    #[test]
    fn a_checkpoint_mid_generation_resumes_that_generation() {
        let config = Config {
            population: 40,
            random_seed: false,
            ..Config::default()
        };
        let mut experiment = Experiment::new(config).unwrap();
        experiment.step(&mut synthetic).unwrap();
        assert!(experiment.evaluated > 0);
        let checkpoint =
            std::env::temp_dir().join(format!("evolution-steady-{}.evo", std::process::id()));
        // A save keeps no ring: the loaded game starts its saved generation
        // again with a ring bred from its archives.
        save(&checkpoint, &experiment).unwrap();
        let loaded = load(&checkpoint).unwrap();
        let _ = std::fs::remove_file(checkpoint);
        assert_eq!(loaded.evaluated, 0);
        assert_eq!(loaded.generation, experiment.generation);
        assert_eq!(loaded.ring_len(), 40);
        assert!(
            loaded
                .blocks
                .iter()
                .flat_map(|b| &b.births)
                .any(|b| b.emitter != Emitter::Restart)
        );
    }

    #[test]
    fn checkpoint_round_trip_keeps_the_autochange_step() {
        let config = Config {
            population: 2,
            random_seed: false,
            autochange: 2,
            ..Config::default()
        };
        let mut experiment = Experiment::new(config).unwrap();
        experiment.config.autochange_step = 7;
        experiment.config.wind = crate::environment::WIND[2];
        let checkpoint = std::env::temp_dir().join(format!(
            "evolution-autochange-step-{}.evo",
            std::process::id()
        ));
        save(&checkpoint, &experiment).unwrap();
        let loaded = load(&checkpoint).unwrap();
        let _ = std::fs::remove_file(checkpoint);
        assert_eq!(loaded.config.autochange, 2);
        assert_eq!(loaded.config.autochange_step, 7);
        assert_eq!(loaded.config.wind, experiment.config.wind);
    }

    #[test]
    fn a_saved_game_keeps_its_last_migration() {
        let config = Config {
            population: 64,
            random_seed: false,
            ..Config::default()
        };
        let mut experiment = Experiment::new(config).unwrap();
        experiment.run_generation(&mut synthetic).unwrap();
        let exchange: Vec<(usize, usize)> = (0..island_count()).map(|i| (i + 2, i + 1)).collect();
        assert!(!exchange.is_empty());
        experiment.last_migration = Some((experiment.generation, exchange.clone()));
        let checkpoint =
            std::env::temp_dir().join(format!("evolution-migration-{}.evo", std::process::id()));
        save(&checkpoint, &experiment).unwrap();
        let loaded = load(&checkpoint).unwrap();
        let _ = std::fs::remove_file(checkpoint);
        assert_eq!(
            loaded.last_migration,
            Some((experiment.generation, exchange))
        );
    }

    /// The autochange button at each speed: set while a generation runs, the
    /// level must survive the generation boundary and keep stepping.
    #[test]
    fn an_autochange_level_set_mid_generation_stays_and_steps() {
        for level in 1u8..=3 {
            let interval = crate::environment::AUTOCHANGE_INTERVALS[usize::from(level)];
            let config = Config {
                population: 4,
                random_seed: false,
                ..Config::default()
            };
            let mut experiment = Experiment::new(config).unwrap();
            for generation in 1..=interval * 2 {
                if generation == 2 {
                    // The panel sends its whole config, as the game does.
                    let mut cfg = experiment.config.clone();
                    cfg.autochange = level;
                    experiment.update_config(cfg).unwrap();
                }
                experiment.run_generation(&mut synthetic).unwrap();
                if generation >= 2 {
                    assert_eq!(
                        experiment.config.autochange, level,
                        "generation {generation}"
                    );
                }
                assert!(experiment.pending.is_none());
            }
            assert!(experiment.config.autochange_step >= 1, "level {level}");
        }
    }

    #[test]
    fn generation_boundaries_advance_the_autochange() {
        let interval = crate::environment::AUTOCHANGE_INTERVALS[2];
        let config = Config {
            population: 4,
            random_seed: false,
            autochange: 2,
            ..Config::default()
        };
        let mut experiment = Experiment::new(config).unwrap();
        for generation in 1..=interval {
            experiment.run_generation(&mut synthetic).unwrap();
            assert_eq!(experiment.generation, generation);
            if generation < interval {
                assert_eq!(experiment.config.autochange_step, 0);
            }
        }
        assert_eq!(experiment.config.autochange_step, 1);
        // The first rung of the ladder is on.
        let (effect, level) = crate::environment::autochange_ladder()[0];
        assert_eq!(
            crate::environment::EFFECTS[effect].level(&experiment.config),
            level
        );
    }
}
