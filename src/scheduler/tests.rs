//! These tests run the scheduler on a `FakeEngine` instead of a GPU. The fake
//! records each submission and returns the results and failures that a test
//! queues for it. The tests cover how work reaches an engine, how results come
//! back, how a reset and a world change treat waiting work, and how a lost GPU
//! is reopened. The last three check that a scheduler needs its primary GPU and
//! adds other devices only when `EVOLUTION_DEVICES` names them.

use super::*;
use crate::engine::Finished;
use std::sync::{Arc, Mutex};

/// One unit that a `FakeEngine` accepted, as the scheduler submitted it.
struct Submission {
    ticket: u64,
    population: Arc<Population>,
    config: Config,
}

/// What a `FakeEngine` records and what a test queues for it. A test shares
/// it with the engine through an `Arc<Mutex<..>>`, so the test can read the
/// submissions and add results while the scheduler owns the engine.
#[derive(Default)]
struct FakeState {
    /// Every unit the engine accepted, in order. The ticket of a unit is its
    /// position in this list plus one.
    submissions: Vec<Submission>,
    /// Finished units that `poll` returns, oldest first.
    results: VecDeque<Finished>,
    /// True from a submission until `poll` returns a result.
    pending: bool,
    /// An error for the next `poll` to return. It is returned once.
    poll_failure: Option<String>,
    /// An error for the next submission to return. It is returned once.
    submit_failure: Option<String>,
}

/// An `Engine` that reports one free slot until a unit is pending. The unit
/// finishes when a test queues its result in the shared `FakeState`.
struct FakeEngine {
    name: &'static str,
    state: Arc<Mutex<FakeState>>,
}

impl Engine for FakeEngine {
    fn name(&self) -> String {
        self.name.into()
    }

    fn max_nodes(&self) -> usize {
        64
    }

    fn free_slots(&self) -> usize {
        let state = self.state.lock().unwrap();
        // A pending unit or a waiting poll failure takes the slot. A waiting
        // submit failure leaves it free, so the scheduler tries and meets it.
        usize::from(!state.pending && state.poll_failure.is_none())
    }

    fn submit_shared(&mut self, population: Arc<Population>, config: &Config) -> Result<u64> {
        let mut state = self.state.lock().unwrap();
        if let Some(error) = state.submit_failure.take() {
            anyhow::bail!("{error}");
        }
        let ticket = state.submissions.len() as u64 + 1;
        state.submissions.push(Submission {
            ticket,
            population,
            config: config.clone(),
        });
        state.pending = true;
        Ok(ticket)
    }

    fn poll(&mut self) -> Result<Option<Finished>> {
        let mut state = self.state.lock().unwrap();
        if let Some(error) = state.poll_failure.take() {
            return Err(anyhow::anyhow!("{error}"));
        }
        let done = state.results.pop_front();
        if done.is_some() {
            state.pending = false;
        }
        Ok(done)
    }

    fn wait(&mut self, _timeout: Duration) {}
}

fn fake_device(name: &'static str, state: &Arc<Mutex<FakeState>>) -> Device {
    Device::new(
        Box::new(FakeEngine {
            name,
            state: Arc::clone(state),
        }),
        100.0,
    )
}

fn fake_scheduler() -> (Scheduler, Arc<Mutex<FakeState>>) {
    let state = Arc::new(Mutex::new(FakeState::default()));
    let scheduler = Scheduler::with_devices(vec![fake_device("fake gpu", &state)]);
    (scheduler, state)
}

/// Two fake GPUs and their states, for a test that needs a second engine.
fn two_gpus() -> (Scheduler, Arc<Mutex<FakeState>>, Arc<Mutex<FakeState>>) {
    let first = Arc::new(Mutex::new(FakeState::default()));
    let second = Arc::new(Mutex::new(FakeState::default()));
    let scheduler = Scheduler::with_devices(vec![
        fake_device("first gpu", &first),
        fake_device("second gpu", &second),
    ]);
    (scheduler, first, second)
}

/// A reopen hook that opens a fresh fake GPU on `state`. It allows three
/// attempts, each with no delay.
fn reopen_on(state: &Arc<Mutex<FakeState>>) -> Reopen {
    let state = Arc::clone(state);
    Reopen {
        open: Box::new(move || {
            Ok(Box::new(FakeEngine {
                name: "reopened gpu",
                state: Arc::clone(&state),
            }) as Box<dyn Engine>)
        }),
        backoff: vec![Duration::ZERO; 3],
    }
}

/// A small config for these tests: 4 creatures, 1 s trials and a fixed seed.
fn submission_config() -> Config {
    Config {
        population: 4,
        duration: 1.0,
        random_seed: false,
        ..Config::default()
    }
}

/// Queues the first half of `pop` as standard work with tag 0 and the second
/// half with tag 1.
fn queue_halves(scheduler: &mut Scheduler, pop: &Population, cfg: &Config) {
    let pop = Arc::new(pop.clone());
    let cfg = Arc::new(cfg.clone());
    let n = pop.genomes.len();
    scheduler.queue(
        0,
        Trial::Standard,
        Arc::clone(&pop),
        Some((0..n / 2).collect()),
        Arc::clone(&cfg),
    );
    scheduler.queue(1, Trial::Standard, pop, Some((n / 2..n).collect()), cfg);
}

/// Queues a finished unit for each submission from index `from` onward, in
/// order. Each carries its submission's ticket and one `GpuResult` per
/// creature, all with `fitness`.
fn complete_submissions(state: &Arc<Mutex<FakeState>>, from: usize, fitness: f32) {
    let mut state = state.lock().unwrap();
    let submissions: Vec<(u64, usize)> = state
        .submissions
        .iter()
        .skip(from)
        .map(|s| (s.ticket, s.population.genomes.len()))
        .collect();
    for (ticket, count) in submissions {
        state.results.push_back(Finished {
            ticket,
            results: vec![
                GpuResult {
                    fitness,
                    ..GpuResult::default()
                };
                count
            ],
            busy_seconds: 0.5,
        });
    }
}

/// Pumps the scheduler, completes each new submission on `state` and collects,
/// until nothing is in flight. Returns the index of every finished creature,
/// sorted. The test fails if this takes more than 5 s.
fn drain_all(scheduler: &mut Scheduler, state: &Arc<Mutex<FakeState>>) -> Vec<usize> {
    let mut seen = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut completed = 0;
    while scheduler.in_flight() > 0 {
        assert!(Instant::now() < deadline, "evaluation stalled");
        scheduler.pump().unwrap();
        let total = state.lock().unwrap().submissions.len();
        if total > completed {
            complete_submissions(state, completed, 3.0);
            completed = total;
        }
        for done in scheduler.collect(Duration::from_millis(5)).unwrap() {
            seen.extend(done.members);
        }
    }
    seen.sort_unstable();
    seen
}

#[test]
fn a_whole_population_goes_to_the_engine_without_a_copy() {
    let cfg = submission_config();
    let pop = Arc::new(crate::evolution::create(&cfg).unwrap());
    let (mut scheduler, state) = fake_scheduler();
    scheduler.queue(
        7,
        Trial::Standard,
        Arc::clone(&pop),
        None,
        Arc::new(cfg.clone()),
    );
    scheduler.pump().unwrap();
    assert!(Arc::ptr_eq(
        &state.lock().unwrap().submissions[0].population,
        &pop
    ));
    complete_submissions(&state, 0, 2.0);
    let done = scheduler.collect(Duration::ZERO).unwrap();
    assert_eq!(done.len(), 1);
    assert_eq!(done[0].tag, 7);
    assert_eq!(done[0].members, vec![0, 1, 2, 3]);
    assert!(done[0].metrics.iter().all(|m| m.fitness == 2.0));
    assert_eq!(scheduler.in_flight(), 0);
}

#[test]
fn confirmations_go_before_standard_work_and_keep_the_pose() {
    let cfg = submission_config();
    let pop = Arc::new(crate::evolution::create(&cfg).unwrap());
    let (mut scheduler, state) = fake_scheduler();
    let standard = Arc::new(cfg.clone());
    let fine = Arc::new(confirm_config(&cfg));
    scheduler.queue(1, Trial::Standard, Arc::clone(&pop), None, standard);
    scheduler.queue(2, Trial::Confirm, Arc::clone(&pop), Some(vec![3, 1]), fine);
    scheduler.pump().unwrap();
    let state = state.lock().unwrap();
    // The fake has one slot, so its one submission is the confirmation.
    let submission = &state.submissions[0];
    assert_eq!(
        submission.config,
        Config {
            fidelity: Some(crate::physics::Fidelity::fine()),
            ..cfg
        }
    );
    for (slot, &index) in [3, 1].iter().enumerate() {
        let expected = pop.creature(index);
        let actual = submission.population.creature(slot);
        assert_eq!(actual.id, expected.id);
        assert_eq!(actual.nodes, expected.nodes);
        assert_eq!(actual.bones, expected.bones);
        assert_eq!(actual.muscles, expected.muscles);
    }
}

#[test]
fn results_use_the_submitted_population_and_config() {
    let cfg = submission_config();
    let pop = crate::evolution::create(&cfg).unwrap();
    let (mut scheduler, state) = fake_scheduler();
    scheduler.queue(
        0,
        Trial::Standard,
        Arc::new(pop.clone()),
        Some(vec![3, 1]),
        Arc::new(cfg.clone()),
    );
    scheduler.pump().unwrap();
    let (ticket, results) = {
        let state = state.lock().unwrap();
        let submission = &state.submissions[0];
        assert_eq!(submission.config, cfg);
        let results = submission
            .population
            .genomes
            .iter()
            .map(|genome| GpuResult {
                fitness: genome.id as f32,
                ground_contact: cfg.steps() as f32 * genome.node_count as f32 * 0.25,
                height_sum: cfg.steps() as f32 * 1.5,
                ..GpuResult::default()
            })
            .collect();
        (submission.ticket, results)
    };
    state.lock().unwrap().results.push_back(Finished {
        ticket,
        results,
        busy_seconds: 0.5,
    });
    let output = scheduler.collect(Duration::ZERO).unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].members, vec![3, 1]);
    for (&index, metric) in output[0].members.iter().zip(&output[0].metrics) {
        assert_eq!(metric.fitness, pop.genomes[index].id as f32);
        assert_eq!(metric.behavior.ground_contact, 0.25);
        assert_eq!(metric.behavior.mean_height, 1.5);
    }
    assert_eq!(scheduler.in_flight(), 0);
}

#[test]
fn malformed_results_consume_no_work() {
    let cfg = submission_config();
    let pop = crate::evolution::create(&cfg).unwrap();
    // An unknown ticket, too few results and too many results.
    for (wrong_ticket, result_count) in [(true, 4), (false, 1), (false, 5)] {
        let (mut scheduler, state) = fake_scheduler();
        scheduler.queue(
            0,
            Trial::Standard,
            Arc::new(pop.clone()),
            None,
            Arc::new(cfg.clone()),
        );
        scheduler.pump().unwrap();
        let ticket = state.lock().unwrap().submissions[0].ticket;
        let before = scheduler.in_flight();
        state.lock().unwrap().results.push_back(Finished {
            ticket: ticket + u64::from(wrong_ticket),
            results: vec![GpuResult::default(); result_count],
            busy_seconds: 0.5,
        });
        assert!(scheduler.collect(Duration::ZERO).is_err());
        assert_eq!(scheduler.in_flight(), before, "pending work was consumed");
        assert_eq!(scheduler.devices[0].creatures, 0);
        complete_submissions(&state, 0, 1.0);
        let output = scheduler.collect(Duration::ZERO).unwrap();
        assert_eq!(output.iter().map(|d| d.members.len()).sum::<usize>(), 4);
        assert_eq!(scheduler.in_flight(), 0);
    }
}

#[test]
fn a_reset_drops_waiting_work_and_old_results() {
    let cfg = submission_config();
    let pop = crate::evolution::create(&cfg).unwrap();
    let (mut scheduler, state) = fake_scheduler();
    queue_halves(&mut scheduler, &pop, &cfg);
    // The first half is on the engine and the second half waits.
    scheduler.pump().unwrap();
    scheduler.reset();
    // The unit on the engine finishes, but its session is over.
    complete_submissions(&state, 0, 1.0);
    assert!(scheduler.collect(Duration::ZERO).unwrap().is_empty());
    assert_eq!(scheduler.in_flight(), 0);
}

#[test]
fn a_world_change_retargets_only_waiting_work() {
    let cfg = submission_config();
    let pop = crate::evolution::create(&cfg).unwrap();
    let (mut scheduler, state) = fake_scheduler();
    queue_halves(&mut scheduler, &pop, &cfg);
    // The first half is on the engine, so only the second half is retargeted.
    scheduler.pump().unwrap();
    let rough = Arc::new(Config {
        terrain: 3,
        ..cfg.clone()
    });
    assert_eq!(scheduler.retarget(&rough), vec![1]);
    complete_submissions(&state, 0, 1.0);
    scheduler.collect(Duration::ZERO).unwrap();
    scheduler.pump().unwrap();
    let state = state.lock().unwrap();
    assert_eq!(state.submissions[0].config.terrain, cfg.terrain);
    assert_eq!(state.submissions[1].config.terrain, 3);
}

#[test]
fn a_lost_gpu_is_reopened_and_gets_its_units_again() {
    let cfg = submission_config();
    let pop = crate::evolution::create(&cfg).unwrap();
    let (mut scheduler, gpu) = fake_scheduler();
    let reopened = Arc::new(Mutex::new(FakeState::default()));
    let engine_state = Arc::clone(&reopened);
    let mut attempts = 0;
    scheduler.devices[0].reopen = Some(Reopen {
        open: Box::new(move || {
            attempts += 1;
            // The first attempt finds no device, the second one opens.
            if attempts == 1 {
                anyhow::bail!("device not ready");
            }
            Ok(Box::new(FakeEngine {
                name: "reopened gpu",
                state: Arc::clone(&engine_state),
            }) as Box<dyn Engine>)
        }),
        backoff: vec![Duration::ZERO; 3],
    });
    queue_halves(&mut scheduler, &pop, &cfg);
    scheduler.pump().unwrap();
    let (gpu_units, gpu_populations) = {
        let gpu = gpu.lock().unwrap();
        assert!(!gpu.submissions.is_empty(), "the GPU took no work");
        (
            gpu.submissions.len(),
            gpu.submissions
                .iter()
                .map(|s| Arc::clone(&s.population))
                .collect::<Vec<_>>(),
        )
    };
    gpu.lock().unwrap().poll_failure = Some("device lost".into());
    assert!(scheduler.collect(Duration::ZERO).unwrap().is_empty());
    assert_eq!(scheduler.devices[0].engine.name(), "reopened gpu");
    {
        let state = reopened.lock().unwrap();
        assert_eq!(state.submissions.len(), gpu_units);
        for (again, original) in state.submissions.iter().zip(&gpu_populations) {
            assert!(
                Arc::ptr_eq(&again.population, original),
                "the resubmitted unit must keep the exact creatures"
            );
            assert_eq!(again.config, cfg);
        }
    }
    let notices = scheduler.take_notices();
    assert!(
        notices.iter().any(|n| n.contains("GPU is back")),
        "{notices:?}"
    );
    assert!(scheduler.devices[0].queued.iter().all(|u| u.retries == 1));
}

#[test]
fn a_gpu_that_never_reopens_stops_evolution() {
    let cfg = submission_config();
    let pop = crate::evolution::create(&cfg).unwrap();
    let (mut scheduler, gpu) = fake_scheduler();
    scheduler.devices[0].reopen = Some(Reopen {
        open: Box::new(|| anyhow::bail!("no device")),
        backoff: vec![Duration::ZERO; 3],
    });
    queue_halves(&mut scheduler, &pop, &cfg);
    scheduler.pump().unwrap();
    gpu.lock().unwrap().poll_failure = Some("device lost".into());
    // The error comes back on every call.
    for _ in 0..2 {
        let error = scheduler
            .collect(Duration::ZERO)
            .expect_err("the lost GPU must stop evolution");
        assert!(error.to_string().contains("device lost"), "{error}");
    }
    assert!(scheduler.in_flight() > 0, "the failure must stay visible");
    let notices = scheduler.take_notices();
    assert!(notices.iter().any(|n| n.contains("could not be reopened")));
}

#[test]
fn a_failed_gpu_submission_runs_on_the_reopened_gpu() {
    let cfg = submission_config();
    let pop = crate::evolution::create(&cfg).unwrap();
    let (mut scheduler, gpu) = fake_scheduler();
    let reopened = Arc::new(Mutex::new(FakeState::default()));
    scheduler.devices[0].reopen = Some(reopen_on(&reopened));
    gpu.lock().unwrap().submit_failure = Some("device lost".into());
    queue_halves(&mut scheduler, &pop, &cfg);
    scheduler.pump().unwrap();
    assert_eq!(gpu.lock().unwrap().submissions.len(), 0);
    // This call reopens the GPU that `pump` marked failed.
    scheduler.collect(Duration::ZERO).unwrap();
    assert_eq!(scheduler.devices[0].engine.name(), "reopened gpu");
    let seen = drain_all(&mut scheduler, &reopened);
    assert_eq!(seen, vec![0, 1, 2, 3], "every creature must finish once");
}

#[test]
fn a_gpu_that_cannot_reopen_is_terminal_and_delivers_completed_output_first() {
    let cfg = submission_config();
    let pop = crate::evolution::create(&cfg).unwrap();
    let (mut scheduler, first, second) = two_gpus();
    queue_halves(&mut scheduler, &pop, &cfg);
    scheduler.pump().unwrap();
    let (ticket, count) = {
        let first = first.lock().unwrap();
        let submission = &first.submissions[0];
        (submission.ticket, submission.population.genomes.len())
    };
    // One unit completes on the first GPU. The second GPU fails before it
    // returns anything and cannot reopen.
    first.lock().unwrap().results.push_back(Finished {
        ticket,
        results: vec![
            GpuResult {
                fitness: 7.0,
                ..GpuResult::default()
            };
            count
        ],
        busy_seconds: 0.5,
    });
    second.lock().unwrap().poll_failure = Some("second lost".into());
    let completed = scheduler.collect(Duration::ZERO).unwrap();
    assert_eq!(
        completed.iter().map(|d| d.members.len()).sum::<usize>(),
        count,
        "completed output must be delivered before the GPU error"
    );
    for _ in 0..3 {
        let error = scheduler
            .collect(Duration::ZERO)
            .expect_err("the GPU error must persist");
        assert!(error.to_string().contains("second lost"));
    }
    assert!(scheduler.in_flight() > 0, "the failure must stay visible");
}

#[test]
fn a_missing_primary_gpu_is_an_error() {
    // An invalid device name cannot open, and there is no CPU engine to fall
    // back on.
    let error = Scheduler::new("definitely not a gpu")
        .err()
        .expect("no scheduler without its GPU");
    assert!(error.to_string().contains("NVIDIA"), "{error:#}");
}

#[test]
fn secondary_devices_are_opt_in_and_selection_sentinels_disable_them() {
    assert_eq!(secondary_device_names(None), Vec::<&str>::new());
    assert_eq!(secondary_device_names(Some("primary")), Vec::<&str>::new());
    assert_eq!(secondary_device_names(Some("off")), Vec::<&str>::new());
    assert_eq!(secondary_device_names(Some(" OFF ")), Vec::<&str>::new());
}

#[test]
fn secondary_device_names_preserve_explicit_comma_separated_selection() {
    assert_eq!(
        secondary_device_names(Some("radeon, RTX 4060")),
        vec!["radeon", "RTX 4060"]
    );
    // A `primary` or `off` entry inside a list is dropped.
    assert_eq!(
        secondary_device_names(Some("primary, radeon, off")),
        vec!["radeon"]
    );
    assert_eq!(secondary_device_names(Some(" , ")), Vec::<&str>::new());
}
