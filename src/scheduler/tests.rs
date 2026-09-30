use super::*;
use crate::engine::Finished;
use std::sync::{Arc, Mutex};

struct Submission {
    ticket: u64,
    population: Arc<Population>,
    config: Config,
}

#[derive(Default)]
struct FakeState {
    submissions: Vec<Submission>,
    results: VecDeque<Finished>,
    pending: bool,
    /// Returned from the next poll.
    poll_failure: Option<String>,
    /// Returned from the next submission.
    submit_failure: Option<String>,
}

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
        // A pending submission failure still lets the scheduler try once.
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

fn fake_device(name: &'static str, kind: DeviceKind, state: &Arc<Mutex<FakeState>>) -> Device {
    Device::new(
        Box::new(FakeEngine {
            name,
            state: Arc::clone(state),
        }),
        kind,
        100.0,
    )
}

fn fake_scheduler() -> (Scheduler, Arc<Mutex<FakeState>>) {
    let state = Arc::new(Mutex::new(FakeState::default()));
    let scheduler = Scheduler::with_devices(
        vec![fake_device("fake evaluator", DeviceKind::Cpu, &state)],
        None,
    );
    (scheduler, state)
}

/// One fake GPU and one fake CPU, for recovery tests that need both.
fn mixed_scheduler() -> (Scheduler, Arc<Mutex<FakeState>>, Arc<Mutex<FakeState>>) {
    let gpu = Arc::new(Mutex::new(FakeState::default()));
    let cpu = Arc::new(Mutex::new(FakeState::default()));
    let scheduler = Scheduler::with_devices(
        vec![
            fake_device("fake gpu", DeviceKind::Gpu, &gpu),
            fake_device("fake cpu", DeviceKind::Cpu, &cpu),
        ],
        None,
    );
    (scheduler, gpu, cpu)
}

fn submission_config() -> Config {
    Config {
        population: 4,
        duration: 1.0,
        random_seed: false,
        ..Config::default()
    }
}

/// Queues `pop` as two pieces of work with tags 0 and 1.
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

/// Queues one valid result per submission from `from` onward, in order.
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

/// Pumps, completes every new submission on `state` and collects until the
/// scheduler is idle; returns every finished creature index, sorted.
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
    scheduler.pump().unwrap();
    scheduler.reset();
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
    let (mut scheduler, gpu, cpu) = mixed_scheduler();
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
    let cpu_units = cpu.lock().unwrap().submissions.len();
    gpu.lock().unwrap().poll_failure = Some("device lost".into());
    assert!(scheduler.collect(Duration::ZERO).unwrap().is_empty());
    assert!(
        scheduler.devices.iter().any(|d| d.kind == DeviceKind::Gpu),
        "the GPU must stay"
    );
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
    assert_eq!(
        cpu.lock().unwrap().submissions.len(),
        cpu_units,
        "no work moves to the CPU"
    );
    let notices = scheduler.take_notices();
    assert!(
        notices.iter().any(|n| n.contains("GPU is back")),
        "{notices:?}"
    );
    assert!(scheduler.devices[0].queued.iter().all(|u| u.retries == 1));
}

#[test]
fn a_gpu_that_never_reopens_falls_back_to_the_cpu() {
    let cfg = submission_config();
    let pop = crate::evolution::create(&cfg).unwrap();
    let (mut scheduler, gpu, cpu) = mixed_scheduler();
    scheduler.devices[0].reopen = Some(Reopen {
        open: Box::new(|| anyhow::bail!("no device")),
        backoff: vec![Duration::ZERO; 3],
    });
    queue_halves(&mut scheduler, &pop, &cfg);
    scheduler.pump().unwrap();
    let (gpu_units, cpu_units) = (
        gpu.lock().unwrap().submissions.len(),
        cpu.lock().unwrap().submissions.len(),
    );
    gpu.lock().unwrap().poll_failure = Some("device lost".into());
    scheduler.collect(Duration::ZERO).unwrap();
    assert!(scheduler.devices.iter().all(|d| d.kind != DeviceKind::Gpu));
    assert_eq!(cpu.lock().unwrap().submissions.len(), cpu_units + gpu_units);
    let notices = scheduler.take_notices();
    assert!(notices.iter().any(|n| n.contains("Continuing on the CPU")));
}

#[test]
fn a_failed_gpu_retries_its_unfinished_units_on_the_cpu() {
    let cfg = submission_config();
    let pop = crate::evolution::create(&cfg).unwrap();
    let (mut scheduler, gpu, cpu) = mixed_scheduler();
    queue_halves(&mut scheduler, &pop, &cfg);
    scheduler.pump().unwrap();
    let (gpu_units, cpu_units, gpu_populations) = {
        let gpu = gpu.lock().unwrap();
        let cpu = cpu.lock().unwrap();
        assert!(!gpu.submissions.is_empty(), "the GPU took no work");
        assert!(!cpu.submissions.is_empty(), "the CPU took no work");
        (
            gpu.submissions.len(),
            cpu.submissions.len(),
            gpu.submissions
                .iter()
                .map(|s| Arc::clone(&s.population))
                .collect::<Vec<_>>(),
        )
    };
    // The GPU dies with every one of its units still unfinished.
    gpu.lock().unwrap().poll_failure = Some("device lost".into());
    assert!(scheduler.collect(Duration::ZERO).unwrap().is_empty());
    assert!(
        scheduler.devices.iter().all(|d| d.kind != DeviceKind::Gpu),
        "the failed GPU must be retired"
    );
    {
        let cpu = cpu.lock().unwrap();
        assert_eq!(cpu.submissions.len(), cpu_units + gpu_units);
        for (retried, original) in cpu.submissions[cpu_units..].iter().zip(&gpu_populations) {
            assert!(
                Arc::ptr_eq(&retried.population, original),
                "the retried unit must keep the exact submitted creatures"
            );
            assert_eq!(retried.config, cfg);
        }
    }
    assert!(
        scheduler.devices[0]
            .queued
            .iter()
            .skip(cpu_units)
            .all(|unit| unit.retries == 1),
        "moved units must carry their retry state"
    );
    complete_submissions(&cpu, 0, 3.0);
    let mut seen: Vec<usize> = scheduler
        .collect(Duration::ZERO)
        .unwrap()
        .into_iter()
        .flat_map(|d| d.members)
        .collect();
    seen.sort_unstable();
    assert_eq!(seen, vec![0, 1, 2, 3], "every creature must finish once");
    assert_eq!(scheduler.in_flight(), 0);
}

#[test]
fn a_reserve_cpu_stands_by_while_the_gpu_works_and_takes_over_after() {
    let cfg = submission_config();
    let pop = crate::evolution::create(&cfg).unwrap();
    let (mut scheduler, gpu, cpu) = mixed_scheduler();
    reserve_cpu_when_gpu_available(&mut scheduler.devices);
    assert!(scheduler.devices[1].reserve);
    queue_halves(&mut scheduler, &pop, &cfg);
    scheduler.pump().unwrap();
    let gpu_units = gpu.lock().unwrap().submissions.len();
    assert!(gpu_units > 0, "the GPU took no work");
    assert!(
        cpu.lock().unwrap().submissions.is_empty(),
        "a reserve CPU must not take standard work beside a healthy GPU"
    );
    gpu.lock().unwrap().poll_failure = Some("device lost".into());
    assert!(scheduler.collect(Duration::ZERO).unwrap().is_empty());
    let seen = drain_all(&mut scheduler, &cpu);
    assert_eq!(seen, vec![0, 1, 2, 3]);
}

#[test]
fn a_failed_gpu_submission_keeps_its_creatures_for_the_cpu() {
    let cfg = submission_config();
    let pop = crate::evolution::create(&cfg).unwrap();
    let (mut scheduler, gpu, cpu) = mixed_scheduler();
    gpu.lock().unwrap().submit_failure = Some("device lost".into());
    queue_halves(&mut scheduler, &pop, &cfg);
    scheduler.pump().unwrap();
    assert_eq!(gpu.lock().unwrap().submissions.len(), 0);
    assert_eq!(
        cpu.lock().unwrap().submissions.len(),
        1,
        "the CPU must take the rejected unit"
    );
    scheduler.collect(Duration::ZERO).unwrap();
    assert!(scheduler.devices.iter().all(|d| d.kind != DeviceKind::Gpu));
    let seen = drain_all(&mut scheduler, &cpu);
    assert_eq!(seen, vec![0, 1, 2, 3], "every creature must finish once");
}

#[test]
fn a_failed_cpu_is_terminal_and_delivers_completed_output_first() {
    let cfg = submission_config();
    let pop = crate::evolution::create(&cfg).unwrap();
    let (mut scheduler, gpu, cpu) = mixed_scheduler();
    queue_halves(&mut scheduler, &pop, &cfg);
    scheduler.pump().unwrap();
    let (ticket, count) = {
        let gpu = gpu.lock().unwrap();
        let submission = &gpu.submissions[0];
        (submission.ticket, submission.population.genomes.len())
    };
    // One GPU unit completes; the CPU dies before returning anything.
    gpu.lock().unwrap().results.push_back(Finished {
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
    cpu.lock().unwrap().poll_failure = Some("cpu lost".into());
    let completed = scheduler.collect(Duration::ZERO).unwrap();
    assert_eq!(
        completed.iter().map(|d| d.members.len()).sum::<usize>(),
        count,
        "completed output must be delivered before the CPU error"
    );
    let submissions = cpu.lock().unwrap().submissions.len();
    for _ in 0..3 {
        let error = scheduler
            .collect(Duration::ZERO)
            .expect_err("the CPU error must persist");
        assert!(error.to_string().contains("cpu lost"));
    }
    assert_eq!(
        cpu.lock().unwrap().submissions.len(),
        submissions,
        "a failed CPU must not be retried"
    );
    assert!(scheduler.in_flight() > 0, "the failure must stay visible");
}

#[test]
fn a_missing_primary_gpu_falls_back_to_the_cpu() {
    // An invalid adapter name cannot open; the CPU keeps the session alive
    // and reports why the GPU was skipped.
    let scheduler = Scheduler::new("definitely not a vulkan adapter").unwrap();
    assert!(scheduler.startup_failure().is_some());
    assert!(
        scheduler.devices.iter().any(|d| d.kind == DeviceKind::Cpu),
        "the fallback must include a CPU engine"
    );
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
    assert_eq!(
        secondary_device_names(Some("primary, radeon, off")),
        vec!["radeon"]
    );
    assert_eq!(secondary_device_names(Some(" , ")), Vec::<&str>::new());
}

#[test]
fn evaluate_returns_the_cpu_engine_scores_in_order() {
    let cfg = Config {
        population: 24,
        duration: 2.0,
        random_seed: false,
        ..Config::default()
    };
    let pop = crate::evolution::create(&cfg).unwrap();
    let mut sched = Scheduler::cpu_only(2).unwrap();
    let indices: Vec<usize> = (0..cfg.population).rev().collect();
    let got = sched.evaluate(&pop, &indices, &cfg).unwrap();
    let expected = crate::cpu_engine::evaluate(&pop, &cfg);
    for (&i, metric) in indices.iter().zip(&got) {
        assert_eq!(metric.fitness.to_bits(), expected[i].fitness.to_bits());
    }
    assert_eq!(sched.in_flight(), 0);
}
