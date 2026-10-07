//! What the game is waiting for while it starts: devices opening and GPU
//! kernels compiling. Engines report each job here from any thread, and the
//! window reads the registry to draw the loading screen (`ui/loading.rs`).
//! A job is queued, then running, then done, and it belongs to one `Group`.
//! Nothing here affects evaluation.

use std::sync::{
    Mutex,
    atomic::{AtomicBool, Ordering},
};
use std::time::{Duration, Instant};

/// Who waits for a job.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    /// What the loading screen waits for when the game starts: the devices
    /// and the kernels of the worlds the game begins with.
    Startup,
    /// A kernel that evolution waits for now, such as a new world after a
    /// button press.
    Needed,
    /// A kernel of a world one effect level away from the current one. It is
    /// compiled at the lowest priority, for a later button press. Nobody waits
    /// for it, and the window does not show it.
    Idle,
}

/// Where a job is in its life.
#[derive(Clone, Copy, Debug, PartialEq)]
enum State {
    /// Waiting for a thread.
    Queued,
    /// Running since this moment.
    Running(Instant),
    /// Finished. The fields are how long it took, whether it came from the
    /// disk cache and not from a compiler, and the moment it finished.
    Done(Duration, bool, Instant),
}

/// A tracked job: a device opening or a kernel build.
#[derive(Clone, Debug)]
struct Job {
    /// Names the job. A report with a label that is already known is about
    /// that job.
    label: String,
    group: Group,
    state: State,
}

struct Registry {
    /// Every job reported so far, in the order of its first report. Finished
    /// jobs stay, so the screen can count them.
    jobs: Vec<Job>,
    /// When the first job was queued or started.
    since: Option<Instant>,
}

static REGISTRY: Mutex<Registry> = Mutex::new(Registry {
    jobs: Vec::new(),
    since: None,
});

/// While true, `queued` puts every job that is not `Idle` in `Group::Startup`.
static STARTUP: AtomicBool = AtomicBool::new(false);

/// Runs `f` on the registry with its lock held. A lock poisoned by a panic on
/// another thread is used as it is.
fn with<T>(f: impl FnOnce(&mut Registry) -> T) -> T {
    f(&mut REGISTRY.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Records `state` for the job named `label`, and adds the job if it is new.
/// A new job with no `group` joins `Group::Needed`. A known job keeps its
/// group, except that an `Idle` one moves to a `group` that is not `Idle`.
fn set(label: &str, group: Option<Group>, state: State) {
    with(|r| {
        if r.since.is_none() && !matches!(state, State::Done(..)) {
            r.since = Some(Instant::now());
        }
        match r.jobs.iter_mut().find(|j| j.label == label) {
            Some(job) => {
                // Never move a finished job back to waiting, and keep the
                // group a job was queued in.
                if !(matches!(job.state, State::Done(..)) && state == State::Queued) {
                    job.state = state;
                }
                // A kernel that evolution now waits for is no longer idle.
                if let Some(group) = group
                    && job.group == Group::Idle
                    && group != Group::Idle
                {
                    job.group = group;
                }
            }
            None => r.jobs.push(Job {
                label: label.to_owned(),
                group: group.unwrap_or(Group::Needed),
                state,
            }),
        }
    })
}

/// From now until `end_startup`, `queued` puts every job that is not `Idle` in
/// `Group::Startup`, which the loading screen waits for. The CUDA engine calls
/// it around the prefetch of the worlds the game begins with.
pub fn begin_startup() {
    STARTUP.store(true, Ordering::Relaxed);
}

/// Ends the span that `begin_startup` began.
pub fn end_startup() {
    STARTUP.store(false, Ordering::Relaxed);
}

/// Reports a job that will run later, such as a kernel a background compiler
/// has queued. While `begin_startup` is in force it joins `Group::Startup`
/// unless `group` is `Idle`. A job that is already finished stays finished.
pub fn queued(label: &str, group: Group) {
    let group = if STARTUP.load(Ordering::Relaxed) && group != Group::Idle {
        Group::Startup
    } else {
        group
    };
    set(label, Some(group), State::Queued);
}

/// Reports a job that starts now. Mark it done with `Task::finish`. A task
/// dropped without that, as after an error, removes the job. A job that was
/// queued keeps its group, and a job that was not joins `Group::Needed`.
pub fn start(label: impl Into<String>) -> Task {
    start_in(label, None)
}

/// `start` with a group. A new job joins `group`. A job queued as `Idle` moves
/// to `group` unless that is `Idle` too. With `None` it is the same as `start`.
pub fn start_in(label: impl Into<String>, group: Option<Group>) -> Task {
    let label = label.into();
    set(&label, group, State::Running(Instant::now()));
    Task {
        label,
        started: Instant::now(),
        finished: false,
    }
}

/// Forgets the job `label` unless it is finished. The CUDA engine calls it for
/// each kernel still queued when the engine closes, because nobody will run it.
pub fn cancel(label: &str) {
    with(|r| {
        r.jobs
            .retain(|j| j.label != label || matches!(j.state, State::Done(..)))
    });
}

/// A running job; see `start`. `finish` marks it done, and dropping it
/// unfinished removes it.
pub struct Task {
    label: String,
    started: Instant,
    finished: bool,
}

impl Task {
    /// Marks the job done. `cached` says it came from the disk cache and not
    /// from a compiler.
    pub fn finish(mut self, cached: bool) {
        self.finished = true;
        set(
            &self.label,
            None,
            State::Done(self.started.elapsed(), cached, Instant::now()),
        );
    }
}

impl Drop for Task {
    fn drop(&mut self) {
        if !self.finished {
            let label = std::mem::take(&mut self.label);
            with(|r| r.jobs.retain(|j| j.label != label));
        }
    }
}

/// What the loading screen or the corner note shows for one group of jobs.
#[derive(Clone, Debug, Default)]
pub struct Progress {
    /// Jobs running now, with how long each has run, longest first.
    pub running: Vec<(String, Duration)>,
    /// Jobs waiting for a thread.
    pub queued: usize,
    /// Jobs finished.
    pub done: usize,
    /// Of `done`, how many came from the disk cache.
    pub cached: usize,
    /// The job that finished last: its label, how long it took and whether it
    /// came from the disk cache.
    pub last: Option<(String, Duration, bool)>,
    /// Time since the first job of any group was queued or started.
    pub elapsed: Duration,
    /// Seconds each finished job took, apart from the jobs read from the disk
    /// cache.
    pub built_seconds: Vec<f32>,
}

impl Progress {
    /// Jobs known: done, running and waiting.
    pub fn total(&self) -> usize {
        self.done + self.running.len() + self.queued
    }

    /// True while a job of the group is running or waiting.
    pub fn busy(&self) -> bool {
        !self.running.is_empty() || self.queued > 0
    }

    /// Seconds until the rest finish, estimated from the mean of
    /// `built_seconds`. A running job needs the mean less the time it has run,
    /// but not less than half a second. A waiting job needs the mean. The sum
    /// is spread over the jobs that run at once. It is `None` before a build
    /// has finished and when no job is left.
    pub fn seconds_left(&self) -> Option<f32> {
        if self.built_seconds.is_empty() || !self.busy() {
            return None;
        }
        let mean = self.built_seconds.iter().sum::<f32>() / self.built_seconds.len() as f32;
        let running: f32 = self
            .running
            .iter()
            .map(|(_, t)| (mean - t.as_secs_f32()).max(0.5))
            .sum();
        let waiting = self.queued as f32 * mean;
        Some((running + waiting) / self.running.len().max(1) as f32)
    }
}

/// Counts the jobs of `group` by state and lists the running ones, for
/// drawing.
pub fn progress(group: Group) -> Progress {
    with(|r| {
        let now = Instant::now();
        let mut p = Progress {
            elapsed: r.since.map_or(Duration::ZERO, |s| now - s),
            ..Default::default()
        };
        let mut last: Option<(Instant, &Job)> = None;
        for job in r.jobs.iter().filter(|j| j.group == group) {
            match job.state {
                State::Queued => p.queued += 1,
                State::Running(at) => p.running.push((job.label.clone(), now - at)),
                State::Done(took, cached, at) => {
                    p.done += 1;
                    if cached {
                        p.cached += 1;
                    } else {
                        p.built_seconds.push(took.as_secs_f32());
                    }
                    if last.is_none_or(|(t, _)| at >= t) {
                        last = Some((at, job));
                    }
                }
            }
        }
        p.running.sort_by_key(|r| std::cmp::Reverse(r.1));
        if let Some((_, job)) = last
            && let State::Done(took, cached, _) = job.state
        {
            p.last = Some((job.label.clone(), took, cached));
        }
        p
    })
}

/// Replaces the registry with made-up `Startup` jobs for a screenshot run
/// (`EVOLUTION_SMOKE_LOADING`): some done, a few running, many queued. The
/// game seems to have begun 41 s ago.
pub fn demo() {
    with(|r| {
        r.jobs.clear();
        r.since = Some(Instant::now() - Duration::from_secs(41));
        let now = Instant::now();
        let worlds = [
            "calm",
            "Mud",
            "Wind + Slope",
            "Water + Ice patches",
            "Hurdles + Gaps + Wind",
            "Brambles",
            "Quake + Mud",
            "Slope + Air drag",
        ];
        for (w, world) in worlds.iter().enumerate() {
            // Four jobs a world, as when a kernel came in four lane classes.
            // The label no longer names the class, so the loop value only
            // picks the place of the job among the states.
            for class in [4u32, 8, 16, 32] {
                let n = w * 4 + class.ilog2() as usize - 2;
                let state = if n < 13 {
                    State::Done(
                        Duration::from_secs_f32(9.0 + (n % 5) as f32),
                        n.is_multiple_of(7),
                        now - Duration::from_secs((14 - n) as u64),
                    )
                } else if n < 16 {
                    State::Running(now - Duration::from_secs_f32(3.0 + n as f32 % 5.0))
                } else {
                    State::Queued
                };
                r.jobs.push(Job {
                    label: format!("scoring kernel · {world}"),
                    group: Group::Startup,
                    state,
                });
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jobs_move_from_queued_to_running_to_done() {
        let label = "test job moves through its states";
        queued(label, Group::Needed);
        let before = progress(Group::Needed);
        assert!(before.queued >= 1);
        let task = start(label);
        assert!(
            progress(Group::Needed)
                .running
                .iter()
                .any(|(l, _)| l == label)
        );
        task.finish(true);
        let after = progress(Group::Needed);
        assert!(!after.running.iter().any(|(l, _)| l == label));
        let done = after.done;
        // Queuing a finished job again does not bring it back.
        queued(label, Group::Needed);
        assert_eq!(progress(Group::Needed).done, done);
    }

    #[test]
    fn a_failed_task_disappears() {
        let label = "test job that fails";
        drop(start(label));
        assert!(
            !progress(Group::Needed)
                .running
                .iter()
                .any(|(l, _)| l == label)
        );
    }

    #[test]
    fn a_kernel_that_is_wanted_leaves_the_idle_group() {
        let label = "test idle kernel that is then wanted";
        queued(label, Group::Idle);
        let idle = progress(Group::Idle).queued;
        queued(label, Group::Needed);
        assert_eq!(progress(Group::Idle).queued, idle - 1);
    }

    #[test]
    fn seconds_left_spreads_the_rest_over_the_running_jobs() {
        let p = Progress {
            running: vec![("a".into(), Duration::from_secs(2))],
            queued: 3,
            built_seconds: vec![10.0],
            ..Default::default()
        };
        // 8 s left on the running one and 3 x 10 s waiting, on one thread.
        assert!((p.seconds_left().unwrap() - 38.0).abs() < 0.01);
        assert_eq!(Progress::default().seconds_left(), None);
    }
}
