//! What the game is waiting for while it starts: devices opening and GPU
//! kernels compiling. Engines report here from any thread; the window reads
//! it to draw the loading screen. Nothing here affects evaluation.

use std::sync::Mutex;
use std::time::{Duration, Instant};

/// One step the game waits on, such as a kernel compile.
#[derive(Clone, Debug)]
struct Job {
    label: String,
    state: State,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum State {
    Queued,
    Running(Instant),
    /// Finished after this long, at this moment; `true` when it came from
    /// the disk cache.
    Done(Duration, bool, Instant),
}

#[derive(Default)]
struct Registry {
    jobs: Vec<Job>,
    /// When the first job of the current wave was queued or started.
    since: Option<Instant>,
}

static REGISTRY: Mutex<Registry> = Mutex::new(Registry {
    jobs: Vec::new(),
    since: None,
});

fn with<T>(f: impl FnOnce(&mut Registry) -> T) -> T {
    f(&mut REGISTRY.lock().unwrap_or_else(|e| e.into_inner()))
}

fn set(label: &str, state: State) {
    with(|r| {
        if r.since.is_none() || r.jobs.iter().all(|j| matches!(j.state, State::Done(..))) {
            // A new wave starts once everything before it has finished.
            if !matches!(state, State::Done(..)) {
                r.since = Some(Instant::now());
            }
        }
        match r.jobs.iter_mut().find(|j| j.label == label) {
            Some(job) => {
                // A queued job may be picked up again by another path; never
                // move a finished job back to waiting.
                if !(matches!(job.state, State::Done(..)) && state == State::Queued) {
                    job.state = state;
                }
            }
            None => r.jobs.push(Job {
                label: label.to_owned(),
                state,
            }),
        }
    })
}

/// A job that will run later, such as a kernel a background compiler has
/// queued.
pub fn queued(label: &str) {
    set(label, State::Queued);
}

/// A job starting now. Mark it done with `Task::finish`; a task dropped
/// without that (an error) is removed.
pub fn start(label: impl Into<String>) -> Task {
    let label = label.into();
    set(&label, State::Running(Instant::now()));
    Task {
        label,
        started: Instant::now(),
        finished: false,
    }
}

/// Forgets a queued job nobody will run (the engine closed).
pub fn cancel(label: &str) {
    with(|r| {
        r.jobs
            .retain(|j| j.label != label || matches!(j.state, State::Done(..)))
    });
}

/// A running job; see `start`.
pub struct Task {
    label: String,
    started: Instant,
    finished: bool,
}

impl Task {
    /// Marks the job done. `cached` says it was loaded rather than built.
    pub fn finish(mut self, cached: bool) {
        self.finished = true;
        set(
            &self.label,
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

/// What the loading screen shows.
#[derive(Clone, Debug, Default)]
pub struct Progress {
    /// Jobs running now, with how long each has run, oldest first.
    pub running: Vec<(String, Duration)>,
    /// Jobs waiting for a thread.
    pub queued: Vec<String>,
    /// Jobs finished in this wave and before it.
    pub done: usize,
    /// Of `done`, how many came from the disk cache.
    pub cached: usize,
    /// The most recently finished job and how long it took.
    pub last: Option<(String, Duration, bool)>,
    /// Time since the current wave began.
    pub elapsed: Duration,
    /// Seconds the finished kernel compiles took, for estimating the rest.
    pub built_seconds: Vec<f32>,
}

impl Progress {
    /// True while anything is running or waiting.
    pub fn busy(&self) -> bool {
        !self.running.is_empty() || !self.queued.is_empty()
    }
}

/// The current state, for drawing.
pub fn progress() -> Progress {
    with(|r| {
        let now = Instant::now();
        let mut p = Progress {
            elapsed: r.since.map_or(Duration::ZERO, |s| now - s),
            ..Default::default()
        };
        let mut last: Option<(Instant, &Job)> = None;
        let mut later = |at: Instant, job| {
            if last.is_none_or(|(t, _)| at >= t) {
                last = Some((at, job));
            }
        };
        for job in &r.jobs {
            match job.state {
                State::Queued => p.queued.push(job.label.clone()),
                State::Running(at) => p.running.push((job.label.clone(), now - at)),
                State::Done(took, cached, at) => {
                    later(at, job);
                    p.done += 1;
                    if cached {
                        p.cached += 1;
                    } else if job.label.contains("kernel") {
                        p.built_seconds.push(took.as_secs_f32());
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jobs_move_from_queued_to_running_to_done() {
        let label = "test job moves through its states";
        queued(label);
        assert!(progress().queued.iter().any(|l| l == label));
        let task = start(label);
        assert!(progress().running.iter().any(|(l, _)| l == label));
        assert!(!progress().queued.iter().any(|l| l == label));
        task.finish(true);
        let p = progress();
        assert!(!p.running.iter().any(|(l, _)| l == label));
        // Queuing a finished job again does not bring it back.
        queued(label);
        assert!(!progress().queued.iter().any(|l| l == label));
    }

    #[test]
    fn a_failed_task_disappears() {
        let label = "test job that fails";
        drop(start(label));
        let p = progress();
        assert!(!p.running.iter().any(|(l, _)| l == label));
    }
}
