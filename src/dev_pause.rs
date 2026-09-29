//! A pause that developers ask for from outside the game, so a speed
//! measurement gets the GPU and the CPU to itself while the owner's game
//! keeps its run.
//!
//! A tool (`tools/pause-game.sh`) writes the request file `pause` in
//! [`dir`]. The game notices it within about a second, stops handing out new
//! evaluation work, lets the units on the engines finish and be absorbed,
//! closes its GPU engines (freeing their memory) and writes the
//! acknowledgement file `paused` with its pid. It resumes when the request
//! file goes away, when the player presses Resume now, or [`LONGEST`] after
//! the pause began, whichever comes first. After a pause the game runs at
//! least [`REST`] before it honors a new request, and it never honors the
//! same request twice, so a request file left behind cannot pause it again.
//! While it waits out the rest it writes `waiting` with the time it will
//! honor the request.
//!
//! The search is unchanged by a pause: work is held back, never dropped,
//! so the run continues as if the GPU had been slow for a while.

use crate::scheduler::Scheduler;
use std::{
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

/// The longest a pause lasts.
pub const LONGEST: Duration = Duration::from_secs(5 * 60);
/// How long the game runs after a pause before it honors a new request.
pub const REST: Duration = Duration::from_secs(2 * 60);
/// How often the game looks for the request file.
const POLL: Duration = Duration::from_millis(250);

/// Where the request and acknowledgement files live:
/// `$XDG_RUNTIME_DIR/evolution-simulator`, or
/// `<temp>/evolution-simulator-<uid>` without a runtime directory.
pub fn dir() -> PathBuf {
    match std::env::var_os("XDG_RUNTIME_DIR").filter(|d| !d.is_empty()) {
        Some(runtime) => PathBuf::from(runtime).join("evolution-simulator"),
        None => std::env::temp_dir().join(format!("evolution-simulator-{}", user_id())),
    }
}

#[cfg(unix)]
fn user_id() -> u32 {
    // SAFETY: getuid has no preconditions and cannot fail.
    unsafe { libc::getuid() }
}

#[cfg(not(unix))]
fn user_id() -> u32 {
    0
}

/// What the UI shows while a pause is on.
#[derive(Clone, Copy, Debug)]
pub struct View {
    /// When the pause ends at the latest.
    pub ends_at: Instant,
    /// The engines are closed; false while the last units finish.
    pub closed: bool,
}

/// State the worker shares with the UI.
#[derive(Default)]
pub struct Shared {
    view: Mutex<Option<View>>,
    resume: AtomicBool,
}

impl Shared {
    /// The pause on now, if any.
    pub fn view(&self) -> Option<View> {
        *self.view.lock().unwrap_or_else(|e| e.into_inner())
    }
    /// The player's Resume now button.
    pub fn resume_now(&self) {
        self.resume.store(true, Ordering::Relaxed);
    }
    fn set(&self, view: Option<View>) {
        *self.view.lock().unwrap_or_else(|e| e.into_inner()) = view;
    }
}

/// Decides when the game is paused, from the request it sees and the clock.
#[derive(Debug)]
pub struct Controller {
    longest: Duration,
    rest: Duration,
    started: Option<Instant>,
    ended: Option<Instant>,
    /// The last request honored; the same request is not honored again.
    served: Option<String>,
}

impl Controller {
    pub fn new(longest: Duration, rest: Duration) -> Self {
        Self {
            longest,
            rest,
            started: None,
            ended: None,
            served: None,
        }
    }

    /// Whether the game should be paused at `now`. `request` identifies the
    /// request file present (None when there is none); `resume` is the
    /// player's Resume now.
    pub fn update(&mut self, now: Instant, request: Option<&str>, resume: bool) -> bool {
        if let Some(started) = self.started {
            if resume || request.is_none() || now.duration_since(started) >= self.longest {
                self.started = None;
                self.ended = Some(now);
                return false;
            }
            return true;
        }
        match request {
            Some(id) if self.served.as_deref() != Some(id) && self.waiting(now).is_none() => {
                self.started = Some(now);
                self.served = Some(id.to_owned());
                true
            }
            _ => false,
        }
    }

    /// While the game rests after a pause, how long until it honors a
    /// request again.
    pub fn waiting(&self, now: Instant) -> Option<Duration> {
        let ended = self.ended?;
        let rested = now.duration_since(ended);
        (self.started.is_none() && rested < self.rest).then(|| self.rest - rested)
    }

    /// Whether `request` is new and waits for the rest to end.
    fn deferred(&self, now: Instant, request: Option<&str>) -> Option<Duration> {
        let request = request?;
        if self.served.as_deref() == Some(request) {
            return None;
        }
        self.waiting(now)
    }

    /// When the current pause ends at the latest.
    pub fn ends_at(&self) -> Option<Instant> {
        self.started.map(|started| started + self.longest)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Phase {
    Running,
    /// No new work goes out; the units on the engines finish.
    Holding,
    /// The engines are closed.
    Closed,
}

/// The worker's side of the pause: watches the request file, suspends and
/// resumes the scheduler, and writes the acknowledgement.
pub struct DevPause {
    dir: PathBuf,
    shared: Arc<Shared>,
    controller: Controller,
    phase: Phase,
    last_poll: Option<Instant>,
    request: Option<String>,
    /// The `waiting` file is written.
    waiting_written: bool,
}

impl DevPause {
    pub fn new(dir: PathBuf, shared: Arc<Shared>) -> Self {
        Self::with_limits(dir, shared, LONGEST, REST)
    }

    pub fn with_limits(
        dir: PathBuf,
        shared: Arc<Shared>,
        longest: Duration,
        rest: Duration,
    ) -> Self {
        Self {
            dir,
            shared,
            controller: Controller::new(longest, rest),
            phase: Phase::Running,
            last_poll: None,
            request: None,
            waiting_written: false,
        }
    }

    /// True while the scheduler is suspended.
    pub fn holding(&self) -> bool {
        self.phase != Phase::Running
    }

    /// True once the engines are closed and nothing is left to collect: the
    /// worker may sleep between commands.
    pub fn idle(&self) -> bool {
        self.phase == Phase::Closed
    }

    /// Looks at the request and the clock and moves the pause along.
    /// Returns a message for the event log when the pause starts or ends.
    pub fn tick(&mut self, sched: Option<&mut Scheduler>) -> Option<String> {
        let now = Instant::now();
        if self
            .last_poll
            .is_none_or(|at| now.duration_since(at) >= POLL)
        {
            self.last_poll = Some(now);
            self.request = read_request(&self.dir);
        }
        let resume = self.shared.resume.swap(false, Ordering::Relaxed);
        let pause = self.controller.update(now, self.request.as_deref(), resume);
        self.write_waiting(self.controller.deferred(now, self.request.as_deref()));
        let mut message = None;
        match (pause, self.phase, sched) {
            (true, Phase::Running, sched) => {
                if let Some(sched) = sched {
                    sched.suspend();
                }
                self.phase = Phase::Holding;
                self.publish();
                message = Some("Paused for a developer measurement.".to_owned());
            }
            (true, Phase::Holding, sched) => {
                if sched.is_none_or(|sched| sched.close_idle_engines()) {
                    self.phase = Phase::Closed;
                    self.publish();
                    self.write_ack();
                }
            }
            (true, Phase::Closed, _) => {}
            (false, Phase::Running, _) => {}
            (false, _, sched) => {
                let why = if resume {
                    "the player resumed"
                } else if self.request.is_none() {
                    "the measurement finished"
                } else {
                    "the pause reached its limit"
                };
                if let Some(sched) = sched {
                    sched.resume();
                }
                self.phase = Phase::Running;
                self.publish();
                let _ = std::fs::remove_file(self.dir.join("paused"));
                message = Some(format!("Resumed after the developer pause: {why}."));
            }
        }
        message
    }

    fn publish(&self) {
        let view = (self.phase != Phase::Running)
            .then(|| self.controller.ends_at())
            .flatten()
            .map(|ends_at| View {
                ends_at,
                closed: self.phase == Phase::Closed,
            });
        self.shared.set(view);
    }

    fn write_ack(&self) {
        let now = unix_seconds(SystemTime::now());
        let left = self.controller.ends_at().map_or(0, |end| {
            end.saturating_duration_since(Instant::now()).as_secs()
        });
        write_file(
            &self.dir.join("paused"),
            &format!(
                "pid={}\nsince={now}\nresumes_by={}\n",
                std::process::id(),
                now + left
            ),
        );
    }

    fn write_waiting(&mut self, wait: Option<Duration>) {
        match wait {
            Some(wait) if !self.waiting_written => {
                let at = unix_seconds(SystemTime::now()) + wait.as_secs() + 1;
                write_file(
                    &self.dir.join("waiting"),
                    &format!("pid={}\nhonors_at={at}\n", std::process::id()),
                );
                self.waiting_written = true;
            }
            None if self.waiting_written => {
                let _ = std::fs::remove_file(self.dir.join("waiting"));
                self.waiting_written = false;
            }
            _ => {}
        }
    }
}

impl Drop for DevPause {
    fn drop(&mut self) {
        // Leave no acknowledgement behind for a game that has quit.
        if self.phase == Phase::Closed {
            let _ = std::fs::remove_file(self.dir.join("paused"));
        }
        if self.waiting_written {
            let _ = std::fs::remove_file(self.dir.join("waiting"));
        }
    }
}

/// The request's identity: its modification time and contents, or None when
/// there is no request.
fn read_request(dir: &Path) -> Option<String> {
    let path = dir.join("pause");
    let modified = std::fs::metadata(&path).ok()?.modified().ok()?;
    let contents = std::fs::read_to_string(&path).unwrap_or_default();
    let stamp = modified
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    Some(format!("{stamp}:{contents}"))
}

fn unix_seconds(at: SystemTime) -> u64 {
    at.duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Writes `contents` through a temporary file and a rename, so a reader never
/// sees half a file.
fn write_file(path: &Path, contents: &str) {
    let Some(dir) = path.parent() else {
        return;
    };
    let _ = std::fs::create_dir_all(dir);
    let temporary = path.with_extension(format!("tmp{}", std::process::id()));
    if std::fs::write(&temporary, contents).is_ok() && std::fs::rename(&temporary, path).is_err() {
        let _ = std::fs::remove_file(&temporary);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LONG: Duration = Duration::from_millis(300);
    const REST_TEST: Duration = Duration::from_millis(200);

    #[test]
    fn a_pause_ends_when_the_request_goes_away() {
        let mut c = Controller::new(LONG, REST_TEST);
        let t = Instant::now();
        assert!(!c.update(t, None, false));
        assert!(c.update(t, Some("a"), false));
        assert!(c.update(t + Duration::from_millis(100), Some("a"), false));
        assert!(!c.update(t + Duration::from_millis(150), None, false));
    }

    #[test]
    fn a_pause_ends_at_its_limit_even_if_the_request_stays() {
        let mut c = Controller::new(LONG, REST_TEST);
        let t = Instant::now();
        assert!(c.update(t, Some("a"), false));
        assert_eq!(c.ends_at(), Some(t + LONG));
        assert!(c.update(t + LONG - Duration::from_millis(1), Some("a"), false));
        assert!(!c.update(t + LONG, Some("a"), false));
        // The same request is never honored again, even after the rest.
        assert!(!c.update(t + LONG + REST_TEST * 3, Some("a"), false));
    }

    #[test]
    fn a_new_request_waits_for_the_rest_after_a_pause() {
        let mut c = Controller::new(LONG, REST_TEST);
        let t = Instant::now();
        assert!(c.update(t, Some("a"), false));
        let end = t + Duration::from_millis(50);
        assert!(!c.update(end, None, false));
        let soon = end + Duration::from_millis(100);
        assert!(!c.update(soon, Some("b"), false));
        assert_eq!(
            c.deferred(soon, Some("b")),
            Some(Duration::from_millis(100))
        );
        assert!(c.update(end + REST_TEST, Some("b"), false));
    }

    #[test]
    fn resume_now_ends_the_pause_and_serves_the_request() {
        let mut c = Controller::new(LONG, REST_TEST);
        let t = Instant::now();
        assert!(c.update(t, Some("a"), false));
        assert!(!c.update(t + Duration::from_millis(10), Some("a"), true));
        assert!(!c.update(t + REST_TEST * 2, Some("a"), false));
    }

    #[test]
    fn files_drive_a_pause_and_its_acknowledgement() {
        let dir = std::env::temp_dir().join(format!(
            "evolution-dev-pause-test-{}-{}",
            std::process::id(),
            unix_seconds(SystemTime::now())
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let shared = Arc::new(Shared::default());
        let mut pause = DevPause::with_limits(dir.clone(), shared.clone(), LONG, REST_TEST);
        assert!(pause.tick(None).is_none());
        std::fs::write(dir.join("pause"), "test").unwrap();
        pause.last_poll = None;
        assert!(pause.tick(None).is_some());
        assert!(pause.holding() && !pause.idle());
        pause.tick(None);
        assert!(pause.idle());
        assert!(dir.join("paused").exists());
        assert!(shared.view().is_some_and(|v| v.closed));
        std::fs::remove_file(dir.join("pause")).unwrap();
        pause.last_poll = None;
        let message = pause.tick(None).unwrap();
        assert!(message.contains("measurement finished"), "{message}");
        assert!(!pause.holding());
        assert!(!dir.join("paused").exists());
        assert!(shared.view().is_none());
        // A new request during the rest is announced, then honored.
        std::fs::write(dir.join("pause"), "second").unwrap();
        pause.last_poll = None;
        assert!(pause.tick(None).is_none());
        assert!(dir.join("waiting").exists());
        std::thread::sleep(REST_TEST);
        assert!(pause.tick(None).is_some());
        assert!(!dir.join("waiting").exists());
        // The limit ends it with the request still there.
        pause.tick(None);
        std::thread::sleep(LONG);
        let message = pause.tick(None).unwrap();
        assert!(message.contains("limit"), "{message}");
        drop(pause);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
