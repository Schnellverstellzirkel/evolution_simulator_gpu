//! Where the game's threads run, so the window keeps 60 FPS while the
//! host breeds on every core.
//!
//! The process's CPUs are split once, in order: the first hardware thread
//! runs the worker (commands and snapshots), the second runs the GPU engine
//! thread, and the rest run the Rayon pool and the worker's helpers. The UI
//! thread is not pinned: it asks the scheduler for a 1 ms slice instead, so
//! a frame preempts breeding threads when it wakes. Pool threads are
//! SCHED_BATCH at nice 10, so the scheduler never treats them as
//! interactive. With fewer than four CPUs nothing is pinned.

use std::sync::OnceLock;

struct Layout {
    worker: Option<usize>,
    engine: Option<usize>,
    pool: Vec<usize>,
}

fn layout() -> &'static Layout {
    static LAYOUT: OnceLock<Layout> = OnceLock::new();
    LAYOUT.get_or_init(|| {
        let cpus = allowed_cpus();
        if cpus.len() < 4 {
            return Layout {
                worker: None,
                engine: None,
                pool: Vec::new(),
            };
        }
        Layout {
            worker: Some(cpus[0]),
            engine: Some(cpus[1]),
            pool: cpus[2..].to_vec(),
        }
    })
}

/// Reads the process's CPUs before any thread is pinned. Call it first
/// thing in `main`.
pub fn init() {
    layout();
}

/// Rayon pool size for the game: every CPU but the worker's and the
/// engine's. `RAYON_NUM_THREADS` may lower it.
pub fn pool_threads() -> usize {
    let cpus = std::thread::available_parallelism().map_or(2, usize::from);
    let default = if cpus >= 4 { cpus - 2 } else { cpus.max(1) };
    std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&n| n > 0)
        .map_or(default, |n| n.min(default))
}

/// Rayon start handler: pool CPUs, SCHED_BATCH, nice 10.
pub fn pool_thread_start() {
    pin_pool();
    #[cfg(target_os = "linux")]
    unsafe {
        let param = libc::sched_param { sched_priority: 0 };
        libc::sched_setscheduler(0, libc::SCHED_BATCH, &param);
    }
    crate::engine::lower_thread_priority();
}

/// Runs the calling thread on the pool's CPUs (helpers of the worker, which
/// would otherwise inherit its CPU).
pub fn pin_pool() {
    let pool = &layout().pool;
    if !pool.is_empty() {
        set_affinity(pool);
    }
}

/// Runs the calling thread on the worker's CPU.
pub fn pin_worker() {
    if let Some(cpu) = layout().worker {
        set_affinity(&[cpu]);
    }
}

/// Runs the calling thread on the GPU engine thread's CPU.
pub fn pin_engine() {
    if let Some(cpu) = layout().engine {
        set_affinity(&[cpu]);
    }
}

/// Asks EEVDF for a 1 ms slice for the calling thread (Linux 6.12 and
/// later honor `sched_runtime` for normal tasks). A short-slice task
/// preempts long-slice tasks when it wakes. Returns whether it was set.
pub fn short_slice() -> bool {
    #[cfg(target_os = "linux")]
    {
        #[repr(C)]
        struct SchedAttr {
            size: u32,
            policy: u32,
            flags: u64,
            nice: i32,
            priority: u32,
            runtime: u64,
            deadline: u64,
            period: u64,
        }
        let attr = SchedAttr {
            size: std::mem::size_of::<SchedAttr>() as u32,
            policy: libc::SCHED_OTHER as u32,
            flags: 0,
            nice: unsafe { libc::getpriority(libc::PRIO_PROCESS, 0) },
            priority: 0,
            runtime: 1_000_000,
            deadline: 0,
            period: 0,
        };
        let result =
            unsafe { libc::syscall(libc::SYS_sched_setattr, 0, &attr as *const SchedAttr, 0u32) };
        result == 0
    }
    #[cfg(not(target_os = "linux"))]
    false
}

/// Major page faults of the calling thread so far (Linux), for the
/// benchmark report.
pub fn major_faults() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        let stat = std::fs::read_to_string("/proc/thread-self/stat").ok()?;
        // Fields after the command name, which is in parentheses and may
        // hold spaces: state is field 3, majflt field 12.
        let rest = &stat[stat.rfind(')')? + 2..];
        rest.split_whitespace().nth(9)?.parse().ok()
    }
    #[cfg(not(target_os = "linux"))]
    None
}

/// A thread on the pool's CPUs that runs one borrowed job at a time for
/// the worker. A thread per job cost a thread exit each time, and while the
/// pool bred an exit waited up to 140 ms for the process's memory map
/// lock, with the worker stuck in `join`.
pub struct Helper {
    jobs: Option<std::sync::mpsc::Sender<Job>>,
    done: std::sync::mpsc::Receiver<std::thread::Result<()>>,
    thread: Option<std::thread::JoinHandle<()>>,
}

/// A borrowed job, its lifetime erased. `Helper::run` does not return
/// before the helper has finished with it.
struct Job(*mut (dyn FnMut() + Send + 'static));
// The pointee is `Send` and outlives its use (`Helper::run`).
unsafe impl Send for Job {}

impl Helper {
    pub fn new(name: &str) -> Self {
        let (jobs, job_rx) = std::sync::mpsc::channel::<Job>();
        let (done_tx, done) = std::sync::mpsc::channel();
        let thread = std::thread::Builder::new()
            .name(name.into())
            .spawn(move || {
                pin_pool();
                for job in job_rx {
                    // Safety: `run` keeps the job alive and untouched until
                    // this reply arrives.
                    let job = unsafe { &mut *job.0 };
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
                    if done_tx.send(result).is_err() {
                        break;
                    }
                }
            })
            .expect("Start the helper thread");
        Self {
            jobs: Some(jobs),
            done,
            thread: Some(thread),
        }
    }

    /// Runs `job` on the helper and calls `idle` on this thread until it
    /// has finished; `idle` should block briefly (it is called in a loop).
    /// A panic in `job` resumes here.
    pub fn run<'a, R: Send>(
        &self,
        job: impl FnOnce() -> R + Send + 'a,
        mut idle: impl FnMut(),
    ) -> R {
        let mut job = Some(job);
        let mut result = None;
        let mut call = || result = Some((job.take().expect("a job runs once"))());
        let erased: *mut (dyn FnMut() + Send + '_) = &mut call;
        // Safety: the guard below waits for the helper's reply before
        // `call` goes out of scope, even if `idle` panics.
        let erased: *mut (dyn FnMut() + Send + 'static) = unsafe { std::mem::transmute(erased) };
        /// Waits for the helper's reply when dropped, so no unwind leaves
        /// the helper with a dangling job.
        struct Wait<'h> {
            helper: &'h Helper,
            reply: Option<std::thread::Result<()>>,
        }
        impl Wait<'_> {
            fn finished(&mut self) -> bool {
                if self.reply.is_none() {
                    match self.helper.done.try_recv() {
                        Ok(reply) => self.reply = Some(reply),
                        Err(std::sync::mpsc::TryRecvError::Empty) => {}
                        // The helper catches panics, so it only stops with
                        // the process; a borrowed job must not outlive this.
                        Err(std::sync::mpsc::TryRecvError::Disconnected) => std::process::abort(),
                    }
                }
                self.reply.is_some()
            }
        }
        impl Drop for Wait<'_> {
            fn drop(&mut self) {
                while !self.finished() {
                    std::thread::yield_now();
                }
            }
        }
        let mut wait = Wait {
            helper: self,
            reply: None,
        };
        self.jobs
            .as_ref()
            .expect("helper running")
            .send(Job(erased))
            .unwrap_or_else(|_| std::process::abort());
        while !wait.finished() {
            idle();
        }
        let reply = wait.reply.as_mut().map(|r| std::mem::replace(r, Ok(())));
        drop(wait);
        let _ = &mut call;
        if let Some(Err(panic)) = reply {
            std::panic::resume_unwind(panic);
        }
        result.expect("the job ran")
    }
}

impl Drop for Helper {
    fn drop(&mut self) {
        self.jobs = None;
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(target_os = "linux")]
fn allowed_cpus() -> Vec<usize> {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        if libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) != 0 {
            return Vec::new();
        }
        (0..libc::CPU_SETSIZE as usize)
            .filter(|&cpu| libc::CPU_ISSET(cpu, &set))
            .collect()
    }
}

#[cfg(not(target_os = "linux"))]
fn allowed_cpus() -> Vec<usize> {
    Vec::new()
}

#[cfg(target_os = "linux")]
fn set_affinity(cpus: &[usize]) {
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        for &cpu in cpus {
            libc::CPU_SET(cpu, &mut set);
        }
        libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
    }
}

#[cfg(not(target_os = "linux"))]
fn set_affinity(_: &[usize]) {}

#[cfg(test)]
mod tests {
    #[test]
    fn pool_leaves_two_cpus() {
        let cpus = std::thread::available_parallelism().map_or(2, usize::from);
        let pool = super::pool_threads();
        assert!(pool >= 1);
        if cpus >= 4 && std::env::var_os("RAYON_NUM_THREADS").is_none() {
            assert_eq!(pool, cpus - 2);
        }
    }

    #[test]
    fn helper_runs_borrowed_jobs() {
        let helper = super::Helper::new("test-helper");
        let mut data = vec![1, 2, 3];
        let mut idles = 0;
        for round in 0..3 {
            let sum = helper.run(
                || {
                    data.push(round);
                    data.iter().sum::<i32>()
                },
                || {
                    idles += 1;
                    std::thread::sleep(std::time::Duration::from_micros(50));
                },
            );
            assert_eq!(sum, data.iter().sum::<i32>());
        }
        assert_eq!(data, [1, 2, 3, 0, 1, 2]);
        let _ = idles;
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            helper.run(|| panic!("job panics"), || {})
        }));
        assert!(caught.is_err());
        assert_eq!(helper.run(|| 7, || {}), 7);
    }

    #[test]
    fn major_faults_are_readable() {
        #[cfg(target_os = "linux")]
        assert!(super::major_faults().is_some());
    }
}
