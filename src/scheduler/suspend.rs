//! This module suspends the scheduler for `crate::dev_pause`, which holds
//! evaluation back for a developer measurement. A suspended scheduler submits
//! no new unit, and units already on an engine finish and are collected as
//! usual. When no engine holds a unit, `close_idle_engines` frees the GPU
//! memory by closing every engine that can be reopened, and `resume` opens
//! them again. The ring absorbs blocks in a fixed order whatever the timing,
//! so the search sees only a slow GPU and a paused run gives the same results.

use super::*;

/// The suspension state of a `Scheduler`: the flag that holds work back and
/// the engines closed so far.
#[derive(Default)]
pub(super) struct Suspension {
    /// True from `suspend` until `resume`. While it is true `pump` hands out
    /// no work.
    active: bool,
    /// Indices into `Scheduler::devices` of the engines that
    /// `close_idle_engines` closed. `resume` opens them again and empties the
    /// list.
    closed: Vec<usize>,
}

impl Scheduler {
    /// Stops handing new work to the engines. Units already on an engine
    /// finish, and `collect` still returns them.
    pub fn suspend(&mut self) {
        self.suspension.active = true;
    }

    /// True from `suspend` until `resume`.
    pub fn suspended(&self) -> bool {
        self.suspension.active
    }

    /// True when `pump` may hand work to the engines. That is when the
    /// scheduler is not suspended.
    pub(super) fn may_submit(&self) -> bool {
        !self.suspension.active
    }

    /// Closes every GPU engine that can be reopened, which frees its memory.
    /// It does this only while the scheduler is suspended and no engine holds
    /// a unit. It returns true in that case. Otherwise it closes nothing and
    /// returns false. An engine with no reopen hook, or with a failure, is
    /// left as it is. A repeated call closes nothing more.
    pub fn close_idle_engines(&mut self) -> bool {
        if !self.suspension.active {
            return false;
        }
        if self.devices.iter().any(|d| !d.queued.is_empty()) {
            return false;
        }
        for (index, device) in self.devices.iter_mut().enumerate() {
            if device.reopen.is_some()
                && device.failure.is_none()
                && !self.suspension.closed.contains(&index)
            {
                // Dropping the engine joins its thread, which releases the
                // device and its memory.
                device.engine = Box::new(RetiredEngine);
                self.suspension.closed.push(index);
            }
        }
        true
    }

    /// Ends a suspension: opens the closed GPUs again and lets `pump` hand out
    /// work. A GPU that does not open is marked failed, so the next `collect`
    /// runs the usual recovery and tries again.
    pub fn resume(&mut self) {
        self.suspension.active = false;
        for index in std::mem::take(&mut self.suspension.closed) {
            let Some(device) = self.devices.get_mut(index) else {
                continue;
            };
            let Some(reopen) = device.reopen.as_mut() else {
                continue;
            };
            match (reopen.open)() {
                Ok(engine) => device.engine = engine,
                Err(error) => {
                    device.failure = Some(format!("The GPU did not open again: {error:#}"));
                }
            }
        }
    }
}
