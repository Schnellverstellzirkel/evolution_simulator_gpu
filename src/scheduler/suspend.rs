//! Suspending evaluation for a developer measurement (`crate::dev_pause`).
//!
//! A suspended scheduler submits nothing new to any engine. Units already
//! on an engine finish and are collected as usual, so the caller absorbs
//! them in the same order as ever. Once no engine holds work, every GPU that
//! can be opened again is closed, which frees its memory. Resuming opens it
//! again. To the search a suspension looks like a GPU that was slow for a
//! while, and the ring absorbs blocks in a fixed order whatever the timing,
//! so a suspended and resumed run gives the same results.

use super::*;

#[derive(Default)]
pub(super) struct Suspension {
    active: bool,
    /// Devices whose GPU engine was closed, to open again on resume.
    closed: Vec<usize>,
}

impl Scheduler {
    /// Stops handing new work to the engines.
    pub fn suspend(&mut self) {
        self.suspension.active = true;
    }

    pub fn suspended(&self) -> bool {
        self.suspension.active
    }

    /// Whether the scheduler may submit work now.
    pub(super) fn may_submit(&self) -> bool {
        !self.suspension.active
    }

    /// While suspended: once no engine holds a unit, closes every GPU engine
    /// that can be opened again. Returns true when no engine holds work and
    /// the GPUs are closed.
    pub fn close_idle_engines(&mut self) -> bool {
        if !self.suspension.active {
            return false;
        }
        if self.devices.iter().any(|d| !d.queued.is_empty()) {
            return false;
        }
        for (index, device) in self.devices.iter_mut().enumerate() {
            if device.kind == DeviceKind::Gpu
                && device.reopen.is_some()
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

    /// Ends a suspension: opens the closed GPUs again and lets work flow. A
    /// GPU that does not open is marked failed, so the usual recovery tries
    /// again and, failing that, moves its work to the CPU.
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
