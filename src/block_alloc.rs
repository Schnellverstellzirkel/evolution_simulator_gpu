//! The global allocator. Large blocks come straight from the kernel as
//! anonymous mappings, and a freed one is kept for the next block of its
//! size class. Small blocks come from the system allocator, which is set to
//! keep the memory it frees.
//!
//! The breeder, the archives and the pack work on vectors of megabytes to
//! hundreds of megabytes, and the ring makes the same ones again for every
//! block. A vector freed to the kernel gives its pages back, so each block
//! faulted and zeroed them all again (450,000 faults per generation of 1M
//! creatures). Here a vector of a size seen before reuses the pages of the
//! last one, and `mremap` grows a vector with no copy. Transparent huge pages
//! were tried and left out: they cut the faults further but stalled single
//! generations for seconds in compaction.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::{Cell, UnsafeCell};
use std::sync::atomic::{AtomicBool, Ordering};

/// Blocks of at least this many bytes are mapped directly.
const LARGE: usize = 2 << 20;
/// Size classes are multiples of this many bytes.
const STEP: usize = 2 << 20;
/// Mappings kept for reuse, at most this many bytes.
const KEEP_BYTES: usize = 1 << 30;
/// Mappings kept for reuse, at most this many.
const KEEP_COUNT: usize = 64;

/// The program's global allocator, which `lib.rs` installs with
/// `#[global_allocator]`. It has no fields. The kept mappings and the counters
/// are statics of this module.
pub struct BlockAlloc;

use std::sync::atomic::AtomicU64;

/// Large blocks handed out from the kept list.
static HITS: AtomicU64 = AtomicU64::new(0);
/// Large blocks newly mapped from the kernel.
static MAPS: AtomicU64 = AtomicU64::new(0);
/// Mappings given back to the kernel.
static UNMAPS: AtomicU64 = AtomicU64::new(0);

/// How many large blocks were reused from the kept list, newly mapped and
/// unmapped, in that order, since the program started. A diagnostic.
pub fn large_blocks() -> (u64, u64, u64) {
    (
        HITS.load(Ordering::Relaxed),
        MAPS.load(Ordering::Relaxed),
        UNMAPS.load(Ordering::Relaxed),
    )
}

/// Whether `tune` has already set glibc's parameters.
static TUNED: AtomicBool = AtomicBool::new(false);
/// Whether allocations are being counted. `count_allocations` sets it.
static COUNTING: AtomicBool = AtomicBool::new(false);
/// Allocations and reallocations counted on all threads together.
static TOTAL: AtomicU64 = AtomicU64::new(0);

thread_local! {
    // Allocations and reallocations counted on this thread.
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

/// Turns allocation counting on, for each thread and for all threads
/// together. It stays on for the rest of the run. It is a diagnostic for
/// benchmarks. While counting is off, an allocation costs one relaxed load.
pub fn count_allocations() {
    COUNTING.store(true, Ordering::Relaxed);
}

/// Allocations and reallocations every thread made since `count_allocations`.
pub fn total_allocations() -> u64 {
    TOTAL.load(Ordering::Relaxed)
}

/// Allocations and reallocations this thread made since `count_allocations`.
pub fn allocations() -> u64 {
    ALLOCATIONS.try_with(Cell::get).unwrap_or(0)
}

/// Counts one allocation, for this thread and in total, while counting is on.
fn count() {
    if COUNTING.load(Ordering::Relaxed) {
        let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
        TOTAL.fetch_add(1, Ordering::Relaxed);
    }
}

/// Sets glibc's allocator once, before its first use here. Blocks under
/// `LARGE` always come from its heaps. Left alone, it maps blocks over 128 KiB
/// and raises that threshold as it frees mapped blocks. A heap returns memory
/// to the system only when more than 1 GiB is free at its end, and it asks the
/// system for 16 MiB extra each time it grows.
fn tune() {
    if !TUNED.swap(true, Ordering::Relaxed) {
        // SAFETY: mallopt only sets allocator parameters.
        unsafe {
            libc::mallopt(libc::M_MMAP_THRESHOLD, LARGE as libc::c_int);
            libc::mallopt(libc::M_TRIM_THRESHOLD, 1 << 30);
            libc::mallopt(libc::M_TOP_PAD, 16 << 20);
        }
    }
}

/// Whether a block of this layout is mapped directly. It must be at least
/// `LARGE` bytes, and a mapping is page aligned, so an alignment above 4096
/// bytes goes to the system allocator.
fn large(layout: Layout) -> bool {
    layout.size() >= LARGE && layout.align() <= 4096
}

/// The bytes mapped for a block of `size`: a multiple of 2 MiB up to 32 MiB,
/// then steps of an eighth of the power of two below `size`. Blocks of one
/// class are interchangeable. Memory past `size` that nothing touches is never
/// faulted in.
fn class(size: usize) -> usize {
    if size <= 32 << 20 {
        size.next_multiple_of(STEP)
    } else {
        let step = (size.next_power_of_two() / 2 / 8).max(STEP);
        size.next_multiple_of(step)
    }
}

/// Mappings of freed blocks, by address and class length, under a spin lock.
/// The allocator may not allocate, so the list is a fixed array.
struct Kept {
    lock: AtomicBool,
    inner: UnsafeCell<KeptInner>,
}

/// The list behind the lock. `entries[..count]` are the kept mappings as
/// (address, length) pairs, and `bytes` is the sum of their lengths.
struct KeptInner {
    count: usize,
    bytes: usize,
    entries: [(usize, usize); KEEP_COUNT],
}

// SAFETY: `inner` is only touched with `lock` held.
unsafe impl Sync for Kept {}

static KEPT: Kept = Kept {
    lock: AtomicBool::new(false),
    inner: UnsafeCell::new(KeptInner {
        count: 0,
        bytes: 0,
        entries: [(0, 0); KEEP_COUNT],
    }),
};

impl Kept {
    /// Runs `f` on the list while holding the spin lock.
    fn with<R>(&self, f: impl FnOnce(&mut KeptInner) -> R) -> R {
        while self
            .lock
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            std::hint::spin_loop();
        }
        // SAFETY: the lock is held.
        let result = f(unsafe { &mut *self.inner.get() });
        self.lock.store(false, Ordering::Release);
        result
    }

    /// Removes a kept mapping of exactly `len` bytes from the list and returns
    /// it, or returns null when there is none.
    fn take(&self, len: usize) -> *mut u8 {
        self.with(|k| {
            let n = k.count;
            match k.entries[..n].iter().position(|e| e.1 == len) {
                Some(i) => {
                    let (ptr, len) = k.entries[i];
                    k.entries[i] = k.entries[n - 1];
                    k.count -= 1;
                    k.bytes -= len;
                    ptr as *mut u8
                }
                None => std::ptr::null_mut(),
            }
        })
    }

    /// Keeps the mapping and returns true. When the list is full or would pass
    /// `KEEP_BYTES`, mappings from the front of the list go back to the kernel
    /// first. The front is the oldest, except that `take` fills the slot it
    /// frees with the last entry. A mapping longer than `KEEP_BYTES` is refused
    /// and the call returns false.
    fn put(&self, ptr: *mut u8, len: usize) -> bool {
        self.with(|k| {
            if len > KEEP_BYTES {
                return false;
            }
            while k.count == KEEP_COUNT || k.bytes + len > KEEP_BYTES {
                let (old, old_len) = k.entries[0];
                k.entries.copy_within(1..k.count, 0);
                k.count -= 1;
                k.bytes -= old_len;
                UNMAPS.fetch_add(1, Ordering::Relaxed);
                // SAFETY: a kept mapping is whole and unused.
                unsafe { libc::munmap(old as *mut libc::c_void, old_len) };
            }
            k.entries[k.count] = (ptr as usize, len);
            k.count += 1;
            k.bytes += len;
            true
        })
    }
}

/// Maps `len` zeroed bytes, or null.
unsafe fn map(len: usize) -> *mut u8 {
    // SAFETY: an anonymous private mapping with no address constraint.
    unsafe {
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        );
        if ptr == libc::MAP_FAILED {
            return std::ptr::null_mut();
        }
        ptr as *mut u8
    }
}

/// A large block of `size` bytes: a kept mapping of its class (its contents
/// are whatever the last block left) or a new zeroed one. It returns the
/// pointer, which is null if the kernel refused. It also returns whether the
/// memory is a new mapping and so zero.
unsafe fn large_alloc(size: usize) -> (*mut u8, bool) {
    let len = class(size);
    let kept = KEPT.take(len);
    if !kept.is_null() {
        HITS.fetch_add(1, Ordering::Relaxed);
        return (kept, false);
    }
    MAPS.fetch_add(1, Ordering::Relaxed);
    // SAFETY: as in `map`.
    (unsafe { map(len) }, true)
}

unsafe impl GlobalAlloc for BlockAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        count();
        // SAFETY: the caller's layout contract is passed on.
        unsafe {
            if large(layout) {
                large_alloc(layout.size()).0
            } else {
                tune();
                System.alloc(layout)
            }
        }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        count();
        // SAFETY: as in `alloc`; a new mapping is zero and a kept one is
        // cleared.
        unsafe {
            if large(layout) {
                let (ptr, zero) = large_alloc(layout.size());
                if !ptr.is_null() && !zero {
                    std::ptr::write_bytes(ptr, 0, layout.size());
                }
                ptr
            } else {
                tune();
                System.alloc_zeroed(layout)
            }
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from this allocator with this layout, so a large
        // one is a mapping of `class` bytes.
        unsafe {
            if large(layout) {
                let len = class(layout.size());
                if !KEPT.put(ptr, len) {
                    UNMAPS.fetch_add(1, Ordering::Relaxed);
                    libc::munmap(ptr as *mut libc::c_void, len);
                }
            } else {
                System.dealloc(ptr, layout)
            }
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        count();
        // SAFETY: `ptr` came from this allocator with `layout`, and the new
        // layout has the same alignment.
        unsafe {
            let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
            match (large(layout), large(new_layout)) {
                (false, false) => {
                    tune();
                    System.realloc(ptr, layout, new_size)
                }
                (true, true) => {
                    let (old, new) = (class(layout.size()), class(new_size));
                    // The same class: the mapping already fits the new size.
                    if old == new {
                        return ptr;
                    }
                    // A vector that grows block after block, by doubling, would
                    // fault in the new part of a mapping every time. A kept
                    // mapping of the new class costs a copy instead.
                    let kept = KEPT.take(new);
                    if !kept.is_null() {
                        HITS.fetch_add(1, Ordering::Relaxed);
                        std::ptr::copy_nonoverlapping(ptr, kept, layout.size().min(new_size));
                        self.dealloc(ptr, layout);
                        return kept;
                    }
                    // No kept mapping: the kernel extends or moves the pages
                    // without a copy. If it fails, the old mapping stays valid.
                    let moved =
                        libc::mremap(ptr as *mut libc::c_void, old, new, libc::MREMAP_MAYMOVE);
                    if moved == libc::MAP_FAILED {
                        return std::ptr::null_mut();
                    }
                    moved as *mut u8
                }
                _ => {
                    // The block crosses `LARGE`, so it moves between a mapping
                    // and the system allocator, with a copy.
                    let fresh = if large(new_layout) {
                        large_alloc(new_size).0
                    } else {
                        tune();
                        System.alloc(new_layout)
                    };
                    if !fresh.is_null() {
                        std::ptr::copy_nonoverlapping(ptr, fresh, layout.size().min(new_size));
                        self.dealloc(ptr, layout);
                    }
                    fresh
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_are_whole_steps_and_cover_the_size() {
        for size in [
            LARGE,
            LARGE + 1,
            5 << 20,
            32 << 20,
            (32 << 20) + 1,
            100 << 20,
            1 << 30,
        ] {
            let c = class(size);
            assert!(c >= size && c.is_multiple_of(STEP), "{size} -> {c}");
            assert!(c <= size + size / 8 + STEP, "{size} -> {c}");
        }
    }

    #[test]
    fn large_vectors_grow_keep_their_contents_and_come_back_cleared() {
        let mut v: Vec<u64> = Vec::new();
        for i in 0..(5 << 20) / 8 {
            v.push(i as u64);
        }
        v.reserve_exact(40 << 20);
        assert!(v.iter().enumerate().all(|(i, &x)| x == i as u64));
        v.truncate(10);
        v.shrink_to_fit();
        assert_eq!(v, (0..10).collect::<Vec<u64>>());
        drop(v);
        // The freed mapping is kept and handed out again, and a zeroed
        // allocation of it is zero.
        let dirty = vec![7u8; 6 << 20];
        drop(dirty);
        let clean = vec![0u8; 6 << 20];
        assert!(clean.iter().all(|&b| b == 0));
    }
}
