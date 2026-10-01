//! The global allocator. Large blocks come straight from the kernel as
//! anonymous mappings, and a freed one is kept for the next block of its
//! size. Small blocks come from the system allocator, which is set to keep
//! the memory it frees.
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

pub struct BlockAlloc;

use std::sync::atomic::AtomicU64;

/// Large blocks: reused from the kept list, newly mapped, and given back.
static HITS: AtomicU64 = AtomicU64::new(0);
static MAPS: AtomicU64 = AtomicU64::new(0);
static UNMAPS: AtomicU64 = AtomicU64::new(0);

/// (reused, mapped, unmapped) large blocks since the start (a diagnostic).
pub fn large_blocks() -> (u64, u64, u64) {
    (
        HITS.load(Ordering::Relaxed),
        MAPS.load(Ordering::Relaxed),
        UNMAPS.load(Ordering::Relaxed),
    )
}

static TUNED: AtomicBool = AtomicBool::new(false);
static COUNTING: AtomicBool = AtomicBool::new(false);
static TOTAL: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static ALLOCATIONS: Cell<u64> = const { Cell::new(0) };
}

/// Starts counting allocations per thread (a benchmark's diagnostic; off
/// otherwise, and then it costs one relaxed load per allocation).
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

fn count() {
    if COUNTING.load(Ordering::Relaxed) {
        let _ = ALLOCATIONS.try_with(|n| n.set(n.get() + 1));
        TOTAL.fetch_add(1, Ordering::Relaxed);
    }
}

/// Sets glibc's allocator once, before its first use here: blocks under
/// `LARGE` always come from its heaps (by default it maps blocks over 128 KiB
/// and moves that threshold up and down as they are freed), and a heap
/// returns memory to the system only above 1 GiB free at its end.
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

fn large(layout: Layout) -> bool {
    layout.size() >= LARGE && layout.align() <= 4096
}

/// The bytes mapped for a block of `size`: a multiple of 2 MiB up to 32 MiB,
/// then steps of an eighth of the power of two below. Blocks of one class are
/// interchangeable. Memory past `size` that nothing touches is never faulted
/// in.
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

    /// A kept mapping of exactly `len` bytes.
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

    /// Keeps the mapping. When the list is full the oldest mappings go back
    /// to the kernel first; a mapping that cannot fit at all is refused.
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
/// are whatever the last block left) or a new zeroed one.
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
        // SAFETY: `ptr` came from `alloc` with this layout, so a large one
        // is a mapping of `class` bytes.
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
        // SAFETY: `ptr` came from `alloc` with `layout`; the new layout has
        // the same alignment.
        unsafe {
            let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
            match (large(layout), large(new_layout)) {
                (false, false) => {
                    tune();
                    System.realloc(ptr, layout, new_size)
                }
                (true, true) => {
                    let (old, new) = (class(layout.size()), class(new_size));
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
                    let moved =
                        libc::mremap(ptr as *mut libc::c_void, old, new, libc::MREMAP_MAYMOVE);
                    if moved == libc::MAP_FAILED {
                        return std::ptr::null_mut();
                    }
                    moved as *mut u8
                }
                _ => {
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
            assert!(c >= size && c % STEP == 0, "{size} -> {c}");
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
