//! The global allocator: large blocks come straight from the kernel as
//! anonymous mappings advised to use transparent huge pages, small ones from
//! the system allocator, which is set to keep the memory it frees.
//!
//! The breeder, the archives and the pack work on vectors of tens to
//! hundreds of megabytes. With 4 KiB pages every fresh page of them is a
//! fault, and a block of a ring that grows or is bred anew faults them all
//! again. A huge page is one fault for 2 MiB, so the same memory costs 512
//! times fewer faults, and `mremap` grows a vector without copying it. Where
//! the kernel has no huge pages the advice does nothing and the mappings are
//! ordinary ones.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, Ordering};

/// Blocks of at least this many bytes are mapped directly.
const LARGE: usize = 2 << 20;
/// The size of a huge page.
const HUGE: usize = 2 << 20;

pub struct HugeAlloc;

static TUNED: AtomicBool = AtomicBool::new(false);

/// Sets glibc's allocator once, before its first use here: blocks under
/// `LARGE` always come from its heaps (by default it maps blocks over 128 KiB
/// and moves that threshold up and down as they are freed), and a heap
/// returns memory to the system only above 1 GiB free at its end. A vector
/// of a million 8 byte entries that is freed and allocated again every block
/// then reuses its pages instead of faulting them in again.
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

/// A mapping is a whole number of huge pages: the kernel aligns a mapping to
/// 2 MiB only then, and a tail of small pages would fault 512 times as often.
/// Memory past `size` that nothing touches is never faulted in.
fn mapped_len(size: usize) -> usize {
    size.next_multiple_of(HUGE)
}

/// Maps `size` zeroed bytes advised for huge pages, or null.
unsafe fn map(size: usize) -> *mut u8 {
    // SAFETY: an anonymous private mapping with no address constraint.
    unsafe {
        let ptr = libc::mmap(
            std::ptr::null_mut(),
            mapped_len(size),
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
            -1,
            0,
        );
        if ptr == libc::MAP_FAILED {
            return std::ptr::null_mut();
        }
        libc::madvise(ptr, mapped_len(size), libc::MADV_HUGEPAGE);
        ptr as *mut u8
    }
}

unsafe impl GlobalAlloc for HugeAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's layout contract is passed on.
        unsafe {
            if large(layout) {
                map(layout.size())
            } else {
                tune();
                System.alloc(layout)
            }
        }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: as in `alloc`; a fresh mapping is zero.
        unsafe {
            if large(layout) {
                map(layout.size())
            } else {
                tune();
                System.alloc_zeroed(layout)
            }
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: `ptr` came from `alloc` with this layout, so a large one
        // is a mapping of `mapped_len` bytes.
        unsafe {
            if large(layout) {
                libc::munmap(ptr as *mut libc::c_void, mapped_len(layout.size()));
            } else {
                System.dealloc(ptr, layout)
            }
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
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
                    let moved = libc::mremap(
                        ptr as *mut libc::c_void,
                        mapped_len(layout.size()),
                        mapped_len(new_size),
                        libc::MREMAP_MAYMOVE,
                    );
                    if moved == libc::MAP_FAILED {
                        return std::ptr::null_mut();
                    }
                    libc::madvise(moved, mapped_len(new_size), libc::MADV_HUGEPAGE);
                    moved as *mut u8
                }
                _ => {
                    let fresh = self.alloc(new_layout);
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
