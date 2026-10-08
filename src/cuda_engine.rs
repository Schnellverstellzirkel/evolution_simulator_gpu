//! The CUDA backend runs the creature kernel (`shaders/creature.cu`, one
//! creature per thread) for `engine::gpu_engine`, which opens it on its own
//! thread and drives it with `submit`, `record` and `poll`.
//!
//! A unit is uploaded once and runs as waves of up to `kernel::WAVE`
//! creatures, one kernel launch each, on the streams of a submission slot. The
//! CUDA driver (`libcuda`) and NVRTC (`libnvrtc`) load when the engine opens,
//! so the game builds without them but needs them to run (`docs/building.md`).
//! Kernels compile in the background to a cubin for the device, one for each
//! world, fidelity and recording mode, and a disk cache keeps them.
use crate::{
    config::Config,
    creature_kernel::{self, GpuResult, LaneBatch},
    engine::Completed,
    physics::Fidelity,
};
use anyhow::{Context, Result, bail, ensure};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    ffi::{CStr, CString, c_char, c_int, c_uint, c_void},
    path::{Path, PathBuf},
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

/// Wall nanoseconds engine threads have waited for a kernel (compiling or
/// loading one the engine did not have yet).
static KERNEL_WAIT_NANOS: AtomicU64 = AtomicU64::new(0);
/// Engine threads that wait for a kernel now.
static KERNEL_WAITERS: AtomicUsize = AtomicUsize::new(0);

/// Seconds engine threads have waited for kernels since the process began.
/// A GPU with work queued does not run during this time, so a world change
/// that needs a kernel nobody compiled yet shows up here.
pub fn kernel_wait_seconds() -> f64 {
    KERNEL_WAIT_NANOS.load(Ordering::Relaxed) as f64 * 1e-9
}

/// Whether an engine thread waits for a kernel now (the status line says
/// the new world is compiling).
pub fn compiling_world() -> bool {
    KERNEL_WAITERS.load(Ordering::Relaxed) > 0
}

type CuResult = c_int;
type CuDevice = c_int;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuStream = *mut c_void;
type CuEvent = *mut c_void;
type CuDevicePtr = u64;
type NvrtcProgram = *mut c_void;

// Values of the CUDA driver API: result codes, flags and device attributes.
const CUDA_SUCCESS: CuResult = 0;
const CUDA_ERROR_OUT_OF_MEMORY: CuResult = 2;
const CUDA_ERROR_NOT_READY: CuResult = 600;
const CU_STREAM_NON_BLOCKING: c_uint = 1;
/// CUDA stream priority: lower numbers run first and 0 is the default. A
/// `high_priority` stream asks for -100, which the driver clamps to the
/// highest priority it has.
fn stream_priority(high_priority: bool) -> c_int {
    if high_priority { -100 } else { 0 }
}
const CU_EVENT_DISABLE_TIMING: c_uint = 2;
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;

/// The CUDA driver API functions the engine uses.
struct Driver {
    init: unsafe extern "C" fn(c_uint) -> CuResult,
    device_get_count: unsafe extern "C" fn(*mut c_int) -> CuResult,
    device_get: unsafe extern "C" fn(*mut CuDevice, c_int) -> CuResult,
    device_get_name: unsafe extern "C" fn(*mut c_char, c_int, CuDevice) -> CuResult,
    device_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, CuDevice) -> CuResult,
    primary_ctx_retain: unsafe extern "C" fn(*mut CuContext, CuDevice) -> CuResult,
    primary_ctx_release: unsafe extern "C" fn(CuDevice) -> CuResult,
    ctx_set_current: unsafe extern "C" fn(CuContext) -> CuResult,
    ctx_synchronize: unsafe extern "C" fn() -> CuResult,
    module_load_data: unsafe extern "C" fn(*mut CuModule, *const c_void) -> CuResult,
    module_unload: unsafe extern "C" fn(CuModule) -> CuResult,
    module_get_function: unsafe extern "C" fn(*mut CuFunction, CuModule, *const c_char) -> CuResult,
    mem_alloc: unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult,
    mem_free: unsafe extern "C" fn(CuDevicePtr) -> CuResult,
    mem_alloc_host: unsafe extern "C" fn(*mut *mut c_void, usize) -> CuResult,
    mem_free_host: unsafe extern "C" fn(*mut c_void) -> CuResult,
    mem_host_register: unsafe extern "C" fn(*mut c_void, usize, c_uint) -> CuResult,
    mem_host_unregister: unsafe extern "C" fn(*mut c_void) -> CuResult,
    memcpy_htod_async:
        unsafe extern "C" fn(CuDevicePtr, *const c_void, usize, CuStream) -> CuResult,
    memcpy_dtoh_async: unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize, CuStream) -> CuResult,
    stream_create: unsafe extern "C" fn(*mut CuStream, c_uint, c_int) -> CuResult,
    stream_destroy: unsafe extern "C" fn(CuStream) -> CuResult,
    stream_wait_event: unsafe extern "C" fn(CuStream, CuEvent, c_uint) -> CuResult,
    event_create: unsafe extern "C" fn(*mut CuEvent, c_uint) -> CuResult,
    event_destroy: unsafe extern "C" fn(CuEvent) -> CuResult,
    event_record: unsafe extern "C" fn(CuEvent, CuStream) -> CuResult,
    event_query: unsafe extern "C" fn(CuEvent) -> CuResult,
    event_elapsed_time: unsafe extern "C" fn(*mut f32, CuEvent, CuEvent) -> CuResult,
    func_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, CuFunction) -> CuResult,
    func_set_attribute: unsafe extern "C" fn(CuFunction, c_int, c_int) -> CuResult,
    memset_d32_async: unsafe extern "C" fn(CuDevicePtr, c_uint, usize, CuStream) -> CuResult,
    occupancy: unsafe extern "C" fn(*mut c_int, CuFunction, c_int, usize) -> CuResult,
    #[allow(clippy::type_complexity)]
    launch_kernel: unsafe extern "C" fn(
        CuFunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        CuStream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CuResult,
    get_error_name: unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult,
    _library: libloading::Library,
}

/// The NVRTC functions the engine uses.
struct Nvrtc {
    version: unsafe extern "C" fn(*mut c_int, *mut c_int) -> c_int,
    create_program: unsafe extern "C" fn(
        *mut NvrtcProgram,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> c_int,
    compile_program: unsafe extern "C" fn(NvrtcProgram, c_int, *const *const c_char) -> c_int,
    get_program_log_size: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> c_int,
    get_program_log: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> c_int,
    get_cubin_size: unsafe extern "C" fn(NvrtcProgram, *mut usize) -> c_int,
    get_cubin: unsafe extern "C" fn(NvrtcProgram, *mut c_char) -> c_int,
    destroy_program: unsafe extern "C" fn(*mut NvrtcProgram) -> c_int,
    get_error_string: unsafe extern "C" fn(c_int) -> *const c_char,
    path: PathBuf,
    _library: libloading::Library,
    /// NVRTC loads its builtins library by name; loading it first, with
    /// global symbols, lets it be found beside a library outside the
    /// loader's search path (a pip wheel).
    _builtins: Option<libloading::Library>,
}

/// Loaded CUDA libraries, shared by every engine and compile thread.
struct Api {
    cu: Driver,
    nvrtc: Nvrtc,
}

/// Looks up the function `$name` in `$library` and copies its pointer out.
/// It needs an `unsafe` block, and it returns from the calling function with
/// an error that names the symbol when the library lacks it.
macro_rules! symbol {
    ($library:expr, $name:literal) => {
        *$library
            .get(concat!($name, "\0").as_bytes())
            .with_context(|| format!("CUDA symbol {}", $name))?
    };
}

impl Driver {
    fn load() -> Result<Self> {
        let names: &[&str] = if cfg!(windows) {
            &["nvcuda.dll"]
        } else {
            &["libcuda.so.1", "libcuda.so"]
        };
        let library = names
            .iter()
            .find_map(|name| unsafe { libloading::Library::new(name).ok() })
            .context("CUDA driver library (libcuda) not found")?;
        unsafe {
            Ok(Self {
                init: symbol!(library, "cuInit"),
                device_get_count: symbol!(library, "cuDeviceGetCount"),
                device_get: symbol!(library, "cuDeviceGet"),
                device_get_name: symbol!(library, "cuDeviceGetName"),
                device_get_attribute: symbol!(library, "cuDeviceGetAttribute"),
                primary_ctx_retain: symbol!(library, "cuDevicePrimaryCtxRetain"),
                primary_ctx_release: symbol!(library, "cuDevicePrimaryCtxRelease_v2"),
                ctx_set_current: symbol!(library, "cuCtxSetCurrent"),
                ctx_synchronize: symbol!(library, "cuCtxSynchronize"),
                module_load_data: symbol!(library, "cuModuleLoadData"),
                module_unload: symbol!(library, "cuModuleUnload"),
                module_get_function: symbol!(library, "cuModuleGetFunction"),
                mem_alloc: symbol!(library, "cuMemAlloc_v2"),
                mem_free: symbol!(library, "cuMemFree_v2"),
                mem_alloc_host: symbol!(library, "cuMemAllocHost_v2"),
                mem_free_host: symbol!(library, "cuMemFreeHost"),
                mem_host_register: symbol!(library, "cuMemHostRegister_v2"),
                mem_host_unregister: symbol!(library, "cuMemHostUnregister"),
                memcpy_htod_async: symbol!(library, "cuMemcpyHtoDAsync_v2"),
                memcpy_dtoh_async: symbol!(library, "cuMemcpyDtoHAsync_v2"),
                stream_create: symbol!(library, "cuStreamCreateWithPriority"),
                stream_destroy: symbol!(library, "cuStreamDestroy_v2"),
                stream_wait_event: symbol!(library, "cuStreamWaitEvent"),
                event_create: symbol!(library, "cuEventCreate"),
                event_destroy: symbol!(library, "cuEventDestroy_v2"),
                event_record: symbol!(library, "cuEventRecord"),
                event_query: symbol!(library, "cuEventQuery"),
                event_elapsed_time: symbol!(library, "cuEventElapsedTime"),
                func_get_attribute: symbol!(library, "cuFuncGetAttribute"),
                func_set_attribute: symbol!(library, "cuFuncSetAttribute"),
                memset_d32_async: symbol!(library, "cuMemsetD32Async"),
                occupancy: symbol!(library, "cuOccupancyMaxActiveBlocksPerMultiprocessor"),
                launch_kernel: symbol!(library, "cuLaunchKernel"),
                get_error_name: symbol!(library, "cuGetErrorName"),
                _library: library,
            })
        }
    }

    /// Turns a failed driver result into a `CudaError` that names the call
    /// (`what`).
    fn check(&self, result: CuResult, what: &str) -> Result<()> {
        if result == CUDA_SUCCESS {
            return Ok(());
        }
        let mut name: *const c_char = std::ptr::null();
        unsafe { (self.get_error_name)(result, &mut name) };
        let name = if name.is_null() {
            "unknown error".into()
        } else {
            unsafe { CStr::from_ptr(name) }
                .to_string_lossy()
                .into_owned()
        };
        Err(CudaError {
            code: result,
            name,
            what: what.to_owned(),
        }
        .into())
    }
}

/// The NVRTC libraries to try, in order. When the developer diagnostic
/// `EVOLUTION_NVRTC` is set, it is the only one. Otherwise they are the
/// loader's search path, the toolkit folders `/usr/local/cuda` and
/// `/opt/cuda` on Linux, then NVIDIA's pip wheels in the virtual environment
/// that `docs/building.md` sets up.
fn nvrtc_candidates() -> Vec<PathBuf> {
    if let Some(path) = std::env::var_os("EVOLUTION_NVRTC") {
        return vec![PathBuf::from(path)];
    }
    let mut candidates: Vec<PathBuf> = if cfg!(windows) {
        ["nvrtc64_130_0.dll", "nvrtc64_120_0.dll"]
            .iter()
            .map(PathBuf::from)
            .collect()
    } else {
        [
            "libnvrtc.so.13",
            "libnvrtc.so.12",
            "libnvrtc.so",
            "/usr/local/cuda/lib64/libnvrtc.so",
            "/opt/cuda/lib64/libnvrtc.so",
        ]
        .iter()
        .map(PathBuf::from)
        .collect()
    };
    if let Some(home) = std::env::var_os("HOME") {
        let venv = Path::new(&home).join(".local/share/evolution-cuda/venv/lib");
        for python in std::fs::read_dir(&venv).into_iter().flatten().flatten() {
            let nvidia = python.path().join("site-packages/nvidia");
            // cu13 wheels install under nvidia/cu13, cu12 wheels under
            // nvidia/cuda_nvrtc.
            for (folder, name) in [
                ("cu13/lib", "libnvrtc.so.13"),
                ("cuda_nvrtc/lib", "libnvrtc.so.12"),
            ] {
                let path = nvidia.join(folder).join(name);
                if path.exists() {
                    candidates.push(path);
                }
            }
        }
    }
    candidates
}

impl Nvrtc {
    /// Opens the first of `nvrtc_candidates` that loads. The error lists why
    /// each one failed.
    fn load() -> Result<Self> {
        let mut errors = Vec::new();
        for path in nvrtc_candidates() {
            match Self::open(&path) {
                Ok(nvrtc) => return Ok(nvrtc),
                Err(error) => errors.push(format!("{}: {error:#}", path.display())),
            }
        }
        bail!("NVRTC not found ({})", errors.join("; "))
    }

    fn open(path: &Path) -> Result<Self> {
        let builtins = Self::open_builtins(path);
        let library = unsafe { libloading::Library::new(path) }?;
        unsafe {
            Ok(Self {
                version: symbol!(library, "nvrtcVersion"),
                create_program: symbol!(library, "nvrtcCreateProgram"),
                compile_program: symbol!(library, "nvrtcCompileProgram"),
                get_program_log_size: symbol!(library, "nvrtcGetProgramLogSize"),
                get_program_log: symbol!(library, "nvrtcGetProgramLog"),
                get_cubin_size: symbol!(library, "nvrtcGetCUBINSize"),
                get_cubin: symbol!(library, "nvrtcGetCUBIN"),
                destroy_program: symbol!(library, "nvrtcDestroyProgram"),
                get_error_string: symbol!(library, "nvrtcGetErrorString"),
                path: path.to_owned(),
                _library: library,
                _builtins: builtins,
            })
        }
    }

    /// Loads `libnvrtc-builtins` from the directory of an NVRTC given by path.
    #[cfg(unix)]
    fn open_builtins(path: &Path) -> Option<libloading::Library> {
        let folder = path.parent().filter(|p| !p.as_os_str().is_empty())?;
        let builtins = std::fs::read_dir(folder)
            .ok()?
            .flatten()
            .map(|entry| entry.path())
            .find(|file| {
                file.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("libnvrtc-builtins.so."))
            })?;
        let flags = libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL;
        unsafe { libloading::os::unix::Library::open(Some(&builtins), flags) }
            .ok()
            .map(libloading::Library::from)
    }

    #[cfg(not(unix))]
    fn open_builtins(_path: &Path) -> Option<libloading::Library> {
        None
    }

    /// NVRTC's version as `major.minor`. It is part of the kernel cache key.
    fn version_string(&self) -> String {
        let (mut major, mut minor) = (0, 0);
        unsafe { (self.version)(&mut major, &mut minor) };
        format!("{major}.{minor}")
    }

    /// Compiles `source` to a cubin with `options`, returning it with the
    /// compiler log (ptxas statistics included).
    fn compile(&self, source: &str, options: &[String]) -> Result<(Vec<u8>, String)> {
        let source = CString::new(source)?;
        let options: Vec<CString> = options
            .iter()
            .map(|o| CString::new(o.as_str()))
            .collect::<Result<_, _>>()?;
        let option_ptrs: Vec<*const c_char> = options.iter().map(|o| o.as_ptr()).collect();
        unsafe {
            let mut program: NvrtcProgram = std::ptr::null_mut();
            let status = (self.create_program)(
                &mut program,
                source.as_ptr(),
                c"physics_creature.cu".as_ptr(),
                0,
                std::ptr::null(),
                std::ptr::null(),
            );
            if status != 0 {
                bail!("nvrtcCreateProgram: {}", self.error(status));
            }
            let status =
                (self.compile_program)(program, option_ptrs.len() as c_int, option_ptrs.as_ptr());
            let mut log_size = 0usize;
            (self.get_program_log_size)(program, &mut log_size);
            let mut log = vec![0u8; log_size.max(1)];
            (self.get_program_log)(program, log.as_mut_ptr() as *mut c_char);
            let log = CStr::from_bytes_until_nul(&log)
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_default();
            let result = if status != 0 {
                Err(anyhow::anyhow!(
                    "NVRTC compile failed: {}\n{log}",
                    self.error(status)
                ))
            } else {
                let mut size = 0usize;
                (self.get_cubin_size)(program, &mut size);
                let mut cubin = vec![0u8; size];
                let status = (self.get_cubin)(program, cubin.as_mut_ptr() as *mut c_char);
                if status != 0 {
                    Err(anyhow::anyhow!("nvrtcGetCUBIN: {}", self.error(status)))
                } else {
                    Ok((cubin, log))
                }
            };
            (self.destroy_program)(&mut program);
            result
        }
    }

    /// The text of an NVRTC status code.
    fn error(&self, status: c_int) -> String {
        let text = unsafe { (self.get_error_string)(status) };
        if text.is_null() {
            format!("error {status}")
        } else {
            unsafe { CStr::from_ptr(text) }
                .to_string_lossy()
                .into_owned()
        }
    }
}

/// A failed CUDA driver call.
#[derive(Debug)]
pub struct CudaError {
    code: CuResult,
    name: String,
    what: String,
}

impl CudaError {
    /// Whether the call failed for lack of device or pinned host memory.
    pub fn out_of_memory(&self) -> bool {
        self.code == CUDA_ERROR_OUT_OF_MEMORY
    }
}

impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CUDA {} failed: {} ({})",
            self.what, self.name, self.code
        )
    }
}

impl std::error::Error for CudaError {}

/// The loaded libraries. A failure is kept as its message, so every engine
/// that opens later reports it again.
static API: std::sync::OnceLock<Result<Arc<Api>, String>> = std::sync::OnceLock::new();

/// Hardware queues for the process's streams, set unless the environment
/// already has `CUDA_DEVICE_MAX_CONNECTIONS` (1 to 32). CUDA puts every stream
/// on one of 8 queues by default, and kernels on streams that share a queue
/// run one after the other. The engine has two streams for each slot, and the
/// 100 wild islands make about 800 small units a generation, so with 12 slots
/// only 3 or 4 of those ran at once. The driver reads the variable when it
/// starts.
const HARDWARE_QUEUES: &str = "32";

/// What `prepare_environment` sets, given the value the variable has now:
/// `HARDWARE_QUEUES` when it has none, and nothing when a developer set one.
fn queues_to_set(current: Option<&std::ffi::OsStr>) -> Option<&'static str> {
    current.is_none().then_some(HARDWARE_QUEUES)
}

/// Sets `CUDA_DEVICE_MAX_CONNECTIONS` to `HARDWARE_QUEUES` when it is not set.
/// It must run before `cuInit`. `main` calls it before any thread starts, and
/// `api` calls it for the tools and tests that reach the driver another way.
/// It sets the variable once.
pub fn prepare_environment() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let current = std::env::var_os("CUDA_DEVICE_MAX_CONNECTIONS");
        if let Some(queues) = queues_to_set(current.as_deref()) {
            // SAFETY: `std::env::set_var` is unsafe only because another
            // thread may read the environment through libc while it runs.
            // The game runs it first in `main`, before any thread exists.
            // Anywhere else it runs once, as the engine opens.
            unsafe { std::env::set_var("CUDA_DEVICE_MAX_CONNECTIONS", queues) };
        }
    });
}

/// The CUDA driver and NVRTC compiler, shared by all engines. The first call
/// loads both and initializes the driver.
fn api() -> Result<Arc<Api>> {
    API.get_or_init(|| {
        prepare_environment();
        let cu = Driver::load().map_err(|e| format!("{e:#}"))?;
        cu.check(unsafe { (cu.init)(0) }, "cuInit")
            .map_err(|e| format!("{e:#}"))?;
        let nvrtc = Nvrtc::load().map_err(|e| format!("{e:#}"))?;
        Ok(Arc::new(Api { cu, nvrtc }))
    })
    .clone()
    .map_err(|e| anyhow::anyhow!(e))
}

/// Units of more than this many creatures are big. The main islands of a ring
/// block make one, 100,000 creatures at 1M, while each wild island makes a
/// small one of a few hundred.
const BIG_UNIT: usize = 8192;

/// How many of the standard slots, counted from the first, big units use.
/// They are the only slots that ever hold the buffers of a block, so the
/// slots of the wild islands' small units stay small however many there are.
const BIG_SLOTS: usize = 4;

/// Block slots of the GPU that the kernel of a big unit leaves free. That
/// kernel holds every slot it may for as long as it runs, a second or more,
/// and a stream of the highest priority (a replay, a confirmation trial) gets
/// the next slot that frees up, which comes when the wave ends. The slots that
/// the big kernel leaves go to the small units, whose blocks exit within 0.2 s,
/// so a high priority launch starts within that.
const RESERVED_BLOCKS: usize = 2;

/// The blocks of the grid for a wave of `count` creatures when `resident`
/// blocks fit on the GPU at once: a block for every `kernel::BLOCK`
/// creatures, and no more than `resident` less `RESERVED_BLOCKS`. A block
/// takes creatures from the wave's counter until it is empty, so these are
/// enough.
fn wave_blocks(count: usize, resident: usize) -> usize {
    count
        .div_ceil(crate::kernel::BLOCK as usize)
        .min(resident.saturating_sub(RESERVED_BLOCKS))
        .max(1)
}

/// A standard slot as `pick_standard_slot` sees it.
#[derive(Clone, Copy, Debug)]
struct SlotView {
    /// No unit runs on it.
    free: bool,
    /// Its device buffers already hold the unit.
    holds: bool,
    /// Bytes of buffers it keeps.
    bytes: u64,
}

/// The standard slot for a unit, given `views` of all of them in order. A big
/// unit takes one of the first `big_slots` slots, which are the only ones that
/// ever hold a block's buffers. A small unit takes one of the others, or one
/// of the first slots when none of the others is free. In each group the slot
/// is the free one with the least buffers that already hold the unit, so that
/// the big buffers stay free for big units. When no free slot holds the unit
/// it is the free one with the most buffers, which grows. `None` when no slot
/// is free for the unit.
fn pick_standard_slot(views: &[SlotView], big_slots: usize, big: bool) -> Option<usize> {
    let best = |range: std::ops::Range<usize>| {
        let free = || range.clone().filter(|&i| views[i].free);
        free()
            .filter(|&i| views[i].holds)
            .min_by_key(|&i| (views[i].bytes, i))
            .or_else(|| free().max_by_key(|&i| (views[i].bytes, std::cmp::Reverse(i))))
    };
    let big_slots = 0..big_slots.min(views.len());
    let small_slots = big_slots.end..views.len();
    if big {
        best(big_slots)
    } else {
        best(small_slots).or_else(|| best(big_slots))
    }
}

/// Most wave streams a submission slot has. Wave w of a unit runs on stream w
/// modulo the streams in use, so a unit of more waves shares them.
const STREAM_LIMIT: usize = 8;

/// A kernel: recording or scoring, class, world flags (`kernel::world_flags`)
/// and fidelity. The class is the node slots of a frame (`kernel::CLASSES`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct KernelKey {
    record: bool,
    class: usize,
    flags: u32,
    fidelity: Fidelity,
}

/// A compiled CUDA kernel module and function with occupancy information.
struct Kernel {
    module: CuModule,
    function: CuFunction,
    /// Blocks of `kernel::BLOCK` threads resident per multiprocessor.
    blocks_per_sm: u32,
}

// A module belongs to the context, not to a thread: background threads load
// kernels for the engine, which launches them.
unsafe impl Send for Kernel {}

/// A device allocation: its address and its size in bytes, padded by
/// `buffer_size`.
struct DeviceBuf {
    ptr: CuDevicePtr,
    size: usize,
}

/// A pinned host allocation that the GPU copies results and recorded frames
/// into.
struct HostBuf {
    ptr: *mut u8,
    size: usize,
}

thread_local! {
    /// The driver, on a thread where an engine's context is current, so
    /// `HostVec` can register the memory it maps.
    static REGISTER: std::cell::RefCell<Option<Arc<Api>>> = const { std::cell::RefCell::new(None) };
}

/// Host memory the GPU copies from directly, with no staging copy: it is
/// mapped once, registered with the driver once (`cuMemHostRegister`, so it
/// stays page-locked and a copy from it runs asynchronously on the copy
/// engine), and reused from one unit to the next. It grows with headroom
/// and never shrinks. Registration needs an engine's thread, where its
/// context is current; small buffers and memory mapped on other threads
/// (tests) stay ordinary pages, which the driver copies through its own
/// staging.
pub struct HostVec<T: bytemuck::Pod> {
    ptr: *mut T,
    len: usize,
    /// Elements that fit in the mapping.
    capacity: usize,
    /// Mapped bytes; zero when nothing is mapped.
    bytes: usize,
    /// Whether the mapping is registered with the driver.
    registered: bool,
}

// A HostVec owns its memory like a Vec.
unsafe impl<T: bytemuck::Pod + Send> Send for HostVec<T> {}
unsafe impl<T: bytemuck::Pod + Sync> Sync for HostVec<T> {}

impl<T: bytemuck::Pod> Default for HostVec<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: bytemuck::Pod> HostVec<T> {
    /// Below this many bytes a buffer is not registered: the driver's own
    /// staging copies it about as fast.
    const REGISTER_FROM: usize = 1 << 16;

    /// An empty vector that has mapped nothing yet.
    pub const fn new() -> Self {
        Self {
            ptr: std::ptr::NonNull::dangling().as_ptr(),
            len: 0,
            capacity: 0,
            bytes: 0,
            registered: false,
        }
    }

    /// Sets the contents to `len` copies of `value`. It keeps the mapping
    /// when that is large enough and grows it when it is not.
    pub fn reset(&mut self, len: usize, value: T) {
        if len > self.capacity {
            self.grow(len);
        }
        self.len = len;
        self.fill(value);
    }

    /// Whether the driver copies from this memory directly.
    pub fn registered(&self) -> bool {
        self.registered
    }

    /// Bytes of memory held.
    pub fn held_bytes(&self) -> usize {
        self.bytes
    }

    /// Elements that fit in the mapping.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Maps room for at least `len` elements, with the headroom of
    /// `engine::padded_size`. It registers the mapping when it holds
    /// `REGISTER_FROM` bytes or more and an engine's context is current on
    /// this thread. The contents already there stay.
    fn grow(&mut self, len: usize) {
        let bytes = crate::engine::padded_size((len * std::mem::size_of::<T>()) as u64) as usize;
        let bytes = bytes.next_multiple_of(4096);
        let old = std::mem::replace(&mut self.bytes, 0);
        // SAFETY: an anonymous private mapping, owned by this HostVec until
        // `release`. A mapping already held is unregistered and extended in
        // place or moved with the pages it has (`mremap`), so memory already
        // faulted in stays faulted in; only the new part is fresh.
        let ptr = unsafe {
            if old == 0 {
                libc::mmap(
                    std::ptr::null_mut(),
                    bytes,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            } else {
                if self.registered
                    && let Ok(api) = api()
                {
                    (api.cu.mem_host_unregister)(self.ptr as *mut c_void);
                }
                self.registered = false;
                libc::mremap(self.ptr as *mut c_void, old, bytes, libc::MREMAP_MAYMOVE)
            }
        };
        if ptr == libc::MAP_FAILED {
            std::alloc::handle_alloc_error(
                std::alloc::Layout::from_size_align(bytes, 4096).expect("a page layout"),
            );
        }
        self.registered = bytes >= Self::REGISTER_FROM
            && REGISTER.with(|register| {
                register.borrow().as_ref().is_some_and(|api| unsafe {
                    (api.cu.mem_host_register)(ptr, bytes, 0) == CUDA_SUCCESS
                })
            });
        self.ptr = ptr as *mut T;
        self.bytes = bytes;
        self.capacity = bytes / std::mem::size_of::<T>().max(1);
    }

    /// Unregisters and unmaps the memory, and leaves the vector empty.
    fn release(&mut self) {
        if self.bytes == 0 {
            return;
        }
        // SAFETY: the mapping and its registration are this HostVec's; no
        // copy reads it any more (the engine keeps a unit's buffers until
        // its submission finished).
        unsafe {
            if self.registered
                && let Ok(api) = api()
            {
                (api.cu.mem_host_unregister)(self.ptr as *mut c_void);
            }
            libc::munmap(self.ptr as *mut c_void, self.bytes);
        }
        // Field by field: assigning a whole HostVec would drop this one again.
        self.ptr = std::ptr::NonNull::dangling().as_ptr();
        self.len = 0;
        self.capacity = 0;
        self.bytes = 0;
        self.registered = false;
    }
}

impl<T: bytemuck::Pod> std::ops::Deref for HostVec<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        // SAFETY: `len` elements are initialized (`reset`).
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<T: bytemuck::Pod> std::ops::DerefMut for HostVec<T> {
    fn deref_mut(&mut self) -> &mut [T] {
        // SAFETY: as in `deref`, and `&mut self` is unique.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl<T: bytemuck::Pod> Drop for HostVec<T> {
    fn drop(&mut self) {
        self.release();
    }
}

/// The device buffers of one batch, in the order of the kernel's first five
/// arguments: the node and bone records (`WavePack::lanes`), the muscle
/// records, the unused `WavePack::ends`, the heads and the results.
struct GroupRes {
    bufs: [DeviceBuf; 5],
}

/// The resources of one submission slot. Each wave runs on one of the slot's
/// streams. The main stream uploads, joins the wave streams and reads the
/// results back.
struct Slot {
    main: CuStream,
    /// The streams the waves run on, made as units need them.
    streams: Vec<CuStream>,
    /// One event per wave stream, recorded when that stream's waves finish.
    joins: Vec<CuEvent>,
    /// Timing events on the main stream: `start` after the uploads and `stop`
    /// after the waves. They give the GPU seconds of the waves.
    start: CuEvent,
    stop: CuEvent,
    /// Recorded after the readback copies. `poll` queries it.
    done: CuEvent,
    /// The buffers of each batch of the unit, by batch index. The slot keeps
    /// them for the next unit.
    groups: Vec<Option<GroupRes>>,
    /// One counter per wave: the next creature a thread of the wave takes.
    counters: Option<DeviceBuf>,
    /// Where the results and any recorded frames are copied to on the host.
    readback: Option<HostBuf>,
    /// Recorded replay frames; only the replay slot has one.
    frames: Option<DeviceBuf>,
    /// The submission that runs on this slot, until `poll` returns it.
    pending: Option<Pending>,
}

/// A GPU submission queued on a slot, waiting for results.
struct Pending {
    ticket: u64,
    /// The unit's batches: their buffers are copied from until the
    /// submission finishes, then go back to the caller for reuse.
    batches: Vec<LaneBatch>,
    result_count: usize,
    /// Recorded frames: their byte offset in the readback buffer and their
    /// count of float pairs.
    frames: Option<(usize, usize)>,
}

/// The engine of one CUDA GPU: it scores creatures and records replays.
pub struct CudaEngine {
    api: Arc<Api>,
    device: CuDevice,
    context: CuContext,
    /// The device name followed by "(CUDA)".
    pub name: String,
    /// The device's compute capability as an NVRTC architecture, such as
    /// `sm_89`.
    arch: String,
    multiprocessors: i32,
    /// Kernels loaded into the context and ready to launch.
    kernels: HashMap<KernelKey, Kernel>,
    /// Kernels compiling on background threads.
    prefetch: Arc<Prefetch>,
    /// Worlds (effect levels and flags) whose kernels have been queued on the
    /// background compiler.
    worlds: HashSet<(Vec<u8>, u32)>,
    /// Submission slots: `standard` slots for standard trials, then one for
    /// confirmation trials and a last one for replays. The last two have
    /// streams of the highest priority, so a confirmation or a replay never
    /// waits behind queued standard work for the multiprocessors.
    slots: Vec<Slot>,
    /// Slots of standard trials.
    standard: usize,
    /// The ticket the next submission gets.
    next_ticket: u64,
    /// Largest body, in nodes, the engine runs. It is at most
    /// `kernel::MAX_NODES`.
    pub max_capacity: usize,
    /// Bytes of device and pinned host buffers the slots hold, as
    /// `recount_allocated` counts them.
    pub allocated_bytes: u64,
    /// GPU seconds of the submission that `poll` returned last.
    pub last_gpu_seconds: f64,
}

// The context and every pointer of a slot are only used by the thread that
// owns the engine, which makes the context current on itself when it opens.
// Background threads share only `Prefetch`, and load kernels into the context.
unsafe impl Send for CudaEngine {}

/// Rounds a buffer size up (`engine::padded_size`), so buffers are reused
/// across units of slightly different sizes.
fn buffer_size(bytes: usize) -> usize {
    crate::engine::padded_size(bytes as u64) as usize
}

/// The cubin of one kernel, compiled by NVRTC or read from the disk cache. It
/// needs no CUDA context, so any thread may run it. Compiled kernels are kept
/// on disk (`kernel_cache_dir`), keyed by a hash of the source, the options
/// and the NVRTC version, so a later start loads them in milliseconds. With
/// `use_cache` false the compile ignores an entry (it may be damaged) and
/// writes a fresh one. With the developer diagnostic `EVOLUTION_CUDA_VERBOSE`
/// set, a compile prints its time and the compiler log.
fn compile_kernel(
    api: &Api,
    options: &[String],
    key: KernelKey,
    use_cache: bool,
) -> Result<Vec<u8>> {
    let task = crate::loading::start(kernel_label(key));
    let source = crate::kernel::cuda_source(key.class, key.flags, key.fidelity, key.record);
    let path = kernel_cache_dir().map(|dir| {
        use std::hash::{Hash, Hasher};
        let mut halves = [0u64; 2];
        for (i, half) in halves.iter_mut().enumerate() {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            (i, &source, options, api.nvrtc.version_string()).hash(&mut hasher);
            *half = hasher.finish();
        }
        dir.join(format!("{:016x}{:016x}.cubin", halves[0], halves[1]))
    });
    if use_cache
        && let Some(path) = &path
        && let Ok(bytes) = std::fs::read(path)
        && !bytes.is_empty()
    {
        // A kernel in use stays young, so eviction by age (`evict_cache`)
        // keeps it.
        if let Ok(file) = std::fs::File::open(path) {
            let _ = file.set_modified(std::time::SystemTime::now());
        }
        task.finish(true);
        return Ok(bytes);
    }
    let started = Instant::now();
    let (cubin, log) = api
        .nvrtc
        .compile(&source, options)
        .with_context(|| "the creature CUDA kernel")?;
    if std::env::var_os("EVOLUTION_CUDA_VERBOSE").is_some() {
        eprintln!(
            "CUDA: compiled {key:?} in {:.1} s\n{}",
            started.elapsed().as_secs_f64(),
            log.trim()
        );
    }
    if let Some(path) = path {
        // Through a temporary file and a rename, so a reader never sees a
        // half-written kernel. A failed write only costs the next start.
        let temporary = path.with_extension(format!("tmp{}", std::process::id()));
        let written = path
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|()| std::fs::write(&temporary, &cubin))
            .and_then(|()| std::fs::rename(&temporary, &path));
        if written.is_err() {
            let _ = std::fs::remove_file(&temporary);
        }
    }
    task.finish(false);
    Ok(cubin)
}

/// How the loading screen names a kernel: what it does and which effects the
/// world compiles in.
fn kernel_label(key: KernelKey) -> String {
    let physics = if key.fidelity == Fidelity::standard() {
        ""
    } else {
        " (fine physics)"
    };
    format!(
        "{} kernel{physics} · {}",
        if key.record { "replay" } else { "scoring" },
        crate::kernel::world_label(key.flags)
    )
}

/// Compiled kernels kept on disk, about 50 to 80 KB each. The starting worlds
/// alone are 53 kernels: the default world's 4 and the scoring kernels that
/// `prefetch_islands` queues for the 49 distinct worlds of the 100 wild
/// islands. Every start reads them again, so a cap below their number would
/// delete and compile them again at every start. A test keeps the cap at
/// twice their number or more.
const CACHE_FILES: usize = 600;

/// Deletes the oldest compiled kernels beyond `CACHE_FILES`, and temporary
/// files over an hour old that a crashed compile left. Every source edit
/// makes new kernels for every world, and nothing else ever removes the old
/// ones. A kernel another process still wants is compiled again. It runs once
/// per process, when the first engine opens.
fn evict_cache(dir: &Path) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let now = std::time::SystemTime::now();
    let mut kernels = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(modified) = entry.metadata().ok().and_then(|m| m.modified().ok()) else {
            continue;
        };
        match path.extension().and_then(|e| e.to_str()) {
            Some("cubin") => kernels.push((modified, path)),
            Some(e)
                if e.starts_with("tmp")
                    && now
                        .duration_since(modified)
                        .is_ok_and(|age| age > Duration::from_secs(3600)) =>
            {
                let _ = std::fs::remove_file(&path);
            }
            _ => {}
        }
    }
    if kernels.len() > CACHE_FILES {
        kernels.sort();
        let extra = kernels.len() - CACHE_FILES;
        for (_, path) in kernels.drain(..extra) {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Where compiled kernels are kept: the developer diagnostic
/// `EVOLUTION_KERNEL_CACHE`, else `evolution-simulator/cuda` in the user's
/// cache directory (`XDG_CACHE_HOME`, `LOCALAPPDATA` or `~/.cache`), else
/// nowhere.
fn kernel_cache_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os("EVOLUTION_KERNEL_CACHE") {
        return Some(PathBuf::from(dir));
    }
    let base = std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("LOCALAPPDATA").map(PathBuf::from))
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))?;
    Some(base.join("evolution-simulator").join("cuda"))
}

/// Loads a cubin into `context` (the calling thread makes it current). It
/// sets the kernel's cache preference and measures how many blocks of
/// `kernel::BLOCK` threads fit on a multiprocessor. With the developer
/// diagnostic `EVOLUTION_CUDA_VERBOSE` set it also prints the kernel's
/// registers and memory use.
fn load_kernel(api: &Api, context: CuContext, cubin: &[u8], key: KernelKey) -> Result<Kernel> {
    let cu = &api.cu;
    unsafe {
        cu.check((cu.ctx_set_current)(context), "cuCtxSetCurrent")?;
        let mut module = std::ptr::null_mut();
        let loading = Instant::now();
        cu.check(
            (cu.module_load_data)(&mut module, cubin.as_ptr() as *const c_void),
            "cuModuleLoadData",
        )?;
        let load_seconds = loading.elapsed().as_secs_f64();
        let mut function = std::ptr::null_mut();
        if let Err(error) = cu.check(
            (cu.module_get_function)(&mut function, module, c"advance".as_ptr()),
            "cuModuleGetFunction",
        ) {
            (cu.module_unload)(module);
            return Err(error);
        }
        // The kernel keeps the node state of a body of 16 nodes or fewer in
        // 48 KB of shared memory per block. With the preferred shared memory
        // carveout (CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
        // percent) at 0 the driver gives the multiprocessor just enough for
        // one block, and at 100 it fits two. A developer can set it with
        // `EVOLUTION_WARP_CARVEOUT`.
        let carveout = crate::kernel::solver_setting("CARVEOUT", 100) as c_int;
        cu.check(
            (cu.func_set_attribute)(function, 9, carveout),
            "cuFuncSetAttribute",
        )?;
        let mut blocks = 0;
        cu.check(
            (cu.occupancy)(&mut blocks, function, crate::kernel::BLOCK as c_int, 0),
            "cuOccupancyMaxActiveBlocksPerMultiprocessor",
        )?;
        if std::env::var_os("EVOLUTION_CUDA_VERBOSE").is_some() {
            // CU_FUNC_ATTRIBUTE_NUM_REGS, _SHARED_SIZE_BYTES, _LOCAL_SIZE_BYTES
            let attribute = |which: c_int| {
                let mut value = 0;
                (cu.func_get_attribute)(&mut value, which, function);
                value
            };
            eprintln!(
                "CUDA: {} kernel: {} registers, {} B shared, {} B local per thread, {} blocks of {} threads per SM, module loaded in {:.3} s",
                if key.record { "recording" } else { "scoring" },
                attribute(4),
                attribute(1),
                attribute(3),
                blocks,
                crate::kernel::BLOCK,
                load_seconds
            );
        }
        Ok(Kernel {
            module,
            function,
            blocks_per_sm: blocks.max(1) as u32,
        })
    }
}

/// A kernel from the disk cache or NVRTC, loaded into `context`. A damaged
/// cache entry fails to load: the kernel is compiled again.
fn compile_and_load(
    api: &Api,
    context: CuContext,
    options: &[String],
    key: KernelKey,
) -> Result<Kernel> {
    let cubin = compile_kernel(api, options, key, true)?;
    match load_kernel(api, context, &cubin, key) {
        Ok(kernel) => Ok(kernel),
        Err(_) => load_kernel(
            api,
            context,
            &compile_kernel(api, options, key, false)?,
            key,
        ),
    }
}

/// Counts the time an engine thread spends getting a kernel.
struct KernelWait(Instant);

impl Drop for KernelWait {
    fn drop(&mut self) {
        KERNEL_WAIT_NANOS.fetch_add(self.0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        KERNEL_WAITERS.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Kernels that background threads compile, in the order they are wanted. The
/// kernels of the world the engine runs ("wanted") compile on up to three
/// threads at once, and a thread loads each into the context after it
/// compiles, because `cuModuleLoadData` waits for the kernels running on the
/// GPU: on the engine thread it would hold up submissions for up to a second.
/// The kernels of the worlds one effect level away ("idle") only compile, on
/// one thread at nice 19, into the disk cache, so the next button press loads
/// them in milliseconds. A load by a background thread holds the driver
/// while it waits for the GPU, which blocks every call of the engine thread
/// with it, so an idle load was a drain of the whole pipeline: with the
/// neighbours of a world not yet in the cache (a new build, or a cache that
/// holds fewer kernels than the neighbours), 3 to 9 of them in every 10 s for
/// minutes, 4 to 8% of the creatures per second. The kernels of the wild
/// islands' worlds ("warm") are not loaded either. They compile into the disk
/// cache at the start, on as many threads as `warm_threads` allows.
struct Prefetch {
    api: Arc<Api>,
    /// The engine's context (an address, so the struct is `Send`).
    context: usize,
    /// The NVRTC options of the device, from `nvrtc_options`.
    options: Vec<String>,
    state: Mutex<PrefetchState>,
    /// Notified when a thread finishes a kernel or exits.
    ready: Condvar,
}

/// The queues and counts that `Prefetch` keeps behind its lock.
#[derive(Default)]
struct PrefetchState {
    /// Kernels of the world the engine runs, in the order they are needed.
    wanted: VecDeque<KernelKey>,
    /// Kernels of the worlds one effect level away.
    idle: VecDeque<KernelKey>,
    /// Kernels of the islands' worlds that compile into the disk cache at
    /// the start, on several threads, and are not loaded.
    warm: VecDeque<KernelKey>,
    /// Kernels a thread is on, with the number of wanted, idle and warm
    /// threads on each.
    running: HashMap<KernelKey, [usize; 3]>,
    /// The loaded kernels, or the message of the error, that wanted threads
    /// finished, until the engine takes them.
    done: HashMap<KernelKey, std::result::Result<Kernel, String>>,
    /// The order wanted kernels finished in, oldest first (entries of
    /// kernels already taken stay until they reach the front).
    finished: VecDeque<KernelKey>,
    /// Background threads alive: wanted ones, idle ones, warm ones.
    workers: [usize; 3],
    /// Set by `close`: threads take no new kernel, and queued ones are gone.
    closed: bool,
}

/// Most wanted threads at once. A world queues four kernels: scoring and
/// recording, each at the standard and the fine physics.
const WANTED_THREADS: usize = 3;

/// Most threads that compile the islands' kernels into the cache at the
/// start. The machine is idle then, and the window waits for them.
fn warm_threads() -> usize {
    std::thread::available_parallelism()
        .map_or(4, |n| n.get())
        .saturating_sub(2)
        .clamp(2, 5)
}

/// Most loaded kernels waiting to be used. Each module holds device memory,
/// so the oldest go when more finish.
const READY_KERNELS: usize = 48;

/// Lowers the calling thread's priority to the least (Linux: per thread).
fn nice_idle() {
    #[cfg(target_os = "linux")]
    unsafe {
        let tid = libc::syscall(libc::SYS_gettid) as libc::id_t;
        libc::setpriority(libc::PRIO_PROCESS, tid, 19);
    }
}

impl Prefetch {
    fn new(api: Arc<Api>, context: CuContext, options: Vec<String>) -> Arc<Self> {
        Arc::new(Self {
            api,
            context: context as usize,
            options,
            state: Mutex::default(),
            ready: Condvar::new(),
        })
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, PrefetchState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Compiles queued kernels, and for the wanted ones loads them, until the
    /// queue is empty or the engine closes. Kind 0 threads take the wanted
    /// queue, kind 1 (idle) the idle queue and kind 2 (warm) the warm queue.
    fn work(&self, kind: usize) {
        loop {
            let key = {
                let mut state = self.lock();
                let next = if state.closed {
                    None
                } else {
                    match kind {
                        0 => state.wanted.pop_front(),
                        1 => state.idle.pop_front(),
                        _ => state.warm.pop_front(),
                    }
                };
                let Some(key) = next else {
                    state.workers[kind] -= 1;
                    self.ready.notify_all();
                    return;
                };
                state.running.entry(key).or_default()[kind] += 1;
                key
            };
            if kind != 0 {
                // Compiled into the disk cache and not loaded (see `Prefetch`).
                let _ = compile_kernel(&self.api, &self.options, key, true);
                let mut state = self.lock();
                if let Some(counts) = state.running.get_mut(&key) {
                    counts[kind] -= 1;
                    if counts == &[0, 0, 0] {
                        state.running.remove(&key);
                    }
                }
                self.ready.notify_all();
                continue;
            }
            let result = compile_and_load(&self.api, self.context as CuContext, &self.options, key)
                .map_err(|e| format!("{e:#}"));
            let mut state = self.lock();
            if let Some(counts) = state.running.get_mut(&key) {
                counts[kind] -= 1;
                if counts == &[0, 0, 0] {
                    state.running.remove(&key);
                }
            }
            // A second thread on one kernel (a neighbour that became wanted)
            // leaves a module nobody needs. Unloading waits for the kernels
            // running on the GPU, so it happens outside the lock.
            let mut unused = Vec::new();
            if let Some(Ok(extra)) = state.done.insert(key, result) {
                unused.push(extra);
            }
            state.finished.push_back(key);
            while state.finished.len() > READY_KERNELS {
                let Some(oldest) = state.finished.pop_front() else {
                    break;
                };
                if let Some(Ok(old)) = state.done.remove(&oldest) {
                    unused.push(old);
                }
            }
            self.ready.notify_all();
            drop(state);
            for kernel in &unused {
                self.unload(kernel);
            }
        }
    }

    /// Unloads a module from the context. It waits for the kernels running on
    /// the GPU, so it must not run under the lock.
    fn unload(&self, kernel: &Kernel) {
        let cu = &self.api.cu;
        unsafe {
            (cu.ctx_set_current)(self.context as CuContext);
            (cu.module_unload)(kernel.module);
        }
    }

    /// The kernel, waiting if a thread is on it. None when nobody is: the
    /// caller compiles it (a queued job is taken off its queue).
    fn take(&self, key: KernelKey) -> Option<Result<Kernel>> {
        let mut state = self.lock();
        loop {
            if let Some(result) = state.done.remove(&key) {
                return Some(result.map_err(|e| anyhow::anyhow!(e)));
            }
            if state.running.contains_key(&key) {
                state = self.ready.wait(state).unwrap_or_else(|e| e.into_inner());
                continue;
            }
            state.wanted.retain(|k| *k != key);
            state.idle.retain(|k| *k != key);
            state.warm.retain(|k| *k != key);
            return None;
        }
    }

    /// The loaded kernels among `keys`, without waiting for any.
    fn take_ready(&self, keys: &[KernelKey]) -> Vec<(KernelKey, Kernel)> {
        let mut state = self.lock();
        keys.iter()
            .filter_map(|key| match state.done.remove(key) {
                Some(Ok(kernel)) => Some((*key, kernel)),
                Some(Err(error)) => {
                    // Left for `take`, which reports it.
                    state.done.insert(*key, Err(error));
                    None
                }
                None => None,
            })
            .collect()
    }

    /// Queues `wanted` and `idle` ahead of everything in their queues and
    /// `warm` behind it, each list in its own order, and starts threads. A
    /// kernel that is done, queued as wanted or already being compiled is
    /// skipped. The exception is an idle or warm kernel that is now wanted:
    /// it goes to the front of the wanted queue (a thread at nice 19 may be
    /// slow, so a wanted thread compiles it too).
    fn enqueue(
        self: &Arc<Self>,
        wanted: Vec<KernelKey>,
        idle: Vec<KernelKey>,
        warm: Vec<KernelKey>,
    ) {
        let mut state = self.lock();
        if state.closed {
            return;
        }
        let fresh: Vec<KernelKey> = wanted
            .into_iter()
            .filter(|key| {
                if state.done.contains_key(key)
                    || state.wanted.contains(key)
                    || state.running.get(key).is_some_and(|counts| counts[0] > 0)
                {
                    return false;
                }
                state.idle.retain(|k| k != key);
                state.warm.retain(|k| k != key);
                true
            })
            .collect();
        for key in fresh.into_iter().rev() {
            crate::loading::queued(&kernel_label(key), crate::loading::Group::Needed);
            state.wanted.push_front(key);
        }
        // The newest world's neighbours go first: the player's next press is
        // one of them, not one of an earlier world's.
        let fresh: Vec<KernelKey> = idle
            .into_iter()
            .filter(|key| {
                !(state.done.contains_key(key)
                    || state.running.contains_key(key)
                    || state.wanted.contains(key))
            })
            .collect();
        state.idle.retain(|key| !fresh.contains(key));
        for key in fresh.into_iter().rev() {
            crate::loading::queued(&kernel_label(key), crate::loading::Group::Idle);
            state.idle.push_front(key);
        }
        // The islands' kernels go to the back of their queue, in order.
        let fresh: Vec<KernelKey> = warm
            .into_iter()
            .filter(|key| {
                !(state.done.contains_key(key)
                    || state.running.contains_key(key)
                    || state.wanted.contains(key)
                    || state.warm.contains(key))
            })
            .collect();
        for key in fresh {
            crate::loading::queued(&kernel_label(key), crate::loading::Group::Startup);
            state.warm.push_back(key);
        }
        for (kind, queued) in [state.wanted.len(), state.idle.len(), state.warm.len()]
            .into_iter()
            .enumerate()
        {
            let most = [WANTED_THREADS, 1, warm_threads()][kind];
            while state.workers[kind] < most && state.workers[kind] < queued {
                let prefetch = self.clone();
                let spawned = std::thread::Builder::new()
                    .name(["cuda-compile", "cuda-compile-idle", "cuda-compile-warm"][kind].into())
                    .spawn(move || {
                        // Compile beside the pool, not on the engine thread's CPU.
                        crate::threads::pin_pool();
                        if kind == 1 {
                            nice_idle();
                        }
                        prefetch.work(kind)
                    });
                if spawned.is_err() {
                    break;
                }
                state.workers[kind] += 1;
            }
        }
    }

    /// Stops the background threads once their current kernels finish, and
    /// waits for them, at most `patience`: a compiler thread still inside
    /// NVRTC when the process exits crashes it. Returns the kernels that
    /// were loaded and not used, for the caller to unload.
    fn close(&self, patience: Duration) -> Vec<Kernel> {
        let mut state = self.lock();
        state.closed = true;
        let queued: Vec<KernelKey> = [
            std::mem::take(&mut state.wanted),
            std::mem::take(&mut state.idle),
            std::mem::take(&mut state.warm),
        ]
        .into_iter()
        .flatten()
        .collect();
        for key in queued {
            crate::loading::cancel(&kernel_label(key));
        }
        let deadline = Instant::now() + patience;
        while state.workers.iter().sum::<usize>() > 0 {
            let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                break;
            };
            state = self
                .ready
                .wait_timeout(state, left)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        state.done.drain().filter_map(|(_, r)| r.ok()).collect()
    }
}

/// NVRTC options for a device of architecture `arch`.
fn nvrtc_options(arch: &str) -> Vec<String> {
    let mut options = vec![
        format!("--gpu-architecture={arch}"),
        "--std=c++17".into(),
        "--prec-div=false".into(),
        "--prec-sqrt=false".into(),
        "--fmad=true".into(),
        "--extra-device-vectorization".into(),
        "--ptxas-options=-v".into(),
    ];
    // A developer's extra options (`EVOLUTION_NVRTC_EXTRA`), such as
    // -lineinfo for a profiler.
    if let Ok(extra) = std::env::var("EVOLUTION_NVRTC_EXTRA") {
        options.extend(extra.split_whitespace().map(String::from));
    }
    options
}

impl CudaEngine {
    /// Opens the first CUDA device whose name contains `name`
    /// (case-insensitive) and starts compiling the kernels of the default
    /// world and of the wild islands' worlds in the background.
    /// `max_capacity` is the largest body, in nodes, the caller wants. The
    /// engine caps it at `kernel::MAX_NODES` and reports the result in its
    /// `max_capacity` field, and the scheduler sends it no larger bodies.
    pub fn new(name: &str, max_capacity: usize) -> Result<Self> {
        Self::open(name, max_capacity.min(crate::kernel::MAX_NODES))
    }

    fn open(name: &str, max_capacity: usize) -> Result<Self> {
        let api = api()?;
        let cu = &api.cu;
        unsafe {
            let mut count = 0;
            cu.check((cu.device_get_count)(&mut count), "cuDeviceGetCount")?;
            let wanted = name.to_lowercase();
            let mut found = None;
            for ordinal in 0..count {
                let mut device = 0;
                cu.check((cu.device_get)(&mut device, ordinal), "cuDeviceGet")?;
                let mut buffer = [0 as c_char; 256];
                cu.check(
                    (cu.device_get_name)(buffer.as_mut_ptr(), buffer.len() as c_int, device),
                    "cuDeviceGetName",
                )?;
                let device_name = CStr::from_ptr(buffer.as_ptr())
                    .to_string_lossy()
                    .into_owned();
                if device_name.to_lowercase().contains(&wanted) {
                    found = Some((device, device_name));
                    break;
                }
            }
            let (device, device_name) =
                found.with_context(|| format!("No CUDA device matching {name:?}"))?;
            let attribute = |attribute| -> Result<i32> {
                let mut value = 0;
                cu.check(
                    (cu.device_get_attribute)(&mut value, attribute, device),
                    "cuDeviceGetAttribute",
                )?;
                Ok(value)
            };
            let arch = format!(
                "sm_{}{}",
                attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?,
                attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?
            );
            let multiprocessors = attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
            let mut context = std::ptr::null_mut();
            cu.check(
                (cu.primary_ctx_retain)(&mut context, device),
                "cuDevicePrimaryCtxRetain",
            )?;
            if let Err(error) = cu.check((cu.ctx_set_current)(context), "cuCtxSetCurrent") {
                (cu.primary_ctx_release)(device);
                return Err(error);
            }
            // Host buffers mapped on this thread register with the context.
            REGISTER.with(|register| *register.borrow_mut() = Some(api.clone()));
            let options = nvrtc_options(&arch);
            let mut engine = Self {
                name: String::new(),
                api: api.clone(),
                device,
                context,
                arch,
                multiprocessors,
                kernels: HashMap::new(),
                prefetch: Prefetch::new(api.clone(), context, options),
                worlds: HashSet::new(),
                slots: Vec::new(),
                next_ticket: 0,
                max_capacity,
                allocated_bytes: 0,
                last_gpu_seconds: 0.0,
                standard: crate::engine::gpu_slots() as usize,
            };
            engine.name = format!("{device_name} (CUDA)");
            // Standard slots, one for confirmation trials and one for replays.
            for index in 0..engine.standard + 2 {
                let slot = engine.create_slot(index >= engine.standard)?;
                engine.slots.push(slot);
            }
            static EVICT: std::sync::Once = std::sync::Once::new();
            EVICT.call_once(|| {
                if let Some(dir) = kernel_cache_dir() {
                    evict_cache(&dir);
                }
            });
            // The game starts in the default world, and its 100 wild islands
            // each run in a world of their own. Their scoring kernels compile
            // into the disk cache now, on several threads, because a block
            // of an island waits for its kernels and 100 worlds compile one
            // after another in the first generations otherwise. The loading
            // screen follows these jobs.
            crate::loading::begin_startup();
            engine.prefetch_world(&Config::default());
            engine.prefetch_islands();
            crate::loading::end_startup();
            Ok(engine)
        }
    }

    /// Makes a submission slot with its main stream and its events. The wave
    /// streams come later, in `ensure_buffers`. The streams of a
    /// `high_priority` slot, which is a confirmation or a replay slot, have
    /// the highest priority, so a confirmation or a recording gets the GPU's
    /// blocks ahead of queued scoring work. A priority out of range is clamped
    /// to the greatest one.
    fn create_slot(&self, high_priority: bool) -> Result<Slot> {
        let cu = &self.api.cu;
        unsafe {
            let mut main = std::ptr::null_mut();
            cu.check(
                (cu.stream_create)(
                    &mut main,
                    CU_STREAM_NON_BLOCKING,
                    stream_priority(high_priority),
                ),
                "cuStreamCreateWithPriority",
            )?;
            let event = |flags| -> Result<CuEvent> {
                let mut event = std::ptr::null_mut();
                cu.check((cu.event_create)(&mut event, flags), "cuEventCreate")?;
                Ok(event)
            };
            Ok(Slot {
                main,
                streams: Vec::new(),
                joins: Vec::new(),
                start: event(0)?,
                stop: event(0)?,
                done: event(CU_EVENT_DISABLE_TIMING)?,
                groups: Vec::new(),
                counters: None,
                readback: None,
                frames: None,
                pending: None,
            })
        }
    }

    /// NVRTC options for this device.
    fn options(&self) -> Vec<String> {
        nvrtc_options(&self.arch)
    }

    /// The kernel for `key`, compiled and loaded on this thread.
    fn load_here(&self, key: KernelKey) -> Result<Kernel> {
        compile_and_load(&self.api, self.context, &self.options(), key)
    }

    /// Queues the kernels of `cfg`'s world on the background compiler:
    /// scoring at the standard physics first (the engine waits for these),
    /// then scoring at the fine physics of the confirmation trials, then the
    /// recordings. A world with a fidelity of its own, neither standard nor
    /// fine, queues its kernels too. After them, at idle priority, come the
    /// worlds one effect level away, compiled into the disk cache so the next
    /// button press loads its kernels in milliseconds: the scoring kernels at
    /// the standard physics of all of them first, then at the fine physics,
    /// then the recordings at the standard physics. A world already queued is
    /// skipped. Kernels of this world that have finished loading become the
    /// engine's.
    fn prefetch_world(&mut self, cfg: &Config) {
        let flags = crate::kernel::world_flags(cfg);
        let mut fidelities = vec![Fidelity::standard(), Fidelity::fine()];
        if !fidelities.contains(&cfg.fidelity()) {
            fidelities.push(cfg.fidelity());
        }
        // Kernels of a world in the order of (record, fidelity) steps.
        let steps: Vec<(bool, Fidelity)> = [false, true]
            .into_iter()
            .flat_map(|record| fidelities.iter().map(move |&f| (record, f)))
            .collect();
        let keys = |flags: u32, steps: &[(bool, Fidelity)]| -> Vec<KernelKey> {
            steps
                .iter()
                .flat_map(|&(record, fidelity)| {
                    crate::kernel::CLASSES
                        .into_iter()
                        .map(move |class| KernelKey {
                            record,
                            class,
                            flags,
                            fidelity,
                        })
                })
                .collect()
        };
        let levels: Vec<u8> = crate::environment::EFFECTS
            .iter()
            .map(|effect| effect.level(cfg) as u8)
            .collect();
        let mut world = keys(flags, &steps);
        if self.worlds.insert((levels, flags)) {
            let wanted = world
                .iter()
                .copied()
                .filter(|key| !self.kernels.contains_key(key))
                .collect();
            let mut near = Vec::new();
            for neighbour in crate::environment::one_level_away(cfg) {
                let neighbour_flags = crate::kernel::world_flags(&neighbour);
                if neighbour_flags != flags && !near.contains(&neighbour_flags) {
                    near.push(neighbour_flags);
                }
            }
            // Standard scoring, fine scoring, then recordings at the
            // standard physics, each across all the neighbours.
            let mut idle = Vec::new();
            for step in [
                &steps[..1],
                &steps[1..2],
                &steps[fidelities.len()..fidelities.len() + 1],
            ] {
                for &neighbour_flags in &near {
                    idle.extend(
                        keys(neighbour_flags, step)
                            .into_iter()
                            .filter(|key| !self.kernels.contains_key(key)),
                    );
                }
            }
            self.prefetch.enqueue(wanted, idle, Vec::new());
        }
        // Kernels of this world that finished loading are the engine's from
        // now on (the background threads drop the ones nobody takes).
        world.retain(|key| !self.kernels.contains_key(key));
        for (key, kernel) in self.prefetch.take_ready(&world) {
            self.kernels.insert(key, kernel);
        }
    }

    /// Queues the scoring kernels of the wild islands' worlds
    /// (`environment::wild_levels`, the same 100 in every game) to compile
    /// into the disk cache. A cached kernel loads in milliseconds when its
    /// island first needs it.
    fn prefetch_islands(&mut self) {
        let base = Config::default();
        let mut seen = HashSet::new();
        let mut keys = Vec::new();
        for levels in crate::environment::wild_levels(0) {
            let cfg = crate::environment::wild_world(&base, &levels);
            let (flags, fidelity) = (crate::kernel::world_flags(&cfg), cfg.fidelity());
            if seen.insert((flags, fidelity)) {
                keys.extend(crate::kernel::CLASSES.into_iter().map(|class| KernelKey {
                    record: false,
                    class,
                    flags,
                    fidelity,
                }));
            }
        }
        keys.retain(|key| !self.kernels.contains_key(key));
        self.prefetch.enqueue(Vec::new(), Vec::new(), keys);
    }

    /// The function of the kernel for `key` and its resident blocks per
    /// multiprocessor. The kernel comes from the background compiler if it
    /// has it or is on it, else it is compiled here. A wait counts in
    /// `kernel_wait_seconds`.
    fn kernel(&mut self, key: KernelKey) -> Result<(CuFunction, u32)> {
        if !self.kernels.contains_key(&key) {
            let started = Instant::now();
            KERNEL_WAITERS.fetch_add(1, Ordering::Relaxed);
            let _waiting = KernelWait(started);
            let kernel = match self.prefetch.take(key) {
                Some(result) => result?,
                None => self.load_here(key)?,
            };
            if std::env::var_os("EVOLUTION_CUDA_VERBOSE").is_some() {
                eprintln!(
                    "CUDA: {key:?} ready after {:.2} s with NVRTC {} ({})",
                    started.elapsed().as_secs_f64(),
                    self.api.nvrtc.version_string(),
                    self.api.nvrtc.path.display()
                );
            }
            self.kernels.insert(key, kernel);
        }
        let kernel = &self.kernels[&key];
        Ok((kernel.function, kernel.blocks_per_sm))
    }

    /// Streaming multiprocessors on the device.
    pub fn multiprocessors(&self) -> i32 {
        self.multiprocessors
    }

    fn alloc_device(&self, bytes: usize) -> Result<DeviceBuf> {
        let size = buffer_size(bytes);
        let mut ptr = 0;
        self.api.cu.check(
            unsafe { (self.api.cu.mem_alloc)(&mut ptr, size) },
            "cuMemAlloc",
        )?;
        Ok(DeviceBuf { ptr, size })
    }

    fn free_device(&self, buf: DeviceBuf) {
        unsafe { (self.api.cu.mem_free)(buf.ptr) };
    }

    fn alloc_host(&self, bytes: usize) -> Result<HostBuf> {
        let size = buffer_size(bytes);
        let mut ptr = std::ptr::null_mut();
        self.api.cu.check(
            unsafe { (self.api.cu.mem_alloc_host)(&mut ptr, size) },
            "cuMemAllocHost",
        )?;
        Ok(HostBuf {
            ptr: ptr as *mut u8,
            size,
        })
    }

    fn free_host(&self, buf: HostBuf) {
        unsafe { (self.api.cu.mem_free_host)(buf.ptr as *mut c_void) };
    }

    /// Frees the device buffers of batch `group` in `slot`.
    fn drop_group(&mut self, slot: usize, group: usize) {
        if let Some(res) = self.slots[slot].groups[group].take() {
            for buf in res.bufs {
                self.free_device(buf);
            }
        }
    }

    /// Bytes each device buffer of a batch needs, in the order of
    /// `GroupRes::bufs`: lanes, muscles, ends, heads, results.
    fn needs(batch: &LaneBatch) -> Result<[usize; 5]> {
        let wave = batch
            .wave
            .as_ref()
            .context("The CUDA kernel needs a batch from kernel::pack")?;
        Ok([
            std::mem::size_of_val(&*wave.lanes),
            std::mem::size_of_val(&*wave.muscles),
            std::mem::size_of_val(&*wave.ends),
            std::mem::size_of_val(&*wave.heads),
            batch.slots.len() * std::mem::size_of::<GpuResult>(),
        ])
    }

    /// Grows the slot's device buffers, wave streams, counters and readback
    /// buffer to fit `batches` in `waves` waves. A recording also gets a frame
    /// buffer of `frame_bytes`. A buffer that is large enough stays.
    fn ensure_buffers(
        &mut self,
        slot: usize,
        batches: &[LaneBatch],
        waves: usize,
        frame_bytes: usize,
    ) -> Result<()> {
        let api = self.api.clone();
        let cu = &api.cu;
        while self.slots[slot].streams.len() < waves.min(STREAM_LIMIT) {
            unsafe {
                let mut stream = std::ptr::null_mut();
                cu.check(
                    (cu.stream_create)(
                        &mut stream,
                        CU_STREAM_NON_BLOCKING,
                        stream_priority(slot >= self.standard),
                    ),
                    "cuStreamCreateWithPriority",
                )?;
                let mut event = std::ptr::null_mut();
                cu.check(
                    (cu.event_create)(&mut event, CU_EVENT_DISABLE_TIMING),
                    "cuEventCreate",
                )?;
                self.slots[slot].streams.push(stream);
                self.slots[slot].joins.push(event);
            }
        }
        if self.slots[slot].groups.len() < batches.len() {
            self.slots[slot].groups.resize_with(batches.len(), || None);
        }
        let mut readback = frame_bytes;
        for (group, batch) in batches.iter().enumerate() {
            let need = Self::needs(batch)?;
            readback += need[4];
            if let Some(res) = &self.slots[slot].groups[group]
                && res.bufs.iter().zip(need).all(|(h, n)| h.size >= n)
            {
                continue;
            }
            self.drop_group(slot, group);
            // Nothing leaks when an allocation fails part way.
            let mut made = Vec::with_capacity(need.len());
            for bytes in need {
                match self.alloc_device(bytes) {
                    Ok(buf) => made.push(buf),
                    Err(error) => {
                        for buf in made {
                            self.free_device(buf);
                        }
                        return Err(error);
                    }
                }
            }
            let Ok(bufs) = <[DeviceBuf; 5]>::try_from(made) else {
                unreachable!("one buffer per binding");
            };
            self.slots[slot].groups[group] = Some(GroupRes { bufs });
        }
        // One counter per wave: the next creature a thread of the wave takes.
        let counter_bytes = 4 * waves;
        if self.slots[slot]
            .counters
            .as_ref()
            .is_none_or(|b| b.size < counter_bytes)
        {
            if let Some(old) = self.slots[slot].counters.take() {
                self.free_device(old);
            }
            self.slots[slot].counters = Some(self.alloc_device(counter_bytes)?);
        }
        if frame_bytes > 0
            && self.slots[slot]
                .frames
                .as_ref()
                .is_none_or(|b| b.size < frame_bytes)
        {
            if let Some(old) = self.slots[slot].frames.take() {
                self.free_device(old);
            }
            self.slots[slot].frames = Some(self.alloc_device(frame_bytes)?);
        }
        if self.slots[slot]
            .readback
            .as_ref()
            .is_none_or(|b| b.size < readback)
        {
            if let Some(old) = self.slots[slot].readback.take() {
                self.free_host(old);
            }
            self.slots[slot].readback = Some(self.alloc_host(readback)?);
        }
        self.recount_allocated();
        Ok(())
    }

    /// The free standard slot for a unit of `batches` (`pick_standard_slot`).
    fn standard_slot_for(&self, batches: &[LaneBatch]) -> Result<usize> {
        let creatures: usize = batches.iter().map(|b| b.slots.len()).sum();
        let needs: Vec<[usize; 5]> = batches.iter().map(Self::needs).collect::<Result<_>>()?;
        let views: Vec<SlotView> = self
            .standard_slots()
            .map(|i| {
                let slot = &self.slots[i];
                SlotView {
                    free: slot.pending.is_none(),
                    holds: needs.iter().enumerate().all(|(group, need)| {
                        slot.groups
                            .get(group)
                            .and_then(Option::as_ref)
                            .is_some_and(|res| {
                                res.bufs.iter().zip(need).all(|(buf, n)| buf.size >= *n)
                            })
                    }),
                    bytes: Self::slot_bytes(slot),
                }
            })
            .collect();
        pick_standard_slot(&views, BIG_SLOTS, creatures > BIG_UNIT)
            .context("No free GPU submission slot")
    }

    /// Bytes of host memory a batch's buffers hold.
    pub fn held_bytes(batch: &LaneBatch) -> usize {
        batch.wave.as_ref().map_or(0, |w| {
            w.lanes.held_bytes()
                + w.muscles.held_bytes()
                + w.ends.held_bytes()
                + w.heads.held_bytes()
        })
    }

    /// Bytes of device and pinned host buffers a slot keeps for reuse, and
    /// of the host buffers of the unit it runs.
    fn slot_bytes(slot: &Slot) -> u64 {
        slot.groups
            .iter()
            .flatten()
            .flat_map(|g| g.bufs.iter())
            .map(|b| b.size as u64)
            .sum::<u64>()
            + slot.counters.as_ref().map_or(0, |b| b.size as u64)
            + slot.pending.as_ref().map_or(0, |p| {
                p.batches
                    .iter()
                    .map(|b| Self::held_bytes(b) as u64)
                    .sum::<u64>()
            })
            + slot.readback.as_ref().map_or(0, |b| b.size as u64)
            + slot.frames.as_ref().map_or(0, |b| b.size as u64)
    }

    /// Recomputes `allocated_bytes` from the slots.
    fn recount_allocated(&mut self) {
        self.allocated_bytes = self.slots.iter().map(Self::slot_bytes).sum();
    }

    /// Frees the buffers that idle slots keep for reuse, so a submission that
    /// ran out of memory can try again. Returns the bytes freed.
    pub fn release_idle(&mut self) -> u64 {
        self.recount_allocated();
        let before = self.allocated_bytes;
        for slot in 0..self.slots.len() {
            if self.slots[slot].pending.is_some() {
                continue;
            }
            for group in 0..self.slots[slot].groups.len() {
                self.drop_group(slot, group);
            }
            if let Some(b) = self.slots[slot].counters.take() {
                self.free_device(b);
            }
            if let Some(b) = self.slots[slot].readback.take() {
                self.free_host(b);
            }
            if let Some(b) = self.slots[slot].frames.take() {
                self.free_device(b);
            }
        }
        self.recount_allocated();
        before.saturating_sub(self.allocated_bytes)
    }

    /// Number of standard submissions that can be queued without waiting.
    pub fn free_slots(&self) -> usize {
        self.standard_slots()
            .filter(|&i| self.slots[i].pending.is_none())
            .count()
    }

    /// Number of standard submissions of `creatures` creatures that can be
    /// queued without waiting. A big unit (`BIG_UNIT`) needs one of the first
    /// `BIG_SLOTS` slots, and a small one takes any.
    pub fn free_slots_for(&self, creatures: usize) -> usize {
        let slots = if creatures > BIG_UNIT {
            0..BIG_SLOTS.min(self.standard)
        } else {
            self.standard_slots()
        };
        slots.filter(|&i| self.slots[i].pending.is_none()).count()
    }

    /// Whether a confirmation trial can be queued without waiting: its slot
    /// is free, or a standard slot is.
    pub fn confirm_free(&self) -> bool {
        self.slots[self.confirm_slot()].pending.is_none() || self.free_slots() > 0
    }

    /// Submissions in flight, replays included.
    pub fn in_flight(&self) -> usize {
        self.slots.iter().filter(|s| s.pending.is_some()).count()
    }

    fn standard_slots(&self) -> std::ops::Range<usize> {
        0..self.standard
    }

    fn confirm_slot(&self) -> usize {
        self.standard
    }

    fn replay_slot(&self) -> usize {
        self.standard + 1
    }

    /// Whether a replay can be recorded now.
    pub fn replay_free(&self) -> bool {
        self.slots[self.replay_slot()].pending.is_none()
    }

    /// Queues the whole trial of the one creature in `batch` on the replay
    /// slot with the recording kernel, which writes every frame. The result
    /// arrives through `poll` with `Completed::frames`. The trial is the one
    /// `submit` scores, computed the same way. Returns the ticket.
    pub fn record(&mut self, batch: LaneBatch, cfg: &Config) -> Result<u64> {
        ensure!(self.replay_free(), "A replay is already being recorded");
        self.submit_as(&mut vec![batch], cfg, true)
    }

    /// Uploads the batches and queues their whole trials, without waiting.
    /// On success the engine takes the batches, copies straight from their
    /// buffers, and hands them back with the results (`Completed::batches`)
    /// for the next unit to reuse; on failure they stay with the caller.
    /// Returns the ticket that `poll` reports with the results.
    pub fn submit(&mut self, batches: &mut Vec<LaneBatch>, cfg: &Config) -> Result<u64> {
        self.submit_as(batches, cfg, false)
    }

    /// What `submit` and `record` share: launches the batches, and on success
    /// keeps them on their slot as a `Pending` submission and returns its
    /// ticket.
    fn submit_as(
        &mut self,
        batches: &mut Vec<LaneBatch>,
        cfg: &Config,
        record: bool,
    ) -> Result<u64> {
        let mut uploading = false;
        match self.launch(batches, cfg, record, &mut uploading) {
            Ok((slot, frames)) => {
                let ticket = self.next_ticket;
                self.next_ticket += 1;
                let batches = std::mem::take(batches);
                self.slots[slot].pending = Some(Pending {
                    ticket,
                    result_count: batches.iter().map(|b| b.slots.len()).sum(),
                    batches,
                    frames,
                });
                self.recount_allocated();
                Ok(ticket)
            }
            Err(error) => {
                if uploading {
                    // Copies from the caller's buffers may still run: let
                    // them finish before the caller can reuse or free them.
                    unsafe { (self.api.cu.ctx_synchronize)() };
                }
                Err(error)
            }
        }
    }

    /// Queues the uploads, trials and readback of `batches` on a free slot.
    /// Returns the slot and, for a recording, where its frames are read
    /// back. `uploading` turns true once copies from the batches are queued.
    fn launch(
        &mut self,
        batches: &[LaneBatch],
        cfg: &Config,
        record: bool,
        uploading: &mut bool,
    ) -> Result<(usize, Option<(usize, usize)>)> {
        ensure!(!batches.is_empty(), "Empty GPU batch");
        let flags = crate::kernel::world_flags(cfg);
        let fidelity = cfg.fidelity();
        self.prefetch_world(cfg);
        let kernels: Vec<(CuFunction, u32)> = batches
            .iter()
            .map(|batch| {
                self.kernel(KernelKey {
                    record,
                    class: batch.capacity,
                    flags,
                    fidelity,
                })
            })
            .collect::<Result<_>>()?;
        // A recording has the replay slot. A confirmation trial has its own
        // slot when that is free. Every other unit takes the free standard
        // slot with the most buffers to reuse: when memory is short, a new
        // allocation may fail where reuse does not.
        let confirming = !record && crate::engine::is_confirmation(cfg);
        let slot = if record {
            self.replay_slot()
        } else if confirming && self.slots[self.confirm_slot()].pending.is_none() {
            self.confirm_slot()
        } else {
            self.standard_slot_for(batches)?
        };
        // Waves of at most `kernel::WAVE` creatures: (batch, first creature,
        // count).
        let mut waves: Vec<(usize, usize, usize)> = Vec::new();
        for (b, batch) in batches.iter().enumerate() {
            let count = batch.slots.len();
            let mut at = 0;
            while at < count {
                let n = (count - at).min(crate::kernel::WAVE);
                waves.push((b, at, n));
                at += n;
            }
        }
        let total = (fidelity.settle() + cfg.steps()) as usize;
        let stride = batches.first().map_or(0, creature_kernel::frame_stride);
        let frame_count = if record {
            ensure!(
                batches.len() == 1 && batches[0].slots.len() == 1,
                "A recording holds one creature"
            );
            stride * (total + 1)
        } else {
            0
        };
        let frame_bytes = frame_count * std::mem::size_of::<[f32; 2]>();
        let buffers = self.ensure_buffers(slot, batches, waves.len(), frame_bytes);
        self.recount_allocated();
        buffers?;
        let cu = &self.api.cu;
        let resources = &self.slots[slot];
        let results_bytes =
            batches.iter().map(|b| b.slots.len()).sum::<usize>() * std::mem::size_of::<GpuResult>();
        unsafe {
            cu.check((cu.ctx_set_current)(self.context), "cuCtxSetCurrent")?;
            // Copy every upload to the device on the main stream, straight
            // from the batch's registered buffers.
            *uploading = true;
            let copy = |data: &[u8], dst: CuDevicePtr| -> Result<()> {
                if data.is_empty() {
                    return Ok(());
                }
                cu.check(
                    (cu.memcpy_htod_async)(
                        dst,
                        data.as_ptr() as *const c_void,
                        data.len(),
                        resources.main,
                    ),
                    "cuMemcpyHtoDAsync",
                )
            };
            for (group, batch) in batches.iter().enumerate() {
                let res = resources.groups[group].as_ref().unwrap();
                let wave = batch.wave.as_ref().unwrap();
                copy(bytemuck::cast_slice(&wave.lanes[..]), res.bufs[0].ptr)?;
                copy(bytemuck::cast_slice(&wave.muscles[..]), res.bufs[1].ptr)?;
                copy(bytemuck::cast_slice(&wave.ends[..]), res.bufs[2].ptr)?;
                copy(bytemuck::cast_slice(&wave.heads[..]), res.bufs[3].ptr)?;
            }
            let counters = resources.counters.as_ref().unwrap();
            cu.check(
                (cu.memset_d32_async)(counters.ptr, 0, waves.len(), resources.main),
                "cuMemsetD32Async",
            )?;
            cu.check(
                (cu.event_record)(resources.start, resources.main),
                "cuEventRecord",
            )?;
            let streams = waves.len().min(resources.streams.len());
            for &stream in &resources.streams[..streams] {
                cu.check(
                    (cu.stream_wait_event)(stream, resources.start, 0),
                    "cuStreamWaitEvent",
                )?;
            }
            for (w, &(b, first, count)) in waves.iter().enumerate() {
                let res = resources.groups[b].as_ref().unwrap();
                let (kernel, blocks_per_sm) = kernels[b];
                let mut params = crate::kernel::params(cfg, first, count, stride);
                // A thread runs one creature at a time and takes the next
                // from the wave's counter, so the resident blocks are enough.
                let blocks = wave_blocks(
                    count,
                    blocks_per_sm as usize * self.multiprocessors as usize,
                );
                let mut pointers = [
                    res.bufs[0].ptr,
                    res.bufs[1].ptr,
                    res.bufs[2].ptr,
                    res.bufs[3].ptr,
                    res.bufs[4].ptr,
                    counters.ptr + (4 * w) as u64,
                ];
                // The frames pointer is the eighth argument. Only the
                // recording kernel declares it.
                let mut frames = resources.frames.as_ref().map_or(0, |f| f.ptr);
                let mut args: [*mut c_void; 8] = [
                    &mut pointers[0] as *mut u64 as *mut c_void,
                    &mut pointers[1] as *mut u64 as *mut c_void,
                    &mut pointers[2] as *mut u64 as *mut c_void,
                    &mut pointers[3] as *mut u64 as *mut c_void,
                    &mut pointers[4] as *mut u64 as *mut c_void,
                    &mut pointers[5] as *mut u64 as *mut c_void,
                    &mut params as *mut crate::kernel::Params as *mut c_void,
                    &mut frames as *mut u64 as *mut c_void,
                ];
                cu.check(
                    (cu.launch_kernel)(
                        kernel,
                        blocks as c_uint,
                        1,
                        1,
                        crate::kernel::BLOCK,
                        1,
                        1,
                        0,
                        resources.streams[w % streams],
                        args.as_mut_ptr(),
                        std::ptr::null_mut(),
                    ),
                    "cuLaunchKernel",
                )?;
            }
            for b in 0..streams {
                cu.check(
                    (cu.event_record)(resources.joins[b], resources.streams[b]),
                    "cuEventRecord",
                )?;
                cu.check(
                    (cu.stream_wait_event)(resources.main, resources.joins[b], 0),
                    "cuStreamWaitEvent",
                )?;
            }
            cu.check(
                (cu.event_record)(resources.stop, resources.main),
                "cuEventRecord",
            )?;
            let readback = resources.readback.as_ref().unwrap();
            let mut offset = 0usize;
            let mut read = |src: CuDevicePtr, bytes: usize| -> Result<()> {
                if bytes == 0 {
                    return Ok(());
                }
                assert!(offset + bytes <= readback.size);
                cu.check(
                    (cu.memcpy_dtoh_async)(
                        readback.ptr.add(offset) as *mut c_void,
                        src,
                        bytes,
                        resources.main,
                    ),
                    "cuMemcpyDtoHAsync",
                )?;
                offset += bytes;
                Ok(())
            };
            for (group, batch) in batches.iter().enumerate() {
                let res = resources.groups[group].as_ref().unwrap();
                read(
                    res.bufs[4].ptr,
                    batch.slots.len() * std::mem::size_of::<GpuResult>(),
                )?;
            }
            if record {
                read(
                    resources.frames.as_ref().expect("frames buffer").ptr,
                    frame_bytes,
                )?;
            }
            cu.check(
                (cu.event_record)(resources.done, resources.main),
                "cuEventRecord",
            )?;
        }
        Ok((slot, record.then_some((results_bytes, frame_count))))
    }

    /// Returns the results of the oldest finished submission, waiting up to
    /// `timeout` for one. It returns `None` when nothing is in flight or
    /// nothing finishes in time.
    pub fn poll(&mut self, timeout: Duration) -> Result<Option<Completed>> {
        let mut pending: Vec<(u64, usize)> = self
            .slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.pending.as_ref().map(|p| (p.ticket, i)))
            .collect();
        if pending.is_empty() {
            return Ok(None);
        }
        pending.sort_unstable();
        let cu = &self.api.cu;
        let deadline = Instant::now() + timeout;
        let slot = loop {
            let mut finished = None;
            for &(_, i) in &pending {
                match unsafe { (cu.event_query)(self.slots[i].done) } {
                    CUDA_SUCCESS => {
                        finished = Some(i);
                        break;
                    }
                    CUDA_ERROR_NOT_READY => {}
                    error => cu.check(error, "cuEventQuery")?,
                }
            }
            if let Some(slot) = finished {
                break slot;
            }
            let now = Instant::now();
            if now >= deadline {
                return Ok(None);
            }
            std::thread::sleep((deadline - now).min(Duration::from_micros(200)));
        };
        let resources = &self.slots[slot];
        let mut milliseconds = 0f32;
        let gpu_seconds = if unsafe {
            (cu.event_elapsed_time)(&mut milliseconds, resources.start, resources.stop)
        } == CUDA_SUCCESS
        {
            f64::from(milliseconds) * 1e-3
        } else {
            0.0
        };
        let pending = self.slots[slot].pending.take().unwrap();
        let readback = self.slots[slot].readback.as_ref().unwrap();
        unsafe {
            let flat: &[GpuResult] =
                std::slice::from_raw_parts(readback.ptr as *const GpuResult, pending.result_count);
            // Batch results in unit order.
            let mut results = vec![GpuResult::default(); pending.result_count];
            let mut start = 0;
            for batch in &pending.batches {
                let end = start + batch.slots.len();
                for (&slot, result) in batch.slots.iter().zip(&flat[start..end]) {
                    results[slot] = *result;
                }
                start = end;
            }
            let frames = pending.frames.map(|(offset, count)| {
                std::slice::from_raw_parts(readback.ptr.add(offset) as *const [f32; 2], count)
                    .to_vec()
            });
            self.last_gpu_seconds = gpu_seconds;
            self.recount_allocated();
            Ok(Some(Completed {
                ticket: pending.ticket,
                results,
                batches: pending.batches,
                frames,
                gpu_seconds,
            }))
        }
    }
}

impl Drop for CudaEngine {
    fn drop(&mut self) {
        let unused = self.prefetch.close(Duration::from_secs(30));
        let api = self.api.clone();
        let cu = &api.cu;
        unsafe {
            (cu.ctx_set_current)(self.context);
            (cu.ctx_synchronize)();
            for kernel in &unused {
                (cu.module_unload)(kernel.module);
            }
        }
        for slot in 0..self.slots.len() {
            // Dropping a pending unit unregisters its host memory, which
            // needs the context alive.
            self.slots[slot].pending.take();
            for group in 0..self.slots[slot].groups.len() {
                self.drop_group(slot, group);
            }
            if let Some(b) = self.slots[slot].counters.take() {
                self.free_device(b);
            }
            if let Some(b) = self.slots[slot].readback.take() {
                self.free_host(b);
            }
            if let Some(b) = self.slots[slot].frames.take() {
                self.free_device(b);
            }
        }
        unsafe {
            for slot in &self.slots {
                for &stream in &slot.streams {
                    (cu.stream_destroy)(stream);
                }
                for &event in slot
                    .joins
                    .iter()
                    .chain([&slot.start, &slot.stop, &slot.done])
                {
                    (cu.event_destroy)(event);
                }
                (cu.stream_destroy)(slot.main);
            }
            for kernel in self.kernels.values() {
                (cu.module_unload)(kernel.module);
            }
            REGISTER.with(|register| register.borrow_mut().take());
            (cu.primary_ctx_release)(self.device);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The kernels of the starting worlds (the default world's, and the
    /// scoring kernels of the wild islands' worlds) must fit the cache twice
    /// over, or every start would delete some and compile them again.
    #[test]
    fn the_starting_kernels_fit_the_cache_twice_over() {
        let base = Config::default();
        let mut worlds = HashSet::new();
        for levels in crate::environment::wild_levels(0) {
            let cfg = crate::environment::wild_world(&base, &levels);
            worlds.insert((crate::kernel::world_flags(&cfg), cfg.fidelity()));
        }
        // The default world: scoring and recording at both fidelities.
        let default = 4 * crate::kernel::CLASSES.len();
        let kernels = worlds.len() * crate::kernel::CLASSES.len() + default;
        assert!(
            worlds.len() > 25,
            "{} worlds: are the wild worlds there?",
            worlds.len()
        );
        assert!(
            2 * kernels <= CACHE_FILES,
            "{kernels} kernels, cache {CACHE_FILES}"
        );
    }

    fn view(free: bool, holds: bool, bytes: u64) -> SlotView {
        SlotView { free, holds, bytes }
    }

    /// Big units use the first slots only, and small units use the others
    /// first, the smallest that holds them, so that no small unit makes a
    /// slot big.
    #[test]
    fn units_take_the_slots_of_their_size() {
        // Two big slots (the first two) and three small ones.
        let views = [
            view(true, true, 90),
            view(true, true, 100),
            view(true, true, 1),
            view(true, true, 2),
            view(true, false, 0),
        ];
        // A big unit takes the smallest big slot that holds it.
        assert_eq!(pick_standard_slot(&views, 2, true), Some(0));
        // A small one takes the smallest small slot that holds it.
        assert_eq!(pick_standard_slot(&views, 2, false), Some(2));
        // When no small slot is free it takes a big one.
        let busy = [
            view(true, true, 90),
            view(true, true, 100),
            view(false, true, 1),
            view(false, true, 2),
            view(false, false, 0),
        ];
        assert_eq!(pick_standard_slot(&busy, 2, false), Some(0));
        // A big unit never takes a small slot, even if that is all there is.
        let small_only = [
            view(false, true, 90),
            view(false, true, 100),
            view(true, true, 1),
            view(true, true, 2),
            view(true, false, 0),
        ];
        assert_eq!(pick_standard_slot(&small_only, 2, true), None);
        // When no free slot holds the unit, the free slot with the most
        // buffers grows.
        let short = [
            view(true, false, 60),
            view(true, false, 70),
            view(true, false, 1),
            view(true, false, 2),
            view(true, false, 0),
        ];
        assert_eq!(pick_standard_slot(&short, 2, true), Some(1));
        assert_eq!(pick_standard_slot(&short, 2, false), Some(3));
        // With fewer slots than big slots every slot is a big slot.
        assert_eq!(pick_standard_slot(&short[..1], 2, false), Some(0));
    }

    /// A big wave leaves the reserved block slots free, and a small one takes
    /// a block for every 128 creatures.
    #[test]
    fn a_big_wave_leaves_block_slots_free() {
        assert_eq!(wave_blocks(1, 48), 1);
        assert_eq!(wave_blocks(250, 48), 2);
        assert_eq!(wave_blocks(5_000, 48), 40);
        assert_eq!(wave_blocks(100_000, 48), 48 - RESERVED_BLOCKS);
        assert_eq!(wave_blocks(100_000, 1), 1);
    }

    /// The engine asks for 32 queues when the environment has no count, and a
    /// developer's count stays.
    #[test]
    fn a_developers_queue_count_stays() {
        assert_eq!(queues_to_set(None), Some("32"));
        assert_eq!(queues_to_set(Some(std::ffi::OsStr::new("8"))), None);
    }

    /// Eviction deletes the oldest kernels beyond `CACHE_FILES`, and leaves
    /// the newest ones and other files alone.
    #[test]
    fn eviction_keeps_the_newest_kernels() {
        let dir = std::env::temp_dir().join(format!("evolution-cache-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let start = std::time::SystemTime::now() - Duration::from_secs(10_000);
        for i in 0..CACHE_FILES + 5 {
            let path = dir.join(format!("{i:04}.cubin"));
            std::fs::write(&path, b"x").unwrap();
            let file = std::fs::File::open(&path).unwrap();
            file.set_modified(start + Duration::from_secs(i as u64))
                .unwrap();
        }
        std::fs::write(dir.join("other.txt"), b"x").unwrap();
        evict_cache(&dir);
        let mut names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        assert_eq!(names.len(), CACHE_FILES + 1);
        assert!(names.contains(&"other.txt".to_string()));
        assert!(!names.contains(&"0004.cubin".to_string()));
        assert!(names.contains(&"0005.cubin".to_string()));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
