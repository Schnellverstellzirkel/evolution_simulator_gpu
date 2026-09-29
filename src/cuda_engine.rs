//! NVIDIA GPU backend.
//!
//! It runs `shaders/physics_creature.cu`, a CUDA C++ port of the WGSL
//! creature kernel, on the same `LaneBatch` packing, `Params` and trial
//! segments as `VkEngine`, and it keeps `VkEngine`'s submit and poll contract.
//! The reason for it is register control: Vulkan offers no way to cap
//! registers per thread, and above 128 registers an SM holds 12 one-warp
//! workgroups instead of 16 (`docs/phase0-measurements.md`).
//!
//! Nothing CUDA is linked at build time. The driver API (`libcuda`) and NVRTC
//! (`libnvrtc`) are loaded when the engine opens, so the game builds and runs
//! unchanged on machines without them; `engine::gpu_engine` then falls back to
//! Vulkan. NVRTC comes from the system CUDA toolkit or from NVIDIA's pip
//! wheel; see `docs/building.md`. Kernels compile to a cubin for the device's
//! architecture when the engine opens, one per node capacity, and on first
//! use for other fidelities.
//!
//! It is the default on NVIDIA GPUs: `engine::gpu_engine` opens it whenever
//! the driver and NVRTC load, and uses Vulkan otherwise. Developer overrides,
//! never needed to play: `EVOLUTION_CUDA=0` keeps Vulkan,
//! `EVOLUTION_CUDA_MAXREG` caps registers per thread (default 128, 0 for the
//! compiler's choice), `EVOLUTION_CUDA_WG` fixes threads per block (by
//! default each capacity gets the size with the most resident warps),
//! `EVOLUTION_CUDA_STREAMS` limits streams per unit, `EVOLUTION_NVRTC` names
//! the NVRTC library, `EVOLUTION_CUDA_FLAGS` adds NVRTC options, and
//! `EVOLUTION_CUDA_VERBOSE` reports compile times.
use crate::{
    config::Config,
    creature_kernel::{self, CAPACITIES, GpuResult, LaneBatch},
    physics::Fidelity,
    vk_engine::Completed,
};
use anyhow::{Context, Result, bail, ensure};
use std::{
    collections::{HashMap, HashSet, VecDeque},
    ffi::{CStr, CString, c_char, c_int, c_uint, c_void},
    path::{Path, PathBuf},
    sync::{Arc, Condvar, Mutex},
    time::{Duration, Instant},
};

type CuResult = c_int;
type CuDevice = c_int;
type CuContext = *mut c_void;
type CuModule = *mut c_void;
type CuFunction = *mut c_void;
type CuStream = *mut c_void;
type CuEvent = *mut c_void;
type CuDevicePtr = u64;
type NvrtcProgram = *mut c_void;

const CUDA_SUCCESS: CuResult = 0;
const CUDA_ERROR_OUT_OF_MEMORY: CuResult = 2;
const CUDA_ERROR_NOT_READY: CuResult = 600;
const CU_STREAM_NON_BLOCKING: c_uint = 1;
const CU_EVENT_DISABLE_TIMING: c_uint = 2;
const CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: c_int = 16;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
const CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: c_int = 76;
const CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES: c_int = 1;
const CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES: c_int = 3;
const CU_FUNC_ATTRIBUTE_NUM_REGS: c_int = 4;

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
    func_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, CuFunction) -> CuResult,
    occupancy: unsafe extern "C" fn(*mut c_int, CuFunction, c_int, usize) -> CuResult,
    mem_alloc: unsafe extern "C" fn(*mut CuDevicePtr, usize) -> CuResult,
    mem_free: unsafe extern "C" fn(CuDevicePtr) -> CuResult,
    mem_alloc_host: unsafe extern "C" fn(*mut *mut c_void, usize) -> CuResult,
    mem_free_host: unsafe extern "C" fn(*mut c_void) -> CuResult,
    memcpy_htod_async:
        unsafe extern "C" fn(CuDevicePtr, *const c_void, usize, CuStream) -> CuResult,
    memcpy_dtoh_async: unsafe extern "C" fn(*mut c_void, CuDevicePtr, usize, CuStream) -> CuResult,
    stream_create: unsafe extern "C" fn(*mut CuStream, c_uint) -> CuResult,
    stream_destroy: unsafe extern "C" fn(CuStream) -> CuResult,
    stream_wait_event: unsafe extern "C" fn(CuStream, CuEvent, c_uint) -> CuResult,
    event_create: unsafe extern "C" fn(*mut CuEvent, c_uint) -> CuResult,
    event_destroy: unsafe extern "C" fn(CuEvent) -> CuResult,
    event_record: unsafe extern "C" fn(CuEvent, CuStream) -> CuResult,
    event_query: unsafe extern "C" fn(CuEvent) -> CuResult,
    event_elapsed_time: unsafe extern "C" fn(*mut f32, CuEvent, CuEvent) -> CuResult,
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
                func_get_attribute: symbol!(library, "cuFuncGetAttribute"),
                occupancy: symbol!(library, "cuOccupancyMaxActiveBlocksPerMultiprocessor"),
                mem_alloc: symbol!(library, "cuMemAlloc_v2"),
                mem_free: symbol!(library, "cuMemFree_v2"),
                mem_alloc_host: symbol!(library, "cuMemAllocHost_v2"),
                mem_free_host: symbol!(library, "cuMemFreeHost"),
                memcpy_htod_async: symbol!(library, "cuMemcpyHtoDAsync_v2"),
                memcpy_dtoh_async: symbol!(library, "cuMemcpyDtoHAsync_v2"),
                stream_create: symbol!(library, "cuStreamCreate"),
                stream_destroy: symbol!(library, "cuStreamDestroy_v2"),
                stream_wait_event: symbol!(library, "cuStreamWaitEvent"),
                event_create: symbol!(library, "cuEventCreate"),
                event_destroy: symbol!(library, "cuEventDestroy_v2"),
                event_record: symbol!(library, "cuEventRecord"),
                event_query: symbol!(library, "cuEventQuery"),
                event_elapsed_time: symbol!(library, "cuEventElapsedTime"),
                launch_kernel: symbol!(library, "cuLaunchKernel"),
                get_error_name: symbol!(library, "cuGetErrorName"),
                _library: library,
            })
        }
    }

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

/// Candidate NVRTC libraries: `EVOLUTION_NVRTC`, the loader's search path
/// (a system CUDA toolkit), then NVIDIA's pip wheels in the virtual
/// environment that `docs/building.md` sets up.
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

static API: std::sync::OnceLock<Result<Arc<Api>, String>> = std::sync::OnceLock::new();

fn api() -> Result<Arc<Api>> {
    API.get_or_init(|| {
        let cu = Driver::load().map_err(|e| format!("{e:#}"))?;
        cu.check(unsafe { (cu.init)(0) }, "cuInit")
            .map_err(|e| format!("{e:#}"))?;
        let nvrtc = Nvrtc::load().map_err(|e| format!("{e:#}"))?;
        Ok(Arc::new(Api { cu, nvrtc }))
    })
    .clone()
    .map_err(|e| anyhow::anyhow!(e))
}

/// Whether the GPU engine tries CUDA before Vulkan. It does unless the
/// developer override `EVOLUTION_CUDA` is `0`, `false` or `off`.
pub fn enabled() -> bool {
    !std::env::var("EVOLUTION_CUDA").is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off"
        )
    })
}

/// Whether `EVOLUTION_CUDA` asks for CUDA (`1`, `true` or `on`), so tests can
/// refuse a Vulkan fallback.
pub fn forced() -> bool {
    std::env::var("EVOLUTION_CUDA").is_ok_and(|value| {
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "on"
        )
    })
}

/// Register cap per thread: `EVOLUTION_CUDA_MAXREG`, 0 for the compiler's
/// choice, default 128 (physics v2: 0, see docs/performance-log.md). At 128 registers an SM holds 16 warps of every
/// kernel up to 8 nodes. The compiler's own choice is 155 to 255 registers
/// (8 to 12 warps) and measured slower, and caps of 80 and 96 add spills that
/// cost more than their extra warps gain (docs/performance-log.md).
pub fn register_cap() -> Option<u32> {
    let cap = std::env::var("EVOLUTION_CUDA_MAXREG")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .unwrap_or(if physics2() { 0 } else { 128 });
    (cap > 0).then(|| cap.clamp(24, 255))
}

/// Threads per block (`EVOLUTION_CUDA_WG`: 32, 64 or 128), or `None` to
/// choose per node capacity (the default). Each block holds one creature per
/// thread, and CUDA reserves 1 KB of shared memory per block, which larger
/// blocks share.
pub fn workgroup_size() -> Option<u32> {
    std::env::var("EVOLUTION_CUDA_WG")
        .ok()
        .and_then(|v| v.trim().parse::<u32>().ok())
        .filter(|v| matches!(v, 32 | 64 | 128))
}

/// Streams per submission slot (`EVOLUTION_CUDA_STREAMS`, default 16). Batch
/// b of a unit runs on stream b modulo this; 1 runs a unit's batches one
/// after another.
fn stream_limit() -> usize {
    std::env::var("EVOLUTION_CUDA_STREAMS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .unwrap_or(16)
}

/// Whether the three shared node arrays of a `threads`-thread block fit the
/// 48 KB a block may declare statically.
fn fits(capacity: usize, threads: u32) -> bool {
    threads == 32 || shared_per_thread(capacity) * threads as usize <= 48 * 1024
}

/// Whether the kernels in use are physics v2's.
fn physics2() -> bool {
    crate::physics2::enabled()
}

/// Shared memory per thread in bytes: v1's three node arrays, or v2's
/// per-lane table (bodies above 32 nodes keep it in local memory).
fn shared_per_thread(capacity: usize) -> usize {
    if !physics2() {
        24 * capacity
    } else if creature_kernel::cuda_table_local(capacity) {
        0
    } else {
        4 * (10 * capacity - 6)
    }
}

/// The recording kernel's source for `capacity` nodes.
fn recording_source(
    capacity: usize,
    threads: u32,
    fidelity: Fidelity,
    launch_bounds: bool,
) -> String {
    if physics2() {
        creature_kernel::cuda_record_source2(capacity, threads, fidelity, launch_bounds)
    } else {
        creature_kernel::cuda_record_source(capacity, threads, fidelity, launch_bounds)
    }
}

/// The scoring kernel's source for `capacity` nodes.
fn scoring_source(
    capacity: usize,
    threads: u32,
    fidelity: Fidelity,
    launch_bounds: bool,
) -> String {
    if physics2() {
        creature_kernel::cuda_source2(capacity, threads, fidelity, launch_bounds)
    } else {
        creature_kernel::cuda_source(capacity, threads, fidelity, launch_bounds)
    }
}

/// Resident warps per SM for `capacity`-node kernels in `threads`-thread
/// blocks at `registers` per thread, by the occupancy rule measured in
/// `docs/phase0-measurements.md` plus the 1 KB of shared memory CUDA reserves
/// per block. It matches the driver's occupancy calculator for every kernel
/// `examples/cuda_stats.rs` reports.
fn predicted_warps(capacity: usize, threads: u32, registers: u32) -> u32 {
    let warps_per_block = threads / 32;
    let register_warps = 4 * (16_384 / (32 * registers.next_multiple_of(8)));
    let shared_blocks = 102_400 / (shared_per_thread(capacity) as u32 * threads + 1024);
    let blocks = (register_warps / warps_per_block)
        .min(shared_blocks)
        .min(24)
        .min(48 / warps_per_block);
    blocks * warps_per_block
}

/// Threads per block for `capacity`-node kernels: `requested`, halved until
/// it fits, or else the block size with the most resident warps, and the
/// largest of equals. At 128 registers, 128-thread blocks measured faster
/// than 32-thread blocks at equal occupancy (docs/performance-log.md).
fn block_size(capacity: usize, requested: Option<u32>, registers: Option<u32>) -> u32 {
    if let Some(mut threads) = requested {
        while !fits(capacity, threads) {
            threads /= 2;
        }
        return threads;
    }
    // Uncapped, the v2 kernels use 222 to 255 registers up to 32 nodes.
    let registers = registers.unwrap_or(if physics2() { 255 } else { 128 });
    [128, 64, 32]
        .into_iter()
        .filter(|&t| fits(capacity, t))
        .max_by_key(|&t| (predicted_warps(capacity, t, registers), t))
        .unwrap_or(32)
}

/// What the compiler made of one kernel, for `examples/cuda_stats.rs`.
#[derive(Clone, Debug)]
pub struct KernelStats {
    pub registers: i32,
    /// Local memory per thread in bytes (stack frame and spills).
    pub local_bytes: i32,
    pub shared_bytes: i32,
    /// Threads per block.
    pub threads: u32,
    /// Resident warps per SM by the driver's occupancy calculator.
    pub warps_per_sm: i32,
    /// The ptxas lines of the compiler log.
    pub log: String,
}

struct Kernel {
    module: CuModule,
    function: CuFunction,
    /// Threads per block (`block_size`).
    threads: u32,
}

struct DeviceBuf {
    ptr: CuDevicePtr,
    size: usize,
}

struct HostBuf {
    ptr: *mut u8,
    size: usize,
}

struct GroupRes {
    nodes: DeviceBuf,
    muscles: DeviceBuf,
    bones: DeviceBuf,
    results: DeviceBuf,
    info: DeviceBuf,
    tiles: DeviceBuf,
}

/// Per-submission resources, as in `VkEngine`. Each batch of a unit runs on
/// its own stream, so its step ranges follow one another while other batches
/// fill the SMs beside it; the slot's main stream uploads, joins the batch
/// streams, and reads the results back.
struct Slot {
    main: CuStream,
    streams: Vec<CuStream>,
    joins: Vec<CuEvent>,
    start: CuEvent,
    stop: CuEvent,
    done: CuEvent,
    groups: Vec<Option<GroupRes>>,
    staging: Option<HostBuf>,
    readback: Option<HostBuf>,
    /// Recorded replay frames; only the replay slot has one.
    frames: Option<DeviceBuf>,
    pending: Option<Pending>,
}

struct Pending {
    ticket: u64,
    layout: Vec<(Vec<usize>, Vec<usize>)>,
    result_count: usize,
    state: Option<Vec<(usize, usize)>>,
    /// Recorded frames: their byte offset in the readback buffer and their
    /// count of node positions.
    frames: Option<(usize, usize)>,
}

pub struct CudaEngine {
    api: Arc<Api>,
    device: CuDevice,
    context: CuContext,
    pub name: String,
    arch: String,
    multiprocessors: i32,
    /// Scoring kernels by fidelity and node capacity.
    kernels: HashMap<(Fidelity, usize), Kernel>,
    /// Recording kernels (`creature_kernel::cuda_record_source`) by fidelity
    /// and node capacity.
    recording: HashMap<(Fidelity, usize), Kernel>,
    /// Kernels compiling on background threads.
    prefetch: Arc<Prefetch>,
    /// Submission slots. The last one is kept for replays, with streams of
    /// its own, so a replay never waits behind evaluation.
    slots: Vec<Slot>,
    next_ticket: u64,
    workgroup: Option<u32>,
    max_registers: Option<u32>,
    pub max_capacity: usize,
    pub allocated_bytes: u64,
    pub last_gpu_seconds: f64,
}

// The context and every mapped pointer are only used by the thread that owns
// the engine; it makes the context current on that thread when it opens.
unsafe impl Send for CudaEngine {}

/// Rounds a buffer size up the way `VkEngine` does, so buffers are reused
/// across units of slightly different sizes.
fn buffer_size(bytes: usize) -> usize {
    crate::vk_engine::padded_size(bytes as u64) as usize
}

/// A kernel to compile: recording or scoring, fidelity, node capacity.
type KernelKey = (bool, Fidelity, usize);

/// NVRTC's source for one kernel, compiled to a cubin. It needs no CUDA
/// context, so any thread may run it. Compiled kernels are kept on disk
/// (`kernel_cache_dir`), keyed by a hash of the source, the options, the
/// NVRTC version and the GPU's architecture, so a later start loads them in
/// milliseconds. With `use_cache` false the compile ignores an entry (it
/// may be damaged) and writes a fresh one.
fn compile_kernel(
    api: &Api,
    options: &[String],
    (record, fidelity, capacity): KernelKey,
    threads: u32,
    launch_bounds: bool,
    use_cache: bool,
) -> Result<Vec<u8>> {
    let source = if record {
        recording_source(capacity, threads, fidelity, launch_bounds)
    } else {
        scoring_source(capacity, threads, fidelity, launch_bounds)
    };
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
        && let Some(bytes) = path.as_ref().and_then(|p| std::fs::read(p).ok())
        && !bytes.is_empty()
    {
        return Ok(bytes);
    }
    let cubin = api
        .nvrtc
        .compile(&source, options)
        .with_context(|| format!("{capacity}-node CUDA kernel"))
        .map(|(cubin, _)| cubin)?;
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
    Ok(cubin)
}

/// Where compiled kernels are kept: `EVOLUTION_KERNEL_CACHE`, else the
/// user's cache directory, else nowhere.
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

/// Kernels that background threads compile, in the order they are wanted.
/// Compiling from a cold compiler cache takes seconds for a 3 to 8 node
/// kernel, 25 s for 16 nodes, and much more above that.
#[derive(Default)]
struct Prefetch {
    state: Mutex<PrefetchState>,
    ready: Condvar,
}

#[derive(Default)]
struct PrefetchState {
    queue: VecDeque<(KernelKey, u32)>,
    running: HashSet<KernelKey>,
    done: HashMap<KernelKey, std::result::Result<Vec<u8>, String>>,
    /// Background threads alive.
    workers: usize,
    closed: bool,
}

impl Prefetch {
    /// Compiles queued kernels until the queue is empty or the engine closes.
    fn work(&self, api: &Api, options: &[String], launch_bounds: bool) {
        loop {
            let (key, threads) = {
                let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                if state.closed {
                    state.workers -= 1;
                    self.ready.notify_all();
                    return;
                }
                let Some(job) = state.queue.pop_front() else {
                    state.workers -= 1;
                    self.ready.notify_all();
                    return;
                };
                state.running.insert(job.0);
                job
            };
            let result = compile_kernel(api, options, key, threads, launch_bounds, true)
                .map_err(|e| format!("{e:#}"));
            let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
            state.running.remove(&key);
            state.done.insert(key, result);
            self.ready.notify_all();
        }
    }

    /// The compiled kernel, waiting if a thread is on it. None when nobody
    /// is: the caller compiles it (a queued job is taken off the queue).
    fn take(&self, key: KernelKey) -> Option<Result<Vec<u8>>> {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(result) = state.done.remove(&key) {
                return Some(result.map_err(|e| anyhow::anyhow!(e)));
            }
            if state.running.contains(&key) {
                state = self.ready.wait(state).unwrap_or_else(|e| e.into_inner());
                continue;
            }
            state.queue.retain(|(k, _)| *k != key);
            return None;
        }
    }

    /// Queues `jobs` not already queued, running or done, and starts
    /// threads (at most three at a time) to compile them.
    fn enqueue(
        self: &Arc<Self>,
        jobs: Vec<(KernelKey, u32)>,
        api: &Arc<Api>,
        options: &Arc<Vec<String>>,
        launch_bounds: bool,
    ) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.closed {
            return;
        }
        for (key, threads) in jobs {
            let known = state.done.contains_key(&key)
                || state.running.contains(&key)
                || state.queue.iter().any(|(k, _)| *k == key);
            if !known {
                state.queue.push_back((key, threads));
            }
        }
        while state.workers < 3 && state.workers < state.queue.len() + state.running.len() {
            let (prefetch, api, options) = (self.clone(), api.clone(), options.clone());
            let spawned = std::thread::Builder::new()
                .name("cuda-compile".into())
                .spawn(move || prefetch.work(&api, &options, launch_bounds));
            if spawned.is_err() {
                break;
            }
            state.workers += 1;
        }
    }

    /// Stops the background threads once their current kernels finish, and
    /// waits for them, at most `patience`: a compiler thread still inside
    /// NVRTC when the process exits crashes it.
    fn close(&self, patience: Duration) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        state.closed = true;
        state.queue.clear();
        let deadline = Instant::now() + patience;
        while state.workers > 0 {
            let Some(left) = deadline.checked_duration_since(Instant::now()) else {
                break;
            };
            state = self
                .ready
                .wait_timeout(state, left)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }
}

impl CudaEngine {
    /// Opens the first CUDA device whose name contains `name`
    /// (case-insensitive) and compiles the standard kernels for bodies up to
    /// `max_capacity` nodes.
    pub fn new(name: &str, max_capacity: usize) -> Result<Self> {
        Self::open(name, max_capacity, register_cap(), true)
    }

    /// Opens the device without compiling kernels, for statistics.
    pub fn open_for_stats(name: &str) -> Result<Self> {
        Self::open(name, 64, None, false)
    }

    fn open(
        name: &str,
        max_capacity: usize,
        max_registers: Option<u32>,
        build: bool,
    ) -> Result<Self> {
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
            let workgroup = workgroup_size();
            let mut engine = Self {
                name: String::new(),
                api: api.clone(),
                device,
                context,
                arch,
                multiprocessors,
                kernels: HashMap::new(),
                recording: HashMap::new(),
                prefetch: Arc::new(Prefetch::default()),
                slots: Vec::new(),
                next_ticket: 0,
                workgroup,
                max_registers,
                max_capacity,
                allocated_bytes: 0,
                last_gpu_seconds: 0.0,
            };
            let threads = workgroup.map_or_else(|| "auto".into(), |n| n.to_string());
            engine.name = match max_registers {
                Some(n) => format!("{device_name} (CUDA, {n} registers, blocks {threads})"),
                None => format!("{device_name} (CUDA, blocks {threads})"),
            };
            // Evaluation slots plus one for replays.
            for _ in 0..crate::vk_engine::gpu_slots() + 1 {
                let slot = engine.create_slot()?;
                engine.slots.push(slot);
            }
            if build {
                engine.start_prefetch();
            }
            Ok(engine)
        }
    }

    fn create_slot(&self) -> Result<Slot> {
        let cu = &self.api.cu;
        unsafe {
            let mut main = std::ptr::null_mut();
            cu.check(
                (cu.stream_create)(&mut main, CU_STREAM_NON_BLOCKING),
                "cuStreamCreate",
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
                staging: None,
                readback: None,
                frames: None,
                pending: None,
            })
        }
    }

    /// NVRTC options for this device and register cap.
    fn options(&self, max_registers: Option<u32>) -> Vec<String> {
        let mut options = vec![
            format!("--gpu-architecture={}", self.arch),
            "--std=c++17".into(),
            // Division and square roots as the Vulkan driver compiles WGSL:
            // approximate, not IEEE rounded.
            "--prec-div=false".into(),
            "--prec-sqrt=false".into(),
            "--fmad=true".into(),
            "--ptxas-options=-v".into(),
        ];
        if let Some(n) = max_registers {
            options.push(format!("--maxrregcount={n}"));
        }
        if let Ok(extra) = std::env::var("EVOLUTION_CUDA_FLAGS") {
            options.extend(extra.split_whitespace().map(str::to_owned));
        }
        options
    }

    /// Compiles `source` in NVRTC (no context needed).
    fn compile(api: &Api, source: &str, options: &[String]) -> Result<(Vec<u8>, String)> {
        api.nvrtc.compile(source, options)
    }

    /// Loads a cubin, built for `threads`-thread blocks, into this engine's
    /// context.
    fn load(&self, cubin: &[u8], threads: u32) -> Result<Kernel> {
        let cu = &self.api.cu;
        unsafe {
            cu.check((cu.ctx_set_current)(self.context), "cuCtxSetCurrent")?;
            let mut module = std::ptr::null_mut();
            cu.check(
                (cu.module_load_data)(&mut module, cubin.as_ptr() as *const c_void),
                "cuModuleLoadData",
            )?;
            let mut function = std::ptr::null_mut();
            if let Err(error) = cu.check(
                (cu.module_get_function)(&mut function, module, c"advance".as_ptr()),
                "cuModuleGetFunction",
            ) {
                (cu.module_unload)(module);
                return Err(error);
            }
            Ok(Kernel {
                module,
                function,
                threads,
            })
        }
    }

    /// Resident warps per SM of `kernel`, by the driver's occupancy calculator.
    fn resident_warps(&self, kernel: &Kernel) -> Result<i32> {
        let cu = &self.api.cu;
        let mut blocks = 0;
        cu.check(
            unsafe { (cu.occupancy)(&mut blocks, kernel.function, kernel.threads as c_int, 0) },
            "cuOccupancyMaxActiveBlocksPerMultiprocessor",
        )?;
        Ok(blocks * (kernel.threads as i32 / 32))
    }

    /// The compile jobs for `capacities`, in the order they are wanted: the
    /// scoring kernels at the standard fidelity, the recording kernels for
    /// replays, then the fine-check kernels.
    fn jobs_for(&self, capacities: &[usize]) -> Vec<(KernelKey, u32)> {
        let mut jobs = Vec::new();
        for (record, fidelity) in [
            (false, Fidelity::standard()),
            (true, Fidelity::standard()),
            (false, Fidelity::fine()),
        ] {
            for &capacity in capacities {
                let built = if record {
                    self.recording.contains_key(&(fidelity, capacity))
                } else {
                    self.kernels.contains_key(&(fidelity, capacity))
                };
                if !built {
                    let threads = block_size(capacity, self.workgroup, self.max_registers);
                    jobs.push(((record, fidelity, capacity), threads));
                }
            }
        }
        jobs
    }

    /// Queues `capacities` on the background compiler.
    fn prefetch_capacities(&self, capacities: &[usize]) {
        let jobs = self.jobs_for(capacities);
        let options = Arc::new(self.options(self.max_registers));
        self.prefetch
            .enqueue(jobs, &self.api, &options, self.max_registers.is_none());
    }

    /// Starts compiling the small bodies' kernels (up to 8 nodes) on
    /// background threads, so the first units wait for what they need and
    /// no more: from a cold compiler cache a 3 to 8 node kernel takes 2 to 7
    /// s, 16 nodes 25 s, and more above. Larger kernels start when the
    /// next size below them is first used (`build_kernel`), or on demand.
    fn start_prefetch(&self) {
        let small: Vec<usize> = CAPACITIES
            .iter()
            .copied()
            .filter(|&c| c <= 8 && c <= self.max_capacity)
            .collect();
        self.prefetch_capacities(&small);
    }

    /// Builds the kernel for `capacity` at `fidelity` (scoring, or with
    /// `record` recording): from the background compiler if it has it or is
    /// on it, else here.
    fn build_kernel(&mut self, fidelity: Fidelity, capacity: usize, record: bool) -> Result<()> {
        let max_registers = self.max_registers;
        let threads = block_size(capacity, self.workgroup, max_registers);
        let started = Instant::now();
        let key = (record, fidelity, capacity);
        let compile = |engine: &Self, use_cache: bool| {
            compile_kernel(
                &engine.api,
                &engine.options(max_registers),
                key,
                threads,
                max_registers.is_none(),
                use_cache,
            )
        };
        let cubin = match self.prefetch.take(key) {
            Some(result) => result?,
            None => compile(self, true)?,
        };
        // A damaged cache entry fails to load: compile it again.
        let kernel = match self.load(&cubin, threads) {
            Ok(kernel) => kernel,
            Err(_) => self.load(&compile(self, false)?, threads)?,
        };
        if record {
            self.recording.insert((fidelity, capacity), kernel);
        } else {
            self.kernels.insert((fidelity, capacity), kernel);
        }
        // Bodies grow: start the next size up in the background.
        if capacity >= 8
            && let Some(&next) = CAPACITIES
                .iter()
                .find(|&&c| c > capacity && c <= self.max_capacity)
        {
            self.prefetch_capacities(&[next]);
        }
        if std::env::var_os("EVOLUTION_CUDA_VERBOSE").is_some() {
            eprintln!(
                "CUDA: {} kernel for {capacity} nodes at {fidelity:?} ready after {:.2} s with NVRTC {} ({})",
                if record { "recording" } else { "scoring" },
                started.elapsed().as_secs_f64(),
                self.api.nvrtc.version_string(),
                self.api.nvrtc.path.display()
            );
        }
        Ok(())
    }

    /// Compiles the kernel for `capacity` at `fidelity` with `max_registers`,
    /// in the block size the engine would use, and reports what the compiler
    /// made of it. The kernel is not kept.
    pub fn kernel_stats(
        &self,
        capacity: usize,
        fidelity: Fidelity,
        max_registers: Option<u32>,
    ) -> Result<KernelStats> {
        let threads = block_size(capacity, self.workgroup, max_registers);
        let source = scoring_source(capacity, threads, fidelity, max_registers.is_none());
        let (cubin, log) = Self::compile(&self.api, &source, &self.options(max_registers))?;
        let kernel = self.load(&cubin, threads)?;
        let cu = &self.api.cu;
        let attribute = |attribute| -> Result<i32> {
            let mut value = 0;
            cu.check(
                unsafe { (cu.func_get_attribute)(&mut value, attribute, kernel.function) },
                "cuFuncGetAttribute",
            )?;
            Ok(value)
        };
        let stats = (|| -> Result<KernelStats> {
            let warps = self.resident_warps(&kernel)?;
            Ok(KernelStats {
                registers: attribute(CU_FUNC_ATTRIBUTE_NUM_REGS)?,
                local_bytes: attribute(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)?,
                shared_bytes: attribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)?,
                threads: kernel.threads,
                warps_per_sm: warps,
                log: log
                    .lines()
                    .filter(|l| l.contains("ptxas"))
                    .map(|l| l.trim().to_owned())
                    .collect::<Vec<_>>()
                    .join("\n"),
            })
        })();
        unsafe { (cu.module_unload)(kernel.module) };
        stats
    }

    /// Streaming multiprocessors on the device.
    pub fn multiprocessors(&self) -> i32 {
        self.multiprocessors
    }

    /// The scoring or, with `record`, recording kernel for `capacity` at
    /// `fidelity`, and its block size.
    fn kernel(
        &mut self,
        fidelity: Fidelity,
        capacity: usize,
        record: bool,
    ) -> Result<(CuFunction, u32)> {
        let key = (fidelity, capacity);
        let missing = if record {
            !self.recording.contains_key(&key)
        } else {
            !self.kernels.contains_key(&key)
        };
        if missing {
            self.build_kernel(fidelity, capacity, record)?;
        }
        let kernel = if record {
            self.recording.get(&key)
        } else {
            self.kernels.get(&key)
        }
        .context("CUDA kernel missing")?;
        Ok((kernel.function, kernel.threads))
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

    fn drop_group(&mut self, slot: usize, group: usize) {
        if let Some(res) = self.slots[slot].groups[group].take() {
            for buf in [
                res.nodes,
                res.muscles,
                res.bones,
                res.results,
                res.info,
                res.tiles,
            ] {
                self.free_device(buf);
            }
        }
    }

    /// Grows the slot's device buffers, streams and host buffers to fit
    /// `batches`. Returns the upload size.
    fn ensure_buffers(
        &mut self,
        slot: usize,
        batches: &[LaneBatch],
        read_state: bool,
        frame_bytes: usize,
    ) -> Result<usize> {
        let api = self.api.clone();
        let cu = &api.cu;
        while self.slots[slot].streams.len() < batches.len().min(stream_limit()) {
            unsafe {
                let mut stream = std::ptr::null_mut();
                cu.check(
                    (cu.stream_create)(&mut stream, CU_STREAM_NON_BLOCKING),
                    "cuStreamCreate",
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
        let mut upload = 0usize;
        let mut readback = 0usize;
        for (group, batch) in batches.iter().enumerate() {
            let need = [
                std::mem::size_of_val(batch.nodes.as_slice()),
                std::mem::size_of_val(batch.muscles.as_slice()),
                std::mem::size_of_val(batch.bones.as_slice()),
                batch.info.len() * std::mem::size_of::<GpuResult>(),
                std::mem::size_of_val(batch.info.as_slice()),
                std::mem::size_of_val(batch.tiles.as_slice()),
            ];
            upload += need.iter().sum::<usize>();
            readback += need[3];
            if read_state {
                readback += need[0] + need[1];
            }
            if let Some(res) = &self.slots[slot].groups[group] {
                let have = [
                    res.nodes.size,
                    res.muscles.size,
                    res.bones.size,
                    res.results.size,
                    res.info.size,
                    res.tiles.size,
                ];
                if have.iter().zip(need).all(|(h, n)| *h >= n) {
                    continue;
                }
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
            let Ok([nodes, muscles, bones, results, info, tiles]) =
                <[DeviceBuf; 6]>::try_from(made)
            else {
                unreachable!("one buffer per binding");
            };
            self.slots[slot].groups[group] = Some(GroupRes {
                nodes,
                muscles,
                bones,
                results,
                info,
                tiles,
            });
        }
        readback += frame_bytes;
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
            .staging
            .as_ref()
            .is_none_or(|b| b.size < upload)
        {
            if let Some(old) = self.slots[slot].staging.take() {
                self.free_host(old);
            }
            self.slots[slot].staging = Some(self.alloc_host(upload)?);
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
        Ok(upload)
    }

    /// Bytes of device and pinned host buffers a slot keeps for reuse.
    fn slot_bytes(slot: &Slot) -> u64 {
        slot.groups
            .iter()
            .flatten()
            .map(|g| {
                (g.nodes.size
                    + g.muscles.size
                    + g.bones.size
                    + g.results.size
                    + g.info.size
                    + g.tiles.size) as u64
            })
            .sum::<u64>()
            + slot.staging.as_ref().map_or(0, |b| b.size as u64)
            + slot.readback.as_ref().map_or(0, |b| b.size as u64)
            + slot.frames.as_ref().map_or(0, |b| b.size as u64)
    }

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
            if let Some(b) = self.slots[slot].staging.take() {
                self.free_host(b);
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

    /// Number of evaluation submissions that can be queued without waiting.
    pub fn free_slots(&self) -> usize {
        self.evaluation_slots()
            .filter(|&i| self.slots[i].pending.is_none())
            .count()
    }

    /// Submissions in flight, replays included.
    pub fn in_flight(&self) -> usize {
        self.slots.iter().filter(|s| s.pending.is_some()).count()
    }

    fn evaluation_slots(&self) -> std::ops::Range<usize> {
        0..self.slots.len() - 1
    }

    fn replay_slot(&self) -> usize {
        self.slots.len() - 1
    }

    /// Whether a replay can be recorded now.
    pub fn replay_free(&self) -> bool {
        self.slots[self.replay_slot()].pending.is_none()
    }

    /// Queues a whole trial of one batch on the replay slot with the
    /// recording kernel, which writes every creature's frames. The result
    /// arrives through `poll` with `Completed::frames`. The trial is the one
    /// `submit` scores, computed the same way. The contract is
    /// `VkEngine::record`'s.
    pub fn record(
        &mut self,
        batch: &LaneBatch,
        cfg: &Config,
        total: u32,
        chunk: u32,
    ) -> Result<u64> {
        ensure!(self.replay_free(), "A replay is already being recorded");
        // Physics v2 trials start at the settling tick, as scoring does, and
        // rebuild their node table from joint state at every dispatch, so
        // bit-exact frames need scoring's dispatch boundaries.
        let (start, cuts) = if physics2() {
            (cfg.fidelity().settle(), crate::engine::segment_ends(cfg))
        } else {
            (0, Vec::new())
        };
        self.submit_as(
            std::slice::from_ref(batch),
            cfg,
            start,
            total,
            total,
            chunk,
            false,
            true,
            &cuts,
        )
    }

    /// Uploads the batches and queues ticks `start..end` of trials that last
    /// `total` ticks, in `chunk`-tick ranges, without waiting. The contract
    /// is `VkEngine::submit`'s.
    #[allow(clippy::too_many_arguments)]
    pub fn submit(
        &mut self,
        batches: &[LaneBatch],
        cfg: &Config,
        start: u32,
        end: u32,
        total: u32,
        chunk: u32,
        read_state: bool,
    ) -> Result<u64> {
        self.submit_as(
            batches,
            cfg,
            start,
            end,
            total,
            chunk,
            read_state,
            false,
            &[],
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn submit_as(
        &mut self,
        batches: &[LaneBatch],
        cfg: &Config,
        start: u32,
        end: u32,
        total: u32,
        chunk: u32,
        read_state: bool,
        record: bool,
        cuts: &[u32],
    ) -> Result<u64> {
        ensure!(
            !batches.is_empty() && start < end && end <= total,
            "Empty GPU batch"
        );
        ensure!(
            batches.iter().all(|b| b.capacity <= self.max_capacity),
            "Body too large for this device's kernels"
        );
        let fidelity = cfg.fidelity();
        let kernels: Vec<(CuFunction, u32)> = batches
            .iter()
            .map(|batch| self.kernel(fidelity, batch.capacity, record))
            .collect::<Result<_>>()?;
        // The free slot with the most buffers to reuse: when memory is short,
        // a new allocation may fail where reuse does not.
        let slot = if record {
            self.replay_slot()
        } else {
            self.evaluation_slots()
                .filter(|&i| self.slots[i].pending.is_none())
                .max_by_key(|&i| (Self::slot_bytes(&self.slots[i]), std::cmp::Reverse(i)))
                .context("No free GPU submission slot")?
        };
        // Node positions for every creature, before each step and after the last.
        let frame_count: usize = if record {
            batches
                .iter()
                .map(|b| b.info.len() * creature_kernel::frame_stride(b) * (total as usize + 1))
                .sum()
        } else {
            0
        };
        let frame_bytes = frame_count * std::mem::size_of::<[f32; 2]>();
        let buffers = self.ensure_buffers(slot, batches, read_state, frame_bytes);
        self.recount_allocated();
        buffers?;
        let cu = &self.api.cu;
        let resources = &self.slots[slot];
        let staging = resources.staging.as_ref().unwrap();
        // The frames follow the results in the readback buffer (a recording
        // reads no node state).
        let frames_offset =
            batches.iter().map(|b| b.info.len()).sum::<usize>() * std::mem::size_of::<GpuResult>();
        unsafe {
            cu.check((cu.ctx_set_current)(self.context), "cuCtxSetCurrent")?;
            // Stage every upload in pinned memory, then copy it to the device
            // on the main stream.
            let mut offset = 0usize;
            let mut copy = |data: &[u8], dst: CuDevicePtr| -> Result<()> {
                if data.is_empty() {
                    return Ok(());
                }
                assert!(offset + data.len() <= staging.size);
                let src = staging.ptr.add(offset);
                std::ptr::copy_nonoverlapping(data.as_ptr(), src, data.len());
                cu.check(
                    (cu.memcpy_htod_async)(dst, src as *const c_void, data.len(), resources.main),
                    "cuMemcpyHtoDAsync",
                )?;
                offset += data.len();
                Ok(())
            };
            for (group, batch) in batches.iter().enumerate() {
                let res = resources.groups[group].as_ref().unwrap();
                copy(bytemuck::cast_slice(&batch.nodes), res.nodes.ptr)?;
                copy(bytemuck::cast_slice(&batch.muscles), res.muscles.ptr)?;
                copy(bytemuck::cast_slice(&batch.bones), res.bones.ptr)?;
                copy(bytemuck::cast_slice(&batch.info), res.info.ptr)?;
                copy(bytemuck::cast_slice(&batch.tiles), res.tiles.ptr)?;
                if let Some(results) = &batch.results {
                    copy(bytemuck::cast_slice(results), res.results.ptr)?;
                }
            }
            cu.check(
                (cu.event_record)(resources.start, resources.main),
                "cuEventRecord",
            )?;
            // Batch b runs on stream b, modulo the streams the slot has.
            let streams = batches.len().min(resources.streams.len());
            for &stream in &resources.streams[..streams] {
                cu.check(
                    (cu.stream_wait_event)(stream, resources.start, 0),
                    "cuStreamWaitEvent",
                )?;
            }
            // Large buckets first, as in VkEngine; each range of every batch
            // is queued before the next range of any.
            let mut order: Vec<usize> = (0..batches.len()).collect();
            order.sort_by_key(|&b| std::cmp::Reverse(batches[b].info.len() * batches[b].capacity));
            // Dispatches: (first tick, steps), in `chunk`-tick pieces that
            // never cross a cut.
            let mut spans: Vec<(u32, u32)> = Vec::new();
            let mut at = start;
            while at < end {
                let stop = cuts
                    .iter()
                    .copied()
                    .find(|&c| c > at && c < end)
                    .unwrap_or(end);
                let steps = (stop - at).min(chunk);
                spans.push((at, steps));
                at += steps;
            }
            for (tick, steps) in spans {
                for &b in &order {
                    let batch = &batches[b];
                    let res = resources.groups[b].as_ref().unwrap();
                    let mut params = creature_kernel::launch_params(
                        cfg,
                        batch.capacity,
                        batch.info.len(),
                        tick,
                        steps,
                        total,
                    );
                    if record {
                        params.stride = creature_kernel::frame_stride(batch) as u32;
                    }
                    let mut pointers = [
                        res.nodes.ptr,
                        res.muscles.ptr,
                        res.bones.ptr,
                        res.results.ptr,
                        res.info.ptr,
                        res.tiles.ptr,
                    ];
                    // A recording kernel takes the frames buffer as an eighth
                    // argument; a scoring kernel reads only the first seven.
                    let mut frames = resources.frames.as_ref().map_or(0, |f| f.ptr);
                    let mut args: [*mut c_void; 8] = [
                        &mut pointers[0] as *mut u64 as *mut c_void,
                        &mut pointers[1] as *mut u64 as *mut c_void,
                        &mut pointers[2] as *mut u64 as *mut c_void,
                        &mut params as *mut creature_kernel::Params as *mut c_void,
                        &mut pointers[3] as *mut u64 as *mut c_void,
                        &mut pointers[4] as *mut u64 as *mut c_void,
                        &mut pointers[5] as *mut u64 as *mut c_void,
                        &mut frames as *mut u64 as *mut c_void,
                    ];
                    let (kernel, threads) = kernels[b];
                    let groups = batch.info.len().div_ceil(threads as usize) as c_uint;
                    cu.check(
                        (cu.launch_kernel)(
                            kernel,
                            groups,
                            1,
                            1,
                            threads,
                            1,
                            1,
                            0,
                            resources.streams[b % streams],
                            args.as_mut_ptr(),
                            std::ptr::null_mut(),
                        ),
                        "cuLaunchKernel",
                    )?;
                }
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
                    res.results.ptr,
                    batch.info.len() * std::mem::size_of::<GpuResult>(),
                )?;
            }
            if read_state {
                for (group, batch) in batches.iter().enumerate() {
                    let res = resources.groups[group].as_ref().unwrap();
                    read(res.nodes.ptr, std::mem::size_of_val(batch.nodes.as_slice()))?;
                    read(
                        res.muscles.ptr,
                        std::mem::size_of_val(batch.muscles.as_slice()),
                    )?;
                }
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
        let ticket = self.next_ticket;
        self.next_ticket += 1;
        self.slots[slot].pending = Some(Pending {
            ticket,
            layout: batches
                .iter()
                .map(|b| (b.slots.clone(), b.creatures.clone()))
                .collect(),
            result_count: batches.iter().map(|b| b.info.len()).sum(),
            state: read_state.then(|| {
                batches
                    .iter()
                    .map(|b| (b.nodes.len(), b.muscles.len()))
                    .collect()
            }),
            frames: record.then_some((frames_offset, frame_count)),
        });
        Ok(ticket)
    }

    /// Returns the oldest finished submission's results, waiting up to
    /// `timeout` for one. The contract is `VkEngine::poll`'s.
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
            let mut batches = Vec::with_capacity(pending.layout.len());
            let mut start = 0;
            for (slots, creatures) in pending.layout {
                let end = start + slots.len();
                batches.push((slots, creatures, flat[start..end].to_vec()));
                start = end;
            }
            let state = pending.state.map(|sizes| {
                let mut at = readback
                    .ptr
                    .add(pending.result_count * std::mem::size_of::<GpuResult>());
                sizes
                    .into_iter()
                    .map(|(nodes, muscles)| {
                        let node_state =
                            std::slice::from_raw_parts(at as *const crate::physics::Node, nodes)
                                .to_vec();
                        at = at.add(nodes * std::mem::size_of::<crate::physics::Node>());
                        let muscle_state =
                            std::slice::from_raw_parts(at as *const f32, muscles).to_vec();
                        at = at.add(muscles * std::mem::size_of::<f32>());
                        (node_state, muscle_state)
                    })
                    .collect()
            });
            let frames = pending.frames.map(|(offset, count)| {
                std::slice::from_raw_parts(readback.ptr.add(offset) as *const [f32; 2], count)
                    .to_vec()
            });
            self.last_gpu_seconds = gpu_seconds;
            Ok(Some(Completed {
                ticket: pending.ticket,
                batches,
                state,
                frames,
                gpu_seconds,
            }))
        }
    }
}

impl Drop for CudaEngine {
    fn drop(&mut self) {
        self.prefetch.close(Duration::from_secs(30));
        let api = self.api.clone();
        let cu = &api.cu;
        unsafe {
            (cu.ctx_set_current)(self.context);
            (cu.ctx_synchronize)();
        }
        for slot in 0..self.slots.len() {
            for group in 0..self.slots[slot].groups.len() {
                self.drop_group(slot, group);
            }
            if let Some(b) = self.slots[slot].staging.take() {
                self.free_host(b);
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
            for kernel in self.kernels.values().chain(self.recording.values()) {
                (cu.module_unload)(kernel.module);
            }
            (cu.primary_ctx_release)(self.device);
        }
    }
}
