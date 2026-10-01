//! NVIDIA GPU backend: the physics authority.
//!
//! It runs `shaders/warp_creature.cu`, one creature per group of 8, 16 or 32
//! lanes (`warp_kernel`), behind `engine::gpu_engine`'s submit and poll
//! contract. A unit is uploaded once and runs as waves of up to `warp_kernel::WAVE`
//! creatures, one kernel launch each, on the slot's streams. Inside a wave
//! every lane group runs its creature to the end of its trial and then takes
//! the next one, so there are no trial segments.
//!
//! Nothing CUDA is linked at build time. The driver API (`libcuda`) and NVRTC
//! (`libnvrtc`) are loaded when the engine opens, so the game builds without
//! them, but it needs them to run. NVRTC comes from the system CUDA toolkit
//! or from NVIDIA's pip wheel; see `docs/building.md`. Kernels compile to a cubin for the device's
//! architecture, one per lane class, world (the effects that are on), rate
//! and recording, in the background when the engine opens and when a new
//! world first appears.
//!
//! Developer diagnostics, never needed to play: `EVOLUTION_NVRTC` names the
//! NVRTC library, and `EVOLUTION_CUDA_VERBOSE` reports compile times.
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
/// CUDA stream priority: lower numbers run first, 0 is the default.
fn stream_priority(replay: bool) -> c_int {
    if replay { -100 } else { 0 }
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
                memset_d32_async: symbol!(library, "cuMemsetD32Async"),
                occupancy: symbol!(library, "cuOccupancyMaxActiveBlocksPerMultiprocessor"),
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

/// Streams per submission slot. Wave w of a unit runs on stream w modulo
/// this.
const STREAM_LIMIT: usize = 8;

/// A kernel: recording or scoring, lanes per creature, world flags
/// (`warp_kernel::world_flags`) and fidelity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct KernelKey {
    record: bool,
    class: usize,
    flags: u32,
    fidelity: Fidelity,
}

struct Kernel {
    module: CuModule,
    function: CuFunction,
    /// Blocks of `warp_kernel::BLOCK` threads resident per multiprocessor.
    blocks_per_sm: u32,
}

/// Take-up counters per wave: one per muscle-rounds bucket.
const BUCKETS: usize = crate::warp_kernel::ROUNDS;

/// The take-up buckets of a wave (`Takeup` in `shaders/warp_creature.cu`).
/// Bucket `b` holds the wave's creatures with `b + 1` muscle rounds (bodies
/// without muscles join the first) as `start[b]..end[b]`, and the warps from
/// `warp[b]` start on it. A warp takes only from its bucket until it runs dry,
/// so its groups run one round count, where one counter for the wave soon
/// fills a warp with creatures from the whole wave and runs it at their
/// largest round count.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Takeup {
    start: [u32; BUCKETS],
    end: [u32; BUCKETS],
    warp: [u32; BUCKETS],
}

impl Takeup {
    /// Buckets for the `count` creatures of a batch from `first` (`heads`
    /// sorted by rounds, as `warp_kernel::pack` writes them) run by `warps`
    /// warps. Each bucket with creatures gets at least one warp and the rest
    /// in proportion to its creatures times their rounds plus one, a round
    /// being about as long as the rest of a step. `buckets` 1 (the
    /// developer diagnostic `EVOLUTION_WARP_BUCKETS=1`) gives one counter.
    fn new(heads: &[[u32; 4]], first: usize, count: usize, warps: usize, buckets: usize) -> Self {
        let bucket_of = |c: usize| {
            let rounds = ((heads[2 * (first + c)][0] >> 16) & 255) as usize;
            if buckets < BUCKETS {
                0
            } else {
                rounds.clamp(1, BUCKETS) - 1
            }
        };
        let mut counts = [0usize; BUCKETS];
        let mut sorted = true;
        let mut last = 0;
        for c in 0..count {
            let b = bucket_of(c);
            sorted &= b >= last;
            last = b;
            counts[b] += 1;
        }
        if !sorted {
            counts = [0; BUCKETS];
            counts[0] = count;
        }
        let weight: [usize; BUCKETS] = std::array::from_fn(|b| counts[b] * (b + 2));
        let total = weight.iter().sum::<usize>().max(1);
        let mut shares: [usize; BUCKETS] = std::array::from_fn(|b| {
            if counts[b] == 0 {
                0
            } else {
                (warps * weight[b] / total).max(1)
            }
        });
        while shares.iter().sum::<usize>() > warps {
            let Some(b) = (0..BUCKETS)
                .filter(|&b| shares[b] > 1)
                .max_by_key(|&b| shares[b])
            else {
                break;
            };
            shares[b] -= 1;
        }
        let spare = warps.saturating_sub(shares.iter().sum());
        let heaviest = (0..BUCKETS).max_by_key(|&b| weight[b]).unwrap_or(0);
        shares[heaviest] += spare;
        let mut take = Takeup {
            start: [0; BUCKETS],
            end: [0; BUCKETS],
            warp: [0; BUCKETS],
        };
        let (mut at, mut warp) = (0usize, 0usize);
        for b in 0..BUCKETS {
            take.start[b] = at as u32;
            at += counts[b];
            take.end[b] = at as u32;
            take.warp[b] = warp as u32;
            warp += shares[b];
        }
        take
    }
}

struct DeviceBuf {
    ptr: CuDevicePtr,
    size: usize,
}

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

    pub const fn new() -> Self {
        Self {
            ptr: std::ptr::NonNull::dangling().as_ptr(),
            len: 0,
            capacity: 0,
            bytes: 0,
            registered: false,
        }
    }

    /// Makes this `len` copies of `value`, in the memory it has when that is
    /// large enough.
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

/// One batch's buffers: lane records, muscles, muscle-end lists, heads and
/// results.
struct GroupRes {
    bufs: [DeviceBuf; 5],
}

/// Per-submission resources. Each wave runs on one of the
/// slot's streams; the main stream uploads, joins the wave streams and reads
/// the results back.
struct Slot {
    main: CuStream,
    streams: Vec<CuStream>,
    joins: Vec<CuEvent>,
    start: CuEvent,
    stop: CuEvent,
    done: CuEvent,
    groups: Vec<Option<GroupRes>>,
    /// One creature counter per wave and take-up bucket.
    counters: Option<DeviceBuf>,
    readback: Option<HostBuf>,
    /// Recorded replay frames; only the replay slot has one.
    frames: Option<DeviceBuf>,
    pending: Option<Pending>,
}

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

pub struct CudaEngine {
    api: Arc<Api>,
    device: CuDevice,
    context: CuContext,
    pub name: String,
    arch: String,
    multiprocessors: i32,
    kernels: HashMap<KernelKey, Kernel>,
    /// Kernels compiling on background threads.
    prefetch: Arc<Prefetch>,
    /// World flags whose kernels have been queued for every class.
    worlds: HashSet<(u32, Fidelity)>,
    /// Submission slots. The last one is kept for replays, with streams of
    /// its own, so a replay never waits behind evaluation.
    slots: Vec<Slot>,
    next_ticket: u64,
    pub max_capacity: usize,
    pub allocated_bytes: u64,
    pub last_gpu_seconds: f64,
}

// The context and every mapped pointer are only used by the thread that owns
// the engine; it makes the context current on that thread when it opens.
unsafe impl Send for CudaEngine {}

/// Rounds a buffer size up (`engine::padded_size`), so buffers are reused
/// across units of slightly different sizes.
fn buffer_size(bytes: usize) -> usize {
    crate::engine::padded_size(bytes as u64) as usize
}

/// NVRTC's source for one kernel, compiled to a cubin. It needs no CUDA
/// context, so any thread may run it. Compiled kernels are kept on disk
/// (`kernel_cache_dir`), keyed by a hash of the source, the options and the
/// NVRTC version, so a later start loads them in milliseconds. With
/// `use_cache` false the compile ignores an entry (it may be damaged) and
/// writes a fresh one.
fn compile_kernel(api: &Api, options: &[String], key: KernelKey, use_cache: bool) -> Result<Vec<u8>> {
    let source = crate::warp_kernel::cuda_source(key.class, key.flags, key.fidelity, key.record);
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
    let started = Instant::now();
    let (cubin, log) = api
        .nvrtc
        .compile(&source, options)
        .with_context(|| format!("{}-lane CUDA kernel", key.class))?;
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
    Ok(cubin)
}

/// Compiles the lane-group kernel for `class` lanes and `cfg`'s world on
/// this machine's NVRTC and returns ptxas's report (registers, spills,
/// stack), for developers measuring register use. Needs no GPU time.
pub fn compile_report(class: usize, cfg: &Config, record: bool, arch: &str) -> Result<String> {
    let api = api()?;
    let options = vec![
        format!("--gpu-architecture={arch}"),
        "--std=c++17".into(),
        "--prec-div=false".into(),
        "--prec-sqrt=false".into(),
        "--fmad=true".into(),
        "--extra-device-vectorization".into(),
        "--ptxas-options=-v".into(),
    ];
    let mut options = options;
    if let Ok(extra) = std::env::var("EVOLUTION_NVRTC_EXTRA") {
        options.extend(extra.split_whitespace().map(String::from));
    }
    let source = crate::warp_kernel::cuda_source(
        class,
        crate::warp_kernel::world_flags(cfg),
        cfg.fidelity(),
        record,
    );
    let (_, log) = api.nvrtc.compile(&source, &options)?;
    Ok(log
        .lines()
        .filter(|l| {
            std::env::var_os("EVOLUTION_NVRTC_LOG").is_some()
                || l.contains("registers")
                || l.contains("spill")
                || l.contains("stack")
        })
        .collect::<Vec<_>>()
        .join("\n"))
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
#[derive(Default)]
struct Prefetch {
    state: Mutex<PrefetchState>,
    ready: Condvar,
}

#[derive(Default)]
struct PrefetchState {
    queue: VecDeque<KernelKey>,
    running: HashSet<KernelKey>,
    done: HashMap<KernelKey, std::result::Result<Vec<u8>, String>>,
    /// Background threads alive.
    workers: usize,
    closed: bool,
}

impl Prefetch {
    /// Compiles queued kernels until the queue is empty or the engine closes.
    fn work(&self, api: &Api, options: &[String]) {
        loop {
            let key = {
                let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
                if state.closed {
                    state.workers -= 1;
                    self.ready.notify_all();
                    return;
                }
                let Some(key) = state.queue.pop_front() else {
                    state.workers -= 1;
                    self.ready.notify_all();
                    return;
                };
                state.running.insert(key);
                key
            };
            let result = compile_kernel(api, options, key, true).map_err(|e| format!("{e:#}"));
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
            state.queue.retain(|k| *k != key);
            return None;
        }
    }

    /// Queues `jobs` not already queued, running or done, and starts
    /// threads (at most three at a time) to compile them.
    fn enqueue(self: &Arc<Self>, jobs: Vec<KernelKey>, api: &Arc<Api>, options: &Arc<Vec<String>>) {
        let mut state = self.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.closed {
            return;
        }
        for key in jobs {
            let known = state.done.contains_key(&key)
                || state.running.contains(&key)
                || state.queue.contains(&key);
            if !known {
                state.queue.push_back(key);
            }
        }
        while state.workers < 3 && state.workers < state.queue.len() + state.running.len() {
            let (prefetch, api, options) = (self.clone(), api.clone(), options.clone());
            let spawned = std::thread::Builder::new()
                .name("cuda-compile".into())
                .spawn(move || {
                    // Compile beside the pool, not on the engine thread's CPU.
                    crate::threads::pin_pool();
                    prefetch.work(&api, &options)
                });
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
    /// (case-insensitive) and starts compiling the kernels of the default
    /// world. Bodies above `warp_kernel::MAX_NODES` nodes stay elsewhere.
    pub fn new(name: &str, max_capacity: usize) -> Result<Self> {
        Self::open(name, max_capacity.min(crate::warp_kernel::MAX_NODES))
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
            let mut engine = Self {
                name: String::new(),
                api: api.clone(),
                device,
                context,
                arch,
                multiprocessors,
                kernels: HashMap::new(),
                prefetch: Arc::new(Prefetch::default()),
                worlds: HashSet::new(),
                slots: Vec::new(),
                next_ticket: 0,
                max_capacity,
                allocated_bytes: 0,
                last_gpu_seconds: 0.0,
            };
            engine.name = format!("{device_name} (CUDA)");
            // Evaluation slots plus one for replays.
            let slots = crate::engine::gpu_slots() + 1;
            for index in 0..slots {
                let slot = engine.create_slot(index + 1 == slots)?;
                engine.slots.push(slot);
            }
            engine.prefetch_world(&Config::default());
            Ok(engine)
        }
    }

    /// A slot's streams. The replay slot's streams have the highest
    /// priority, so a recording gets the GPU's blocks ahead of queued scoring
    /// work (priorities out of range are clamped to the greatest one).
    fn create_slot(&self, replay: bool) -> Result<Slot> {
        let cu = &self.api.cu;
        unsafe {
            let mut main = std::ptr::null_mut();
            cu.check(
                (cu.stream_create)(&mut main, CU_STREAM_NON_BLOCKING, stream_priority(replay)),
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
        let mut options = vec![
            format!("--gpu-architecture={}", self.arch),
            "--std=c++17".into(),
            "--prec-div=false".into(),
            "--prec-sqrt=false".into(),
            "--fmad=true".into(),
            "--extra-device-vectorization".into(),
            "--ptxas-options=-v".into(),
        ];
        // A developer's extra options, such as -lineinfo for a profiler.
        if let Ok(extra) = std::env::var("EVOLUTION_NVRTC_EXTRA") {
            options.extend(extra.split_whitespace().map(String::from));
        }
        options
    }

    /// Loads a cubin into this engine's context.
    fn load(&self, cubin: &[u8], key: KernelKey) -> Result<Kernel> {
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
            let mut blocks = 0;
            cu.check(
                (cu.occupancy)(&mut blocks, function, crate::warp_kernel::BLOCK as c_int, 0),
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
                    "CUDA: {}-lane {} kernel: {} registers, {} B shared, {} B local per thread, {} blocks of {} threads per SM",
                    key.class,
                    if key.record { "recording" } else { "scoring" },
                    attribute(4),
                    attribute(1),
                    attribute(3),
                    blocks,
                    crate::warp_kernel::BLOCK
                );
            }
            Ok(Kernel {
                module,
                function,
                blocks_per_sm: blocks.max(1) as u32,
            })
        }
    }

    /// Queues the scoring and recording kernels of `cfg`'s world, every
    /// lane class, on the background compiler.
    fn prefetch_world(&mut self, cfg: &Config) {
        let flags = crate::warp_kernel::world_flags(cfg);
        let fidelity = cfg.fidelity();
        if !self.worlds.insert((flags, fidelity)) {
            return;
        }
        let mut jobs = Vec::new();
        for record in [false, true] {
            // Fine checks never record.
            if record && fidelity != Fidelity::standard() {
                continue;
            }
            for class in crate::warp_kernel::CLASSES {
                let key = KernelKey {
                    record,
                    class,
                    flags,
                    fidelity,
                };
                if !self.kernels.contains_key(&key) {
                    jobs.push(key);
                }
            }
        }
        let options = Arc::new(self.options());
        self.prefetch.enqueue(jobs, &self.api, &options);
    }

    /// The kernel for `key`: from the background compiler if it has it or is
    /// on it, else compiled here.
    fn kernel(&mut self, key: KernelKey) -> Result<(CuFunction, u32)> {
        if !self.kernels.contains_key(&key) {
            let started = Instant::now();
            let compile = |engine: &Self, use_cache: bool| {
                compile_kernel(&engine.api, &engine.options(), key, use_cache)
            };
            let cubin = match self.prefetch.take(key) {
                Some(result) => result?,
                None => compile(self, true)?,
            };
            // A damaged cache entry fails to load: compile it again.
            let kernel = match self.load(&cubin, key) {
                Ok(kernel) => kernel,
                Err(_) => self.load(&compile(self, false)?, key)?,
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

    fn drop_group(&mut self, slot: usize, group: usize) {
        if let Some(res) = self.slots[slot].groups[group].take() {
            for buf in res.bufs {
                self.free_device(buf);
            }
        }
    }

    /// Bytes each buffer of a batch needs: lanes, muscles, ends, heads,
    /// results.
    fn needs(batch: &LaneBatch) -> Result<[usize; 5]> {
        let wave = batch
            .wave
            .as_ref()
            .context("The CUDA kernel needs a batch from warp_kernel::pack")?;
        Ok([
            std::mem::size_of_val(&*wave.lanes),
            std::mem::size_of_val(&*wave.muscles),
            std::mem::size_of_val(&*wave.ends),
            std::mem::size_of_val(&*wave.heads),
            batch.slots.len() * std::mem::size_of::<GpuResult>(),
        ])
    }

    /// Grows the slot's device buffers, streams and readback buffer to fit
    /// `batches` in `waves` waves.
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
                        stream_priority(slot == self.slots.len() - 1),
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
        let counter_bytes = 4 * BUCKETS * waves;
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

    /// Bytes of host memory a batch's buffers hold.
    pub fn held_bytes(batch: &LaneBatch) -> usize {
        batch.wave.as_ref().map_or(0, |w| {
            w.lanes.held_bytes() + w.muscles.held_bytes() + w.ends.held_bytes() + w.heads.held_bytes()
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
                p.batches.iter().map(|b| Self::held_bytes(b) as u64).sum::<u64>()
            })
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

    /// The whole trial of the one creature in `batch` on the replay slot with
    /// the recording kernel, which writes every frame. The result arrives
    /// through `poll` with `Completed::frames`; it is the trial `submit`
    /// scores, computed the same way.
    pub fn record(&mut self, batch: LaneBatch, cfg: &Config) -> Result<u64> {
        ensure!(self.replay_free(), "A replay is already being recorded");
        self.submit_as(&mut vec![batch], cfg, true)
    }

    /// Uploads the batches and queues their whole trials, without waiting.
    /// On success the engine takes the batches, copies straight from their
    /// buffers, and hands them back with the results (`Completed::batches`)
    /// for the next unit to reuse; on failure they stay with the caller.
    pub fn submit(&mut self, batches: &mut Vec<LaneBatch>, cfg: &Config) -> Result<u64> {
        self.submit_as(batches, cfg, false)
    }

    fn submit_as(&mut self, batches: &mut Vec<LaneBatch>, cfg: &Config, record: bool) -> Result<u64> {
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
        let flags = crate::warp_kernel::world_flags(cfg);
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
        // Waves: (batch, first creature, count), the largest bodies first.
        let mut waves: Vec<(usize, usize, usize)> = Vec::new();
        for (b, batch) in batches.iter().enumerate() {
            let count = batch.slots.len();
            let mut at = 0;
            while at < count {
                let n = (count - at).min(crate::warp_kernel::WAVE);
                waves.push((b, at, n));
                at += n;
            }
        }
        let total = (fidelity.settle() + cfg.steps()) as usize;
        let stride = batches
            .first()
            .map_or(0, creature_kernel::frame_stride);
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
        let buckets = crate::warp_kernel::solver_setting("BUCKETS", BUCKETS as u32) as usize;
        let buffers = self.ensure_buffers(slot, batches, waves.len(), frame_bytes);
        self.recount_allocated();
        buffers?;
        let cu = &self.api.cu;
        let resources = &self.slots[slot];
        let results_bytes = batches.iter().map(|b| b.slots.len()).sum::<usize>()
            * std::mem::size_of::<GpuResult>();
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
                (cu.memset_d32_async)(counters.ptr, 0, BUCKETS * waves.len(), resources.main),
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
                let batch = &batches[b];
                let res = resources.groups[b].as_ref().unwrap();
                let (kernel, blocks_per_sm) = kernels[b];
                let mut params = crate::warp_kernel::params(cfg, first, count, stride);
                let groups_per_block = (crate::warp_kernel::BLOCK as usize / 32) * (32 / batch.capacity);
                let blocks = count
                    .div_ceil(groups_per_block)
                    .min(blocks_per_sm as usize * self.multiprocessors as usize)
                    .max(1);
                let warps = blocks * crate::warp_kernel::BLOCK as usize / 32;
                let heads = &batch.wave.as_ref().unwrap().heads;
                let mut takeup = Takeup::new(heads, first, count, warps, buckets);
                let mut pointers = [
                    res.bufs[0].ptr,
                    res.bufs[1].ptr,
                    res.bufs[2].ptr,
                    res.bufs[3].ptr,
                    res.bufs[4].ptr,
                    counters.ptr + (4 * BUCKETS * w) as u64,
                ];
                let mut frames = resources.frames.as_ref().map_or(0, |f| f.ptr);
                let mut args: [*mut c_void; 9] = [
                    &mut pointers[0] as *mut u64 as *mut c_void,
                    &mut pointers[1] as *mut u64 as *mut c_void,
                    &mut pointers[2] as *mut u64 as *mut c_void,
                    &mut pointers[3] as *mut u64 as *mut c_void,
                    &mut pointers[4] as *mut u64 as *mut c_void,
                    &mut pointers[5] as *mut u64 as *mut c_void,
                    &mut params as *mut crate::warp_kernel::Params as *mut c_void,
                    &mut takeup as *mut Takeup as *mut c_void,
                    &mut frames as *mut u64 as *mut c_void,
                ];
                cu.check(
                    (cu.launch_kernel)(
                        kernel,
                        blocks as c_uint,
                        1,
                        1,
                        crate::warp_kernel::BLOCK,
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

    /// Returns the oldest finished submission's results, waiting up to
    /// `timeout` for one.
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
        self.prefetch.close(Duration::from_secs(30));
        let api = self.api.clone();
        let cu = &api.cu;
        unsafe {
            (cu.ctx_set_current)(self.context);
            (cu.ctx_synchronize)();
        }
        for slot in 0..self.slots.len() {
            // Unregistered while the context lives.
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

    fn heads(rounds: &[u32]) -> Vec<[u32; 4]> {
        rounds
            .iter()
            .flat_map(|&r| [[r << 16, 0, 0, 0], [0; 4]])
            .collect()
    }

    #[test]
    fn takeup_buckets_follow_the_rounds_and_share_the_warps() {
        let h = heads(&[0, 1, 1, 2, 2, 2, 4]);
        let t = Takeup::new(&h, 0, 7, 20, BUCKETS);
        assert_eq!(t.start, [0, 3, 6, 6]);
        assert_eq!(t.end, [3, 6, 6, 7]);
        // Weights 3 x 2, 3 x 3, 0, 1 x 5 of 20: 6, 9, 0 and 5 warps.
        assert_eq!(t.warp, [0, 6, 15, 15]);
        // A wave from the middle of a batch, and fewer warps than buckets.
        let t = Takeup::new(&h, 3, 4, 2, BUCKETS);
        assert_eq!((t.start, t.end), ([0, 0, 3, 3], [0, 3, 3, 4]));
        assert_eq!(t.warp, [0, 0, 1, 1]);
        // One counter.
        let t = Takeup::new(&h, 0, 7, 20, 1);
        assert_eq!(
            (t.start, t.end, t.warp),
            ([0, 7, 7, 7], [7, 7, 7, 7], [0, 20, 20, 20])
        );
    }

    #[test]
    fn unsorted_heads_take_one_counter() {
        let t = Takeup::new(&heads(&[2, 1]), 0, 2, 8, BUCKETS);
        assert_eq!((t.start, t.end), ([0, 2, 2, 2], [2, 2, 2, 2]));
    }
}
