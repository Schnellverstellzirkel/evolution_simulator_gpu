//! Synthetic CUDA kernels for the power rows in `docs/building.md`: each
//! runs at full occupancy for a fixed time so `tools/power-sample.sh` can
//! read the clock, the power and the limiter it settles at. Dividing the
//! power by the warp-instruction rate this prints gives nJ per warp
//! instruction for that instruction mix.
//!
//! - `fma`: 8 independent chains of x = 1.9 - x*x per thread, registers only
//!   (one FFMA each). The map is chaotic, so the bits keep toggling.
//! - `mio`: per 2 FFMAs one shared-memory load (a pointer chase that stays in
//!   the lane's own bank) and one warp shuffle, so half the instructions go
//!   to the MIO pipes.
//! - `int`: integer multiply-add, logic and add chains with one shuffle per
//!   six instructions, no floating point.
//! - `idle`: holds a context and launches nothing, for the idle power.
//!
//! The kernels compile with NVRTC and launch through the CUDA driver API,
//! found the way `src/cuda_engine.rs` finds them (`EVOLUTION_NVRTC`, the
//! loader's path, then the pip wheel of `docs/building.md`). Each launch is
//! about 200 ms. It prints one line a second and a summary.
//!
//! Usage: power_probe <fma|mio|int|idle> [seconds=30] [cubin-dir]
//! With a cubin directory it writes `<mode>.cubin` there, for counting the
//! loop's SASS with `nvdisasm`.
use anyhow::{Context, Result, bail};
use std::{
    ffi::{CStr, CString, c_char, c_int, c_uint, c_void},
    path::{Path, PathBuf},
    time::Instant,
};

const BLOCK: u32 = 256;

const SOURCE: &str = r#"
#define FULL 0xffffffffu
__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
// Block 0, thread 0 writes its SM cycles and nanoseconds for the launch.
#define STAMP_BEGIN long long c0_ = clock64(); unsigned long long t0_ = gtimer();
#define STAMP_END(v) \
    long long c1_ = clock64(); unsigned long long t1_ = gtimer(); \
    out[2 + blockIdx.x * blockDim.x + threadIdx.x] = (unsigned long long)(v); \
    if (blockIdx.x == 0 && threadIdx.x == 0) { out[0] = c1_ - c0_; out[1] = t1_ - t0_; }

extern "C" __global__ void __launch_bounds__(256) fmachain(unsigned long long* out, int iters) {
    float x0 = 0.11f + 0.001f * (threadIdx.x & 31), x1 = x0 + 0.1f, x2 = x0 + 0.2f,
          x3 = x0 + 0.3f, x4 = x0 - 0.1f, x5 = x0 - 0.2f, x6 = x0 - 0.3f, x7 = x0 - 0.4f;
    STAMP_BEGIN
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            x0 = fmaf(x0, -x0, 1.9f); x1 = fmaf(x1, -x1, 1.9f);
            x2 = fmaf(x2, -x2, 1.9f); x3 = fmaf(x3, -x3, 1.9f);
            x4 = fmaf(x4, -x4, 1.9f); x5 = fmaf(x5, -x5, 1.9f);
            x6 = fmaf(x6, -x6, 1.9f); x7 = fmaf(x7, -x7, 1.9f);
        }
    }
    STAMP_END(__float_as_uint(x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7))
}

// 64 rows of 32 words; lane L only reads column L, so every load is
// conflict-free. Each word is the byte offset of the next one in the lane's
// chain (row r goes to row 5r + 1 mod 64).
__device__ __forceinline__ unsigned chase(const unsigned* tab, unsigned p) {
    return *(const unsigned*)((const char*)tab + p);
}
extern "C" __global__ void __launch_bounds__(256) mio(unsigned long long* out, int iters) {
    __shared__ unsigned tab[2048];
    for (unsigned e = threadIdx.x; e < 2048; e += blockDim.x) {
        unsigned row = e >> 5, lane = e & 31;
        tab[e] = ((((row * 5 + 1) & 63) << 5) | lane) << 2;
    }
    __syncthreads();
    unsigned lane = threadIdx.x & 31;
    unsigned p0 = lane << 2, p1 = (32 + lane) << 2, p2 = (64 + lane) << 2, p3 = (96 + lane) << 2;
    float x0 = 0.11f + 0.001f * lane, x1 = x0 + 0.1f, x2 = x0 + 0.2f,
          x3 = x0 + 0.3f, x4 = x0 - 0.1f, x5 = x0 - 0.2f, x6 = x0 - 0.3f, x7 = x0 - 0.4f;
    STAMP_BEGIN
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            x0 = fmaf(x0, -x0, 1.9f); x1 = fmaf(x1, -x1, 1.9f); p0 = chase(tab, p0);
            x1 = __shfl_xor_sync(FULL, x1, 1);
            x2 = fmaf(x2, -x2, 1.9f); x3 = fmaf(x3, -x3, 1.9f); p1 = chase(tab, p1);
            x3 = __shfl_xor_sync(FULL, x3, 2);
            x4 = fmaf(x4, -x4, 1.9f); x5 = fmaf(x5, -x5, 1.9f); p2 = chase(tab, p2);
            x5 = __shfl_xor_sync(FULL, x5, 4);
            x6 = fmaf(x6, -x6, 1.9f); x7 = fmaf(x7, -x7, 1.9f); p3 = chase(tab, p3);
            x7 = __shfl_xor_sync(FULL, x7, 8);
        }
    }
    STAMP_END(__float_as_uint(x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7) ^ p0 ^ p1 ^ p2 ^ p3)
}

extern "C" __global__ void __launch_bounds__(256) intmix(unsigned long long* out, int iters) {
    unsigned h0 = threadIdx.x * 0x9e3779b9u + 1, h1 = h0 ^ 0x85ebca6bu,
             h2 = h0 ^ 0xc2b2ae35u, h3 = h0 ^ 0x27d4eb2fu;
    unsigned g0 = h0 + 7, g1 = h1 + 7, g2 = h2 + 7, g3 = h3 + 7;
    STAMP_BEGIN
    for (int i = 0; i < iters; ++i) {
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            h0 = h0 * 0x01000193u + h3;           g0 = g0 * 0x01000193u + g3;
            h1 = (h1 ^ h0) + 0x7f4a7c15u;          g1 = (g1 ^ g0) + 0x7f4a7c15u;
            h2 = h2 * 0x2545f491u + h1;           g2 = g2 * 0x2545f491u + g1;
            h3 = __shfl_xor_sync(FULL, h3 ^ h2, 1); g3 = __shfl_xor_sync(FULL, g3 ^ g2, 2);
        }
    }
    STAMP_END(h0 ^ h1 ^ h2 ^ h3 ^ g0 ^ g1 ^ g2 ^ g3)
}
"#;

/// One mode: the kernel's entry point, and per thread and loop iteration
/// the named operations and the SASS instructions NVRTC 13.0 emits for the
/// loop on sm_89: the named operations plus IADD3, ISETP and BRA for the
/// loop (counted with `cuobjdump -sass` on the `cubin-dir` output; recount
/// after an NVRTC upgrade).
struct Mode {
    entry: &'static str,
    ops: &'static str,
    sass_per_iter: u64,
}

fn mode(name: &str) -> Result<Mode> {
    Ok(match name {
        "fma" => Mode {
            entry: "fmachain",
            ops: "128 FFMA",
            sass_per_iter: 131,
        },
        "mio" => Mode {
            entry: "mio",
            ops: "64 FFMA, 32 LDS, 32 SHFL",
            sass_per_iter: 131,
        },
        "int" => Mode {
            entry: "intmix",
            ops: "32 IMAD, 32 LOP3, 16 IADD3, 16 SHFL",
            sass_per_iter: 99,
        },
        _ => bail!("mode must be fma, mio, int or idle"),
    })
}

type CuResult = c_int;
type Ptr = *mut c_void;

macro_rules! sym {
    ($lib:expr, $name:literal) => {
        *$lib
            .get(concat!($name, "\0").as_bytes())
            .with_context(|| format!("symbol {}", $name))?
    };
}

struct Cuda {
    init: unsafe extern "C" fn(c_uint) -> CuResult,
    device_get: unsafe extern "C" fn(*mut c_int, c_int) -> CuResult,
    device_get_name: unsafe extern "C" fn(*mut c_char, c_int, c_int) -> CuResult,
    device_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> CuResult,
    primary_ctx_retain: unsafe extern "C" fn(*mut Ptr, c_int) -> CuResult,
    ctx_set_current: unsafe extern "C" fn(Ptr) -> CuResult,
    ctx_synchronize: unsafe extern "C" fn() -> CuResult,
    module_load_data: unsafe extern "C" fn(*mut Ptr, *const c_void) -> CuResult,
    module_get_function: unsafe extern "C" fn(*mut Ptr, Ptr, *const c_char) -> CuResult,
    mem_alloc: unsafe extern "C" fn(*mut u64, usize) -> CuResult,
    memcpy_dtoh: unsafe extern "C" fn(*mut c_void, u64, usize) -> CuResult,
    occupancy: unsafe extern "C" fn(*mut c_int, Ptr, c_int, usize) -> CuResult,
    func_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, Ptr) -> CuResult,
    #[allow(clippy::type_complexity)]
    launch: unsafe extern "C" fn(
        Ptr,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        Ptr,
        *mut Ptr,
        *mut Ptr,
    ) -> CuResult,
    _lib: libloading::Library,
}

fn check(result: CuResult, what: &str) -> Result<()> {
    if result != 0 {
        bail!("{what} failed with CUDA error {result}");
    }
    Ok(())
}

impl Cuda {
    fn load() -> Result<Self> {
        let lib = unsafe { libloading::Library::new("libcuda.so.1") }.context("libcuda.so.1")?;
        unsafe {
            Ok(Self {
                init: sym!(lib, "cuInit"),
                device_get: sym!(lib, "cuDeviceGet"),
                device_get_name: sym!(lib, "cuDeviceGetName"),
                device_get_attribute: sym!(lib, "cuDeviceGetAttribute"),
                primary_ctx_retain: sym!(lib, "cuDevicePrimaryCtxRetain"),
                ctx_set_current: sym!(lib, "cuCtxSetCurrent"),
                ctx_synchronize: sym!(lib, "cuCtxSynchronize"),
                module_load_data: sym!(lib, "cuModuleLoadData"),
                module_get_function: sym!(lib, "cuModuleGetFunction"),
                mem_alloc: sym!(lib, "cuMemAlloc_v2"),
                memcpy_dtoh: sym!(lib, "cuMemcpyDtoH_v2"),
                occupancy: sym!(lib, "cuOccupancyMaxActiveBlocksPerMultiprocessor"),
                func_get_attribute: sym!(lib, "cuFuncGetAttribute"),
                launch: sym!(lib, "cuLaunchKernel"),
                _lib: lib,
            })
        }
    }
}

/// NVRTC candidates in the order `src/cuda_engine.rs` tries them.
fn nvrtc_candidates() -> Vec<PathBuf> {
    if let Some(path) = std::env::var_os("EVOLUTION_NVRTC") {
        return vec![PathBuf::from(path)];
    }
    let mut out: Vec<PathBuf> = ["libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so"]
        .iter()
        .map(PathBuf::from)
        .collect();
    if let Some(home) = std::env::var_os("HOME") {
        let venv = Path::new(&home).join(".local/share/evolution-cuda/venv/lib");
        for python in std::fs::read_dir(&venv).into_iter().flatten().flatten() {
            let nvidia = python.path().join("site-packages/nvidia");
            for (folder, name) in [
                ("cu13/lib", "libnvrtc.so.13"),
                ("cuda_nvrtc/lib", "libnvrtc.so.12"),
            ] {
                let path = nvidia.join(folder).join(name);
                if path.exists() {
                    out.push(path);
                }
            }
        }
    }
    out
}

/// Compiles the probe source for `arch` and returns the cubin.
fn compile(arch: &str) -> Result<Vec<u8>> {
    type Prog = *mut c_void;
    let mut libs = Vec::new();
    let mut found = None;
    for path in nvrtc_candidates() {
        // NVRTC opens its builtins library by name, so load it first when
        // it sits beside a library outside the loader's path.
        if let Some(dir) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
                let file = entry.path();
                if file
                    .file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("libnvrtc-builtins.so."))
                {
                    let flags = libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL;
                    if let Ok(lib) =
                        unsafe { libloading::os::unix::Library::open(Some(&file), flags) }
                    {
                        libs.push(libloading::Library::from(lib));
                    }
                }
            }
        }
        if let Ok(lib) = unsafe { libloading::Library::new(&path) } {
            found = Some(lib);
            break;
        }
    }
    let lib = found.context("NVRTC not found (see docs/building.md)")?;
    unsafe {
        let create: unsafe extern "C" fn(
            *mut Prog,
            *const c_char,
            *const c_char,
            c_int,
            *const *const c_char,
            *const *const c_char,
        ) -> c_int = sym!(lib, "nvrtcCreateProgram");
        let compile: unsafe extern "C" fn(Prog, c_int, *const *const c_char) -> c_int =
            sym!(lib, "nvrtcCompileProgram");
        let log_size: unsafe extern "C" fn(Prog, *mut usize) -> c_int =
            sym!(lib, "nvrtcGetProgramLogSize");
        let get_log: unsafe extern "C" fn(Prog, *mut c_char) -> c_int =
            sym!(lib, "nvrtcGetProgramLog");
        let cubin_size: unsafe extern "C" fn(Prog, *mut usize) -> c_int =
            sym!(lib, "nvrtcGetCUBINSize");
        let get_cubin: unsafe extern "C" fn(Prog, *mut c_char) -> c_int =
            sym!(lib, "nvrtcGetCUBIN");
        let source = CString::new(SOURCE)?;
        let mut prog: Prog = std::ptr::null_mut();
        if create(
            &mut prog,
            source.as_ptr(),
            c"power_probe.cu".as_ptr(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        ) != 0
        {
            bail!("nvrtcCreateProgram failed");
        }
        let options = [CString::new(format!("--gpu-architecture={arch}"))?];
        let pointers: Vec<*const c_char> = options.iter().map(|o| o.as_ptr()).collect();
        let status = compile(prog, pointers.len() as c_int, pointers.as_ptr());
        let mut size = 0;
        log_size(prog, &mut size);
        let mut log = vec![0u8; size.max(1)];
        get_log(prog, log.as_mut_ptr() as *mut c_char);
        if status != 0 {
            bail!("NVRTC: {}", String::from_utf8_lossy(&log));
        }
        cubin_size(prog, &mut size);
        let mut cubin = vec![0u8; size];
        get_cubin(prog, cubin.as_mut_ptr() as *mut c_char);
        drop(libs);
        Ok(cubin)
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let name = args.get(1).map(String::as_str).unwrap_or("fma");
    let seconds: f64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(30.0);
    let cubin_dir = args.get(3);
    let cu = Cuda::load()?;
    let mut device = 0;
    let mut context: Ptr = std::ptr::null_mut();
    let (mut sms, mut major, mut minor) = (0, 0, 0);
    let mut device_name = [0 as c_char; 256];
    unsafe {
        check((cu.init)(0), "cuInit")?;
        check((cu.device_get)(&mut device, 0), "cuDeviceGet")?;
        check(
            (cu.device_get_name)(device_name.as_mut_ptr(), 256, device),
            "cuDeviceGetName",
        )?;
        check(
            (cu.device_get_attribute)(&mut sms, 16, device),
            "multiprocessors",
        )?;
        check((cu.device_get_attribute)(&mut major, 75, device), "major")?;
        check((cu.device_get_attribute)(&mut minor, 76, device), "minor")?;
        check(
            (cu.primary_ctx_retain)(&mut context, device),
            "cuDevicePrimaryCtxRetain",
        )?;
        check((cu.ctx_set_current)(context), "cuCtxSetCurrent")?;
    }
    let device_name = unsafe { CStr::from_ptr(device_name.as_ptr()) }
        .to_string_lossy()
        .into_owned();
    eprintln!("device: {device_name}, {sms} SMs, sm_{major}{minor}");
    if name == "idle" {
        let start = Instant::now();
        while start.elapsed().as_secs_f64() < seconds {
            std::thread::sleep(std::time::Duration::from_millis(100));
        }
        println!("idle: held a context for {seconds:.0} s");
        return Ok(());
    }
    let mode = mode(name)?;
    let cubin = compile(&format!("sm_{major}{minor}"))?;
    if let Some(dir) = cubin_dir {
        std::fs::write(Path::new(dir).join(format!("{name}.cubin")), &cubin)?;
    }
    let mut module: Ptr = std::ptr::null_mut();
    let mut function: Ptr = std::ptr::null_mut();
    let entry = CString::new(mode.entry)?;
    let (mut blocks_per_sm, mut regs) = (0, 0);
    unsafe {
        check(
            (cu.module_load_data)(&mut module, cubin.as_ptr() as *const c_void),
            "cuModuleLoadData",
        )?;
        check(
            (cu.module_get_function)(&mut function, module, entry.as_ptr()),
            "cuModuleGetFunction",
        )?;
        check(
            (cu.occupancy)(&mut blocks_per_sm, function, BLOCK as c_int, 0),
            "occupancy",
        )?;
        check((cu.func_get_attribute)(&mut regs, 4, function), "registers")?;
    }
    let grid = blocks_per_sm as u32 * sms as u32;
    let threads = grid as u64 * BLOCK as u64;
    let warps = threads / 32;
    let mut out: u64 = 0;
    unsafe {
        check(
            (cu.mem_alloc)(&mut out, (threads as usize + 2) * 8),
            "cuMemAlloc",
        )?
    };
    eprintln!(
        "{name}: {} per thread per iteration, {} SASS; {regs} registers, {blocks_per_sm} blocks of {BLOCK} per SM ({} warps per SM)",
        mode.ops,
        mode.sass_per_iter,
        blocks_per_sm as u32 * BLOCK / 32
    );
    // Returns seconds and the SM clock block 0 saw.
    let launch = |iters: i32| -> Result<(f64, f64)> {
        let mut pointer = out;
        let mut iters = iters;
        let mut params: [Ptr; 2] = [
            &mut pointer as *mut u64 as Ptr,
            &mut iters as *mut i32 as Ptr,
        ];
        let start = Instant::now();
        let mut stamp = [0u64; 2];
        unsafe {
            check(
                (cu.launch)(
                    function,
                    grid,
                    1,
                    1,
                    BLOCK,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    params.as_mut_ptr(),
                    std::ptr::null_mut(),
                ),
                "cuLaunchKernel",
            )?;
            check((cu.ctx_synchronize)(), "cuCtxSynchronize")?;
            check(
                (cu.memcpy_dtoh)(stamp.as_mut_ptr() as *mut c_void, out, 16),
                "cuMemcpyDtoH",
            )?;
        }
        let mhz = stamp[0] as f64 / stamp[1].max(1) as f64 * 1e3;
        Ok((start.elapsed().as_secs_f64(), mhz))
    };
    // The first launch pays for loading the module; size the rest from the
    // second.
    launch(200)?;
    let (t, _) = launch(2000)?;
    let iters = ((2000.0 * 0.2 / t.max(1e-6)) as i32).clamp(1, 1 << 30);
    let per_launch = warps as f64 * iters as f64 * mode.sass_per_iter as f64;
    let start = Instant::now();
    let (mut launches, mut busy) = (0u64, 0.0);
    let (mut window_launches, mut window_busy, mut window_start) = (0u64, 0.0, 0.0);
    let mut clocks = Vec::new();
    while start.elapsed().as_secs_f64() < seconds {
        let (t, mhz) = launch(iters)?;
        launches += 1;
        busy += t;
        window_launches += 1;
        window_busy += t;
        clocks.push(mhz);
        let now = start.elapsed().as_secs_f64();
        if now - window_start >= 1.0 {
            let rate = window_launches as f64 * per_launch / window_busy;
            println!(
                "t {now:5.1} s: {:.1} G warp-instructions/s, kernel clock {mhz:.0} MHz, {:.2} per SM per clock",
                rate / 1e9,
                rate / (sms as f64 * mhz * 1e6)
            );
            (window_launches, window_busy, window_start) = (0, 0.0, now);
        }
    }
    let rate = launches as f64 * per_launch / busy;
    let mean_mhz = clocks.iter().sum::<f64>() / clocks.len().max(1) as f64;
    println!(
        "{name} summary: {launches} launches of {iters} iterations in {busy:.1} s, {:.1} G warp-instructions/s, mean kernel clock {mean_mhz:.0} MHz, {:.2} warp-instructions per SM per clock",
        rate / 1e9,
        rate / (sms as f64 * mean_mhz * 1e6)
    );
    Ok(())
}
