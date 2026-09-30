//! The stub of the per-lane kernel (docs/plan-2m.md, section 6 item 4):
//! compiles `shaders/lane_stub.cu` with NVRTC, reports registers, spills,
//! stack frame and local memory, then runs synthetic creatures on the
//! primary GPU and prints creature-steps per second.
//!
//! Usage: lane_stub [key=value ...]
//!   count=262144 steps=300 nodes=8 muscles=19 substeps=2 nb=3 mpl=16
//!   rounds=3 block=128 min_blocks=4 repeat=3 arch=sm_89 seed=1
//!   report        compile and report only (no GPU time)
//!   cubin=PATH, src=PATH, log   write the cubin, the source, the full log
//!   first         print creature 0 and the first non-finite creatures
//! Developer checks (defines through EVOLUTION_NVRTC_EXTRA, see the kernel
//! header): -DTRACE with trace=SECONDS prints each warp's last checkpoint
//! from mapped memory without waiting (for hangs); -DDUMP with trace=2 dump
//! checks creature 0's rod solve against a dense A; -DDUMPW=STEP with
//! trace=3 dumpw checks its Delassus matrix and free contact velocities.
//!
//! Nothing here is the game's physics; the kernel is a stand-in with the
//! planned state layout and instruction mix. The CUDA driver and NVRTC are
//! loaded at run time as in `src/cuda_engine.rs`, which this file does not
//! touch.
#![allow(dead_code)]
use anyhow::{Context, Result, bail};
use std::{
    collections::HashMap,
    ffi::{CStr, CString, c_char, c_int, c_uint, c_void},
    path::{Path, PathBuf},
    time::Instant,
};

pub const SOURCE: &str = include_str!("../shaders/lane_stub.cu");

type CuResult = c_int;
type Ptr = u64;

pub struct Cuda {
    init: unsafe extern "C" fn(c_uint) -> CuResult,
    device_get: unsafe extern "C" fn(*mut c_int, c_int) -> CuResult,
    device_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, c_int) -> CuResult,
    primary_ctx_retain: unsafe extern "C" fn(*mut *mut c_void, c_int) -> CuResult,
    ctx_set_current: unsafe extern "C" fn(*mut c_void) -> CuResult,
    ctx_synchronize: unsafe extern "C" fn() -> CuResult,
    module_load_data: unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> CuResult,
    module_get_function: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const c_char) -> CuResult,
    mem_alloc: unsafe extern "C" fn(*mut Ptr, usize) -> CuResult,
    mem_free: unsafe extern "C" fn(Ptr) -> CuResult,
    htod: unsafe extern "C" fn(Ptr, *const c_void, usize) -> CuResult,
    dtoh: unsafe extern "C" fn(*mut c_void, Ptr, usize) -> CuResult,
    memset_d32: unsafe extern "C" fn(Ptr, c_uint, usize) -> CuResult,
    launch: unsafe extern "C" fn(
        *mut c_void,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        *mut c_void,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CuResult,
    event_create: unsafe extern "C" fn(*mut *mut c_void, c_uint) -> CuResult,
    event_record: unsafe extern "C" fn(*mut c_void, *mut c_void) -> CuResult,
    event_synchronize: unsafe extern "C" fn(*mut c_void) -> CuResult,
    event_elapsed: unsafe extern "C" fn(*mut f32, *mut c_void, *mut c_void) -> CuResult,
    func_get_attribute: unsafe extern "C" fn(*mut c_int, c_int, *mut c_void) -> CuResult,
    occupancy: unsafe extern "C" fn(*mut c_int, *mut c_void, c_int, usize) -> CuResult,
    get_error_name: unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult,
    host_alloc: unsafe extern "C" fn(*mut *mut c_void, usize, c_uint) -> CuResult,
    host_device_pointer: unsafe extern "C" fn(*mut Ptr, *mut c_void, c_uint) -> CuResult,
    _lib: libloading::Library,
}

macro_rules! sym {
    ($lib:expr, $name:literal) => {
        *$lib
            .get(concat!($name, "\0").as_bytes())
            .with_context(|| format!("symbol {}", $name))?
    };
}

impl Cuda {
    pub fn load() -> Result<Self> {
        let lib = unsafe { libloading::Library::new("libcuda.so.1") }
            .or_else(|_| unsafe { libloading::Library::new("libcuda.so") })
            .context("libcuda not found")?;
        let cu = unsafe {
            Self {
                init: sym!(lib, "cuInit"),
                device_get: sym!(lib, "cuDeviceGet"),
                device_get_attribute: sym!(lib, "cuDeviceGetAttribute"),
                primary_ctx_retain: sym!(lib, "cuDevicePrimaryCtxRetain"),
                ctx_set_current: sym!(lib, "cuCtxSetCurrent"),
                ctx_synchronize: sym!(lib, "cuCtxSynchronize"),
                module_load_data: sym!(lib, "cuModuleLoadData"),
                module_get_function: sym!(lib, "cuModuleGetFunction"),
                mem_alloc: sym!(lib, "cuMemAlloc_v2"),
                mem_free: sym!(lib, "cuMemFree_v2"),
                htod: sym!(lib, "cuMemcpyHtoD_v2"),
                dtoh: sym!(lib, "cuMemcpyDtoH_v2"),
                memset_d32: sym!(lib, "cuMemsetD32_v2"),
                launch: sym!(lib, "cuLaunchKernel"),
                event_create: sym!(lib, "cuEventCreate"),
                event_record: sym!(lib, "cuEventRecord"),
                event_synchronize: sym!(lib, "cuEventSynchronize"),
                event_elapsed: sym!(lib, "cuEventElapsedTime"),
                func_get_attribute: sym!(lib, "cuFuncGetAttribute"),
                occupancy: sym!(lib, "cuOccupancyMaxActiveBlocksPerMultiprocessor"),
                get_error_name: sym!(lib, "cuGetErrorName"),
                host_alloc: sym!(lib, "cuMemHostAlloc"),
                host_device_pointer: sym!(lib, "cuMemHostGetDevicePointer_v2"),
                _lib: lib,
            }
        };
        cu.check(unsafe { (cu.init)(0) }, "cuInit")?;
        Ok(cu)
    }

    pub fn check(&self, r: CuResult, what: &str) -> Result<()> {
        if r == 0 {
            return Ok(());
        }
        let mut name: *const c_char = std::ptr::null();
        unsafe { (self.get_error_name)(r, &mut name) };
        let name = if name.is_null() {
            format!("error {r}")
        } else {
            unsafe { CStr::from_ptr(name) }.to_string_lossy().into_owned()
        };
        bail!("CUDA {what}: {name}")
    }

    /// Makes device 0's primary context current. `EVOLUTION_DEVICES=primary`
    /// keeps CUDA on the RTX 4060 (the Radeon has no CUDA anyway).
    pub fn open(&self) -> Result<c_int> {
        let mut dev = 0;
        self.check(unsafe { (self.device_get)(&mut dev, 0) }, "cuDeviceGet")?;
        let mut ctx = std::ptr::null_mut();
        self.check(unsafe { (self.primary_ctx_retain)(&mut ctx, dev) }, "cuDevicePrimaryCtxRetain")?;
        self.check(unsafe { (self.ctx_set_current)(ctx) }, "cuCtxSetCurrent")?;
        Ok(dev)
    }

    pub fn attribute(&self, dev: c_int, attr: c_int) -> Result<i32> {
        let mut v = 0;
        self.check(unsafe { (self.device_get_attribute)(&mut v, attr, dev) }, "cuDeviceGetAttribute")?;
        Ok(v)
    }

    pub fn function(&self, cubin: &[u8], name: &str) -> Result<*mut c_void> {
        let mut module = std::ptr::null_mut();
        self.check(
            unsafe { (self.module_load_data)(&mut module, cubin.as_ptr() as *const c_void) },
            "cuModuleLoadData",
        )?;
        let mut f = std::ptr::null_mut();
        let name = CString::new(name)?;
        self.check(
            unsafe { (self.module_get_function)(&mut f, module, name.as_ptr()) },
            "cuModuleGetFunction",
        )?;
        Ok(f)
    }

    /// (registers, local bytes per thread, static shared bytes per block).
    pub fn function_resources(&self, f: *mut c_void) -> Result<(i32, i32, i32)> {
        let get = |a: c_int| -> Result<i32> {
            let mut v = 0;
            self.check(unsafe { (self.func_get_attribute)(&mut v, a, f) }, "cuFuncGetAttribute")?;
            Ok(v)
        };
        // CU_FUNC_ATTRIBUTE_NUM_REGS 4, LOCAL_SIZE_BYTES 3, SHARED_SIZE_BYTES 1.
        Ok((get(4)?, get(3)?, get(1)?))
    }

    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<Ptr> {
        let bytes = std::mem::size_of_val(data).max(4);
        let mut p = 0;
        self.check(unsafe { (self.mem_alloc)(&mut p, bytes) }, "cuMemAlloc")?;
        if !data.is_empty() {
            self.check(
                unsafe { (self.htod)(p, data.as_ptr() as *const c_void, std::mem::size_of_val(data)) },
                "cuMemcpyHtoD",
            )?;
        }
        Ok(p)
    }

    pub fn alloc(&self, bytes: usize) -> Result<Ptr> {
        let mut p = 0;
        self.check(unsafe { (self.mem_alloc)(&mut p, bytes.max(4)) }, "cuMemAlloc")?;
        Ok(p)
    }
}

/// NVRTC, found as `src/cuda_engine.rs` finds it.
pub struct Nvrtc {
    create: unsafe extern "C" fn(
        *mut *mut c_void,
        *const c_char,
        *const c_char,
        c_int,
        *const *const c_char,
        *const *const c_char,
    ) -> c_int,
    compile: unsafe extern "C" fn(*mut c_void, c_int, *const *const c_char) -> c_int,
    log_size: unsafe extern "C" fn(*mut c_void, *mut usize) -> c_int,
    log: unsafe extern "C" fn(*mut c_void, *mut c_char) -> c_int,
    cubin_size: unsafe extern "C" fn(*mut c_void, *mut usize) -> c_int,
    cubin: unsafe extern "C" fn(*mut c_void, *mut c_char) -> c_int,
    destroy: unsafe extern "C" fn(*mut *mut c_void) -> c_int,
    _builtins: Option<libloading::Library>,
    _lib: libloading::Library,
}

fn nvrtc_candidates() -> Vec<PathBuf> {
    if let Some(p) = std::env::var_os("EVOLUTION_NVRTC") {
        return vec![PathBuf::from(p)];
    }
    let mut v: Vec<PathBuf> = ["libnvrtc.so.13", "libnvrtc.so.12", "libnvrtc.so", "/usr/local/cuda/lib64/libnvrtc.so"]
        .iter()
        .map(PathBuf::from)
        .collect();
    if let Some(home) = std::env::var_os("HOME") {
        let venv = Path::new(&home).join(".local/share/evolution-cuda/venv/lib");
        for py in std::fs::read_dir(&venv).into_iter().flatten().flatten() {
            let nv = py.path().join("site-packages/nvidia");
            for (folder, name) in [("cu13/lib", "libnvrtc.so.13"), ("cuda_nvrtc/lib", "libnvrtc.so.12")] {
                let path = nv.join(folder).join(name);
                if path.exists() {
                    v.push(path);
                }
            }
        }
    }
    v
}

impl Nvrtc {
    pub fn load() -> Result<Self> {
        for path in nvrtc_candidates() {
            let builtins = path.parent().and_then(|dir| {
                let file = std::fs::read_dir(dir).ok()?.flatten().map(|e| e.path()).find(|f| {
                    f.file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|n| n.starts_with("libnvrtc-builtins.so."))
                })?;
                let flags = libloading::os::unix::RTLD_NOW | libloading::os::unix::RTLD_GLOBAL;
                unsafe { libloading::os::unix::Library::open(Some(&file), flags) }
                    .ok()
                    .map(libloading::Library::from)
            });
            let Ok(lib) = (unsafe { libloading::Library::new(&path) }) else { continue };
            return unsafe {
                Ok(Self {
                    create: sym!(lib, "nvrtcCreateProgram"),
                    compile: sym!(lib, "nvrtcCompileProgram"),
                    log_size: sym!(lib, "nvrtcGetProgramLogSize"),
                    log: sym!(lib, "nvrtcGetProgramLog"),
                    cubin_size: sym!(lib, "nvrtcGetCUBINSize"),
                    cubin: sym!(lib, "nvrtcGetCUBIN"),
                    destroy: sym!(lib, "nvrtcDestroyProgram"),
                    _builtins: builtins,
                    _lib: lib,
                })
            };
        }
        bail!("NVRTC not found")
    }

    /// Compiles to a cubin; returns it with the log (ptxas statistics).
    pub fn compile(&self, source: &str, name: &str, options: &[String]) -> Result<(Vec<u8>, String)> {
        let src = CString::new(source)?;
        let name = CString::new(name)?;
        let opts: Vec<CString> = options.iter().map(|o| CString::new(o.as_str())).collect::<Result<_, _>>()?;
        let ptrs: Vec<*const c_char> = opts.iter().map(|o| o.as_ptr()).collect();
        unsafe {
            let mut prog = std::ptr::null_mut();
            if (self.create)(&mut prog, src.as_ptr(), name.as_ptr(), 0, std::ptr::null(), std::ptr::null()) != 0 {
                bail!("nvrtcCreateProgram failed");
            }
            let status = (self.compile)(prog, ptrs.len() as c_int, ptrs.as_ptr());
            let mut size = 0usize;
            (self.log_size)(prog, &mut size);
            let mut log = vec![0u8; size.max(1)];
            (self.log)(prog, log.as_mut_ptr() as *mut c_char);
            let log = CStr::from_bytes_until_nul(&log).map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            let out = if status != 0 {
                Err(anyhow::anyhow!("NVRTC compile failed:\n{log}"))
            } else {
                let mut n = 0usize;
                (self.cubin_size)(prog, &mut n);
                let mut bin = vec![0u8; n];
                (self.cubin)(prog, bin.as_mut_ptr() as *mut c_char);
                Ok((bin, log))
            };
            (self.destroy)(&mut prog);
            out
        }
    }
}

/// The compile options of the game's kernel (`cuda_engine::compile_report`).
pub fn options(arch: &str) -> Vec<String> {
    let mut o: Vec<String> = vec![
        format!("--gpu-architecture={arch}"),
        "--std=c++17".into(),
        "--prec-div=false".into(),
        "--prec-sqrt=false".into(),
        "--fmad=true".into(),
        "--extra-device-vectorization".into(),
        "--ptxas-options=-v".into(),
    ];
    if let Ok(extra) = std::env::var("EVOLUTION_NVRTC_EXTRA") {
        o.extend(extra.split_whitespace().map(String::from));
    }
    o
}

/// The stub's source with its defines in front.
pub fn source(defines: &[(&str, String)]) -> String {
    let mut s = String::new();
    for (k, v) in defines {
        s.push_str(&format!("#define {k} {v}\n"));
    }
    s.push_str(SOURCE);
    s
}

/// ptxas's numbers from an NVRTC log, one entry per function:
/// registers, stack frame, spill stores and loads.
pub fn ptxas_summary(log: &str) -> String {
    let number = |line: &str, before: &str| -> Option<u32> {
        let words: Vec<&str> = line.split_whitespace().collect();
        words.windows(2).find(|w| w[1].starts_with(before)).and_then(|w| w[0].parse().ok())
    };
    let mut out = Vec::new();
    let mut frame = None;
    for line in log.lines() {
        if line.contains("stack frame") {
            frame = Some((number(line, "bytes").unwrap_or(0), line));
        }
        if line.contains("Used") && line.contains("registers") {
            let regs = line.split("Used").nth(1).and_then(|r| r.split_whitespace().next()).unwrap_or("?");
            let (stack, spills) = frame.take().unwrap_or((0, ""));
            let words: Vec<&str> = spills.split_whitespace().collect();
            let spill = |kind: &str| {
                words.windows(3).find(|w| w[2] == kind).map(|w| w[0]).unwrap_or("0").to_owned()
            };
            out.push(format!(
                "{regs} registers, {stack} B stack frame, {} B spill stores, {} B spill loads",
                spill("stores,"),
                spill("loads")
            ));
        }
    }
    out.join(" | ")
}

/// Largest spill (stores or loads) over the log's functions, in bytes.
pub fn max_spill(log: &str) -> u32 {
    let mut worst = 0;
    for line in log.lines().filter(|l| l.contains("spill")) {
        let words: Vec<&str> = line.split_whitespace().collect();
        for w in words.windows(3) {
            if w[1] == "bytes" && w[2] == "spill" {
                worst = worst.max(w[0].parse().unwrap_or(0));
            }
        }
    }
    worst
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn f(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f()
    }
    fn below(&mut self, n: u32) -> u32 {
        (self.next() % n as u64) as u32
    }
}

const RF: usize = 32;

/// Synthetic creatures in the stub's layout.
pub struct Batch {
    pub heads: Vec<[u32; 4]>,
    pub lanes: Vec<f32>,
    pub msa: Vec<[f32; 4]>,
    pub msb: Vec<[f32; 2]>,
    pub mss: Vec<[f32; 4]>,
}

fn unorm2(a: f32, b: f32) -> f32 {
    let q = |x: f32| (x.clamp(0.0, 1.0) * 65535.0).round() as u32;
    f32::from_bits(q(a) | (q(b) << 16))
}
fn bf2(lo: f32, hi: f32) -> f32 {
    let t = |x: f32| (x.to_bits().wrapping_add(0x8000)) >> 16;
    f32::from_bits(t(lo) | (t(hi) << 16))
}

pub fn batch(count: usize, nodes: usize, muscles: usize, mpl: usize, nb: usize, seed: u64) -> Batch {
    const W: usize = 2;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ seed.wrapping_mul(0x2545_f491_4f6c_dd1d));
    let mut b = Batch {
        heads: Vec::with_capacity(count),
        lanes: vec![0.0; count * RF * W],
        msa: vec![[0.0; 4]; count * mpl * W],
        msb: vec![[0.0; 2]; count * mpl * W],
        mss: vec![[0.0; 4]; count * 2 * mpl * W],
    };
    for c in 0..count {
        // A breadth-first tree: node 0 the head with only the neck (node 1);
        // parents never decrease; at most `nb` children per node.
        let n = nodes.clamp(2, 8);
        let mut parent = vec![0usize; n];
        let mut children = vec![0usize; n];
        children[0] = 1;
        let mut cursor = 1;
        for i in 2..n {
            while children[cursor] >= nb || (children[cursor] > 0 && rng.f() < 0.45 && cursor + 1 < i) {
                cursor += 1;
            }
            parent[i] = cursor;
            children[cursor] += 1;
        }
        let mut x = vec![0.0f32; n];
        let mut y = vec![0.0f32; n];
        let mut rad = vec![0.0f32; n];
        let mut invm = vec![0.0f32; n];
        y[0] = 0.8;
        for i in 0..n {
            rad[i] = rng.range(0.03, 0.07);
            invm[i] = 1.0 / rng.range(0.1, 0.6);
            if i > 0 {
                let angle = rng.range(-2.8, -0.35);
                let l = rng.range(0.15, 0.35);
                x[i] = x[parent[i]] + l * angle.cos();
                y[i] = y[parent[i]] + l * angle.sin();
            }
        }
        let low = (0..n).map(|i| y[i] - rad[i]).fold(f32::MAX, f32::min);
        for yi in y.iter_mut() {
            *yi += 0.02 - low;
        }
        // Per lane record: [field][lane], fields k-major as the kernel reads.
        let rec = &mut b.lanes[c * RF * W..(c + 1) * RF * W];
        let mut rank = vec![0usize; n];
        let mut seen = vec![0usize; n];
        for g in 1..n {
            rank[g] = seen[parent[g]];
            seen[parent[g]] += 1;
        }
        for g in 0..8 {
            let (lane, k) = (g / 4, g % 4);
            let put = |rec: &mut [f32], field: usize, v: f32| rec[(field + k) * W + lane] = v;
            if g < n {
                put(rec, 0, invm[g]);
                put(rec, 4, rad[g]);
                put(rec, 8, rng.range(0.6, 1.0));
                put(rec, 12, x[g]);
                put(rec, 16, y[g]);
            }
            let mut topo = 0u32;
            if g >= 1 && g < n {
                let a = parent[g];
                let len = ((x[g] - x[a]).powi(2) + (y[g] - y[a]).powi(2)).sqrt();
                put(rec, 20, len);
                let prod = if a >= 1 { (a - 1) as u32 } else { 31 };
                let gp = if a >= 1 { parent[a] as u32 } else { 31 };
                let sr = rank[g].min(nb - 1) as u32;
                topo = a as u32 | prod << 5 | sr << 10 | gp << 12 | 1 << 17;
                put(rec, 28, invm[a]);
            }
            put(rec, 24, f32::from_bits(topo));
        }
        // Muscles: between two rods that share a node, alternating lanes.
        let m = muscles.min(2 * mpl);
        let per = [m.div_ceil(2), m / 2];
        b.heads.push([n as u32 | (m as u32) << 8 | (per[0] as u32) << 16 | (per[1] as u32) << 24, 0, 0, 0]);
        for i in 0..m {
            let (lane, k) = (i % 2, i / 2);
            let ra = 1 + rng.below((n - 1) as u32) as usize;
            let rb = {
                let near: Vec<usize> = (1..n)
                    .filter(|&j| j != ra && (parent[j] == parent[ra] || parent[j] == ra || parent[ra] == j))
                    .collect();
                if near.is_empty() { 1 + (ra % (n - 1)) } else { near[rng.below(near.len() as u32) as usize] }
            };
            let (ta, tb) = (rng.range(0.2, 0.8), rng.range(0.2, 0.8));
            let at = |j: usize, t: f32| (x[parent[j]] + (x[j] - x[parent[j]]) * t, y[parent[j]] + (y[j] - y[parent[j]]) * t);
            let (pa, pb) = (at(ra, ta), at(rb, tb));
            let dist = ((pa.0 - pb.0).powi(2) + (pa.1 - pb.1).powi(2)).sqrt();
            let sensor = if rng.f() < 0.3 { (1u32 << 25) | (rng.below(n as u32) << 20) } else { 0 };
            let pk = parent[ra] as u32 | (ra as u32) << 5 | (parent[rb] as u32) << 10 | (rb as u32) << 15 | sensor;
            let idx = (c * mpl + k) * W + lane;
            b.msa[idx] = [f32::from_bits(pk), unorm2(ta, tb), rng.range(2.0, 10.0), rng.range(50.0, 200.0)];
            b.msb[idx] = [dist * rng.range(1.05, 1.3), bf2(rng.range(0.0, 2.0), rng.range(0.1, 1.0))];
            let period = rng.range(0.5, 2.0);
            let duty = rng.range(0.3, 0.7);
            let s = (c * 2 * mpl + 2 * k) * W + lane;
            b.mss[s] = [1.0 / period, rng.f(), duty, 1.0 / duty];
            b.mss[s + W] = [1.0 / (1.0 - duty), rng.range(0.02, 0.1), rng.range(50.0, 300.0), 0.0];
        }
    }
    b
}

/// Checks a dumped creature's rod solve (the source built with -DDUMP):
/// A mu0 = rhs0 with A = J M^-1 J^T built from the definition.
fn check_dump(f: &[f32]) {
    let lane = |l: usize, field: usize, k: usize| f[l * 32 + field * 4 + k];
    let mut rods = Vec::new();
    let mut invm = [0f32; 8];
    for l in 0..2 {
        for k in 0..4 {
            invm[l * 4 + k] = lane(l, 4, k);
            let topo = lane(l, 5, k).to_bits();
            if (topo >> 17) & 1 == 1 {
                let child = l * 4 + k;
                rods.push((child, (topo & 31) as usize, [lane(l, 0, k), lane(l, 1, k)], lane(l, 2, k), lane(l, 3, k), lane(l, 6, k), lane(l, 7, k), topo));
            }
        }
    }
    let n = rods.len();
    let mut a = vec![vec![0f64; n]; n];
    let jac = |r: &(usize, usize, [f32; 2], f32, f32, f32, f32, u32), node: usize| -> [f64; 2] {
        let s = if node == r.0 { 1.0 } else if node == r.1 { -1.0 } else { 0.0 };
        [s * r.2[0] as f64, s * r.2[1] as f64]
    };
    for i in 0..n {
        for j in 0..n {
            for node in 0..8 {
                let (ji, jj) = (jac(&rods[i], node), jac(&rods[j], node));
                a[i][j] += (ji[0] * jj[0] + ji[1] * jj[1]) * invm[node] as f64;
            }
        }
    }
    let mut worst = 0f64;
    for i in 0..n {
        let ax: f64 = (0..n).map(|j| a[i][j] * rods[j].4 as f64).sum();
        let r = ax - rods[i].3 as f64;
        worst = worst.max(r.abs());
        println!(
            "rod to node {} pivot {} parent {} sr {}: A_ii {:.4} 1/invD {:.4} Lp {:.4}, (A mu0 - rhs0) {:.3e} of rhs {:.3e}",
            rods[i].0, rods[i].1, (rods[i].7 >> 5) & 31, (rods[i].7 >> 10) & 3, a[i][i],
            if rods[i].5 != 0.0 { 1.0 / rods[i].5 } else { 0.0 }, rods[i].6, r, rods[i].3
        );
    }
    println!("worst residual {worst:.3e}");
}

fn solve_dense(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for i in 0..n {
        let p = (i..n).max_by(|&x, &y| a[x][i].abs().total_cmp(&a[y][i].abs())).unwrap();
        a.swap(i, p);
        b.swap(i, p);
        for r in i + 1..n {
            let f = a[r][i] / a[i][i];
            for c in i..n {
                a[r][c] -= f * a[i][c];
            }
            b[r] -= f * b[i];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = (b[i] - (i + 1..n).map(|c| a[i][c] * x[c]).sum::<f64>()) / a[i][i];
    }
    x
}

/// Checks a dumped Delassus system (the source built with -DDUMPW=step):
/// W = W0 - R^T A^-1 R and the free contact velocities, from the definition.
fn check_dump_w(f: &[f32], rad: &[f32]) {
    const HS: f64 = 1.0 / 120.0;
    let lane = |l: usize, field: usize, k: usize| f[l * 32 + field * 4 + k] as f64;
    let (mut px, mut py, mut vx, mut vy, mut im) = ([0f64; 8], [0f64; 8], [0f64; 8], [0f64; 8], [0f64; 8]);
    let mut rods: Vec<(usize, usize, [f64; 2])> = Vec::new();
    for l in 0..2 {
        for k in 0..4 {
            let g = l * 4 + k;
            px[g] = lane(l, 0, k); py[g] = lane(l, 1, k); vx[g] = lane(l, 2, k); vy[g] = lane(l, 3, k); im[g] = lane(l, 4, k);
            let t = (lane(l, 5, k) as f32).to_bits();
            if (t >> 17) & 1 == 1 {
                rods.push((g, (t & 31) as usize, [lane(l, 6, k), lane(l, 7, k)]));
            }
        }
    }
    let word = f[64].to_bits();
    let mut rows: Vec<(usize, [f64; 2])> = Vec::new();
    for c in 0..4 {
        let e = (word >> (6 * c)) & 63;
        let g = (e & 31) as usize;
        let on = e & 32 != 0;
        rows.push((if on { g } else { 99 }, [0.0, 1.0]));
        rows.push((if on { g } else { 99 }, [1.0, 0.0]));
    }
    let n = rods.len();
    let jr = |j: usize, node: usize| -> [f64; 2] {
        let s = if node == rods[j].0 { 1.0 } else if node == rods[j].1 { -1.0 } else { 0.0 };
        [s * rods[j].2[0], s * rods[j].2[1]]
    };
    let je = |e: usize, node: usize| -> [f64; 2] { if rows[e].0 == node { rows[e].1 } else { [0.0, 0.0] } };
    let dot = |a: [f64; 2], b: [f64; 2]| a[0] * b[0] + a[1] * b[1];
    let a_mat: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| (0..8).map(|g| dot(jr(i, g), jr(j, g)) * im[g]).sum()).collect()).collect();
    let r_mat: Vec<Vec<f64>> = (0..n).map(|i| (0..8).map(|e| (0..8).map(|g| dot(jr(i, g), je(e, g)) * im[g]).sum()).collect()).collect();
    let mut worst = 0f64;
    let mut wd = vec![vec![0f64; 8]; 8];
    for e in 0..8 {
        let col: Vec<f64> = (0..n).map(|i| r_mat[i][e]).collect();
        let m = solve_dense(a_mat.clone(), col);
        for f2 in 0..8 {
            let w0: f64 = (0..8).map(|g| dot(je(e, g), je(f2, g)) * im[g]).sum();
            wd[e][f2] = w0 - (0..n).map(|i| r_mat[i][f2] * m[i]).sum::<f64>();
        }
    }
    for i in 0..8 {
        for j in 0..=i {
            let idx = i * (i + 1) / 2 + j;
            let mut g = f[65 + idx] as f64;
            if i == j { g -= f[109 + i / 2] as f64 * 1.1; }
            worst = worst.max((g - wd[i][j]).abs());
        }
    }
    // Free contact velocities after the rods: w = J v* + R^T A^-1 (c - J_r v*).
    let rhs: Vec<f64> = (0..n).map(|i| {
        let (b, a) = (rods[i].0, rods[i].1);
        let rel = [vx[b] - vx[a], vy[b] - vy[a]];
        let rn = dot(rel, rods[i].2);
        let l = ((px[b] - px[a]).powi(2) + (py[b] - py[a]).powi(2)).sqrt();
        -rn - (dot(rel, rel) - rn * rn).max(0.0) * HS / l
    }).collect();
    let mu0 = solve_dense(a_mat.clone(), rhs);
    for c in 0..4 {
        let g = rows[2 * c].0;
        if g == 99 { continue; }
        let wn = vy[g] + (0..n).map(|i| r_mat[i][2 * c] * mu0[i]).sum::<f64>();
        let gap = py[g] - rad[g] as f64;
        let tgt = if gap >= 0.0 { -gap / HS } else { -gap * 0.2 / HS };
        println!("slot {c} node {g}: gpu b0 normal {:.5} host {:.5}", f[101 + 2 * c], tgt - wn);
    }
    for i in 0..8 {
        let gpu: Vec<String> = (0..=i).map(|j| {
            let idx = i * (i + 1) / 2 + j;
            let mut g = f[65 + idx] as f64;
            if i == j { g -= f[109 + i / 2] as f64 * 1.1; }
            format!("{g:7.3}")
        }).collect();
        let host: Vec<String> = (0..=i).map(|j| format!("{:7.3}", wd[i][j])).collect();
        println!("row {i}: gpu {} | host {}", gpu.join(" "), host.join(" "));
    }
    println!("W: worst |gpu - host| {worst:.3e}; host diag {:?}", (0..8).map(|i| (wd[i][i] * 1e3).round() / 1e3).collect::<Vec<_>>());
}

fn args() -> HashMap<String, String> {
    std::env::args()
        .skip(1)
        .map(|a| match a.split_once('=') {
            Some((k, v)) => (k.to_owned(), v.to_owned()),
            None => (a, String::new()),
        })
        .collect()
}

pub struct Setup {
    pub mpl: usize,
    pub nb: usize,
    pub substeps: u32,
    pub rounds: u32,
    pub block: u32,
    pub min_blocks: u32,
}

impl Setup {
    pub fn defines(&self) -> Vec<(&'static str, String)> {
        vec![
            ("W", "2".into()),
            ("MPL", self.mpl.to_string()),
            ("NB", self.nb.to_string()),
            ("SUBSTEPS", self.substeps.to_string()),
            ("MAX_ROUNDS", self.rounds.to_string()),
            ("BLOCK", self.block.to_string()),
            ("MIN_BLOCKS", self.min_blocks.to_string()),
        ]
    }
}

/// Compiles the stub; prints and returns (cubin, log).
pub fn compile_stub(setup: &Setup, arch: &str) -> Result<(Vec<u8>, String)> {
    unsafe { std::env::set_var("CUDA_CACHE_DISABLE", "1") };
    let nvrtc = Nvrtc::load()?;
    let started = Instant::now();
    let (cubin, log) = nvrtc.compile(&source(&setup.defines()), "lane_stub.cu", &options(arch))?;
    eprintln!("compiled in {:.1} s", started.elapsed().as_secs_f64());
    Ok((cubin, log))
}

fn main() -> Result<()> {
    let a = args();
    let get = |k: &str, d: &str| a.get(k).cloned().unwrap_or_else(|| d.to_owned());
    let num = |k: &str, d: &str| -> usize { get(k, d).parse().unwrap_or_else(|_| panic!("{k} is a number")) };
    let setup = Setup {
        mpl: num("mpl", "16"),
        nb: num("nb", "3"),
        substeps: num("substeps", "2") as u32,
        rounds: num("rounds", "3") as u32,
        block: num("block", "128") as u32,
        min_blocks: num("min_blocks", "4") as u32,
    };
    let arch = get("arch", "sm_89");
    let (cubin, log) = compile_stub(&setup, &arch)?;
    if let Some(path) = a.get("src") {
        std::fs::write(path, source(&setup.defines()))?;
    }
    if let Some(path) = a.get("cubin") {
        std::fs::write(path, &cubin)?;
    }
    if a.contains_key("log") {
        println!("{log}");
    }
    println!("ptxas: {}", ptxas_summary(&log));
    println!("largest spill: {} B", max_spill(&log));
    if a.contains_key("report") {
        return Ok(());
    }
    let count = num("count", "262144");
    let steps = num("steps", "300") as u32;
    let nodes = num("nodes", "8");
    let muscles = num("muscles", "19");
    let repeat = num("repeat", "3");
    let seed = num("seed", "1") as u64;

    let cu = Cuda::load()?;
    let dev = cu.open()?;
    let f = cu.function(&cubin, "lane_stub")?;
    let (regs, local, shared) = cu.function_resources(f)?;
    let sms = cu.attribute(dev, 16)?;
    let mut per_sm = 0;
    cu.check(unsafe { (cu.occupancy)(&mut per_sm, f, setup.block as c_int, 0) }, "occupancy")?;
    println!(
        "driver: {regs} registers, {local} B local memory per thread, {shared} B static shared per block, \
         {per_sm} blocks ({} warps) per SM, {sms} SMs",
        per_sm as u32 * setup.block / 32
    );

    let started = Instant::now();
    let b = batch(count, nodes, muscles, setup.mpl, setup.nb, seed);
    eprintln!("{count} creatures ({nodes} nodes, {muscles} muscles) built in {:.1} s", started.elapsed().as_secs_f64());
    let heads = cu.upload(&b.heads)?;
    let lanes = cu.upload(&b.lanes)?;
    let msa = cu.upload(&b.msa)?;
    let msb = cu.upload(&b.msb)?;
    let mss = cu.upload(&b.mss)?;
    let roff = cu.alloc(count * setup.mpl * 2 * 4)?;
    let mstate = cu.alloc(count * setup.mpl * 2 * 4)?;
    let anch = cu.alloc(count * 4 * 2 * 4)?;
    let results = cu.alloc(count * 32)?;
    let counter = cu.alloc(4)?;
    #[repr(C)]
    struct Params {
        count: u32,
        steps: u32,
        screen_step: u32,
        spare: u32,
        gravity: f32,
        friction: f32,
        recovery: f32,
        screen_bar: f32,
    }
    let grid = (sms * per_sm) as u32;
    // Developer tracing (the source built with -DTRACE): each thread writes
    // its last checkpoint to mapped host memory, printed after `trace`
    // seconds without waiting for the kernel.
    let trace_secs: Option<f64> = a.get("trace").map(|v| v.parse().unwrap_or(3.0));
    let mut trace_host: *mut c_void = std::ptr::null_mut();
    let mut trace_dev: Ptr = 0;
    if trace_secs.is_some() {
        let bytes = grid as usize * setup.block as usize * 4;
        cu.check(unsafe { (cu.host_alloc)(&mut trace_host, bytes, 2) }, "cuMemHostAlloc")?;
        unsafe { std::ptr::write_bytes(trace_host as *mut u8, 0, bytes) };
        cu.check(unsafe { (cu.host_device_pointer)(&mut trace_dev, trace_host, 0) }, "device pointer")?;
    }
    let run = |n: u32| -> Result<f64> {
        let params = Params {
            count: n,
            steps,
            screen_step: u32::MAX,
            spare: 0,
            gravity: 9.8,
            friction: 1.0,
            recovery: 1.0,
            screen_bar: -1e30,
        };
        cu.check(unsafe { (cu.memset_d32)(counter, 0, 1) }, "memset")?;
        let mut ptrs = [heads, lanes, msa, msb, mss, roff, mstate, anch, results, counter];
        let mut p = params;
        let mut argv: Vec<*mut c_void> = ptrs.iter_mut().map(|x| x as *mut u64 as *mut c_void).collect();
        argv.push(&mut p as *mut Params as *mut c_void);
        let mut tp = trace_dev;
        if trace_secs.is_some() {
            argv.push(&mut tp as *mut u64 as *mut c_void);
        }
        let (mut e0, mut e1) = (std::ptr::null_mut(), std::ptr::null_mut());
        cu.check(unsafe { (cu.event_create)(&mut e0, 0) }, "event")?;
        cu.check(unsafe { (cu.event_create)(&mut e1, 0) }, "event")?;
        cu.check(unsafe { (cu.event_record)(e0, std::ptr::null_mut()) }, "record")?;
        cu.check(
            unsafe {
                (cu.launch)(
                    f,
                    grid,
                    1,
                    1,
                    setup.block,
                    1,
                    1,
                    0,
                    std::ptr::null_mut(),
                    argv.as_mut_ptr(),
                    std::ptr::null_mut(),
                )
            },
            "launch",
        )?;
        cu.check(unsafe { (cu.event_record)(e1, std::ptr::null_mut()) }, "record")?;
        if let Some(secs) = trace_secs {
            std::thread::sleep(std::time::Duration::from_secs_f64(secs));
            let t = unsafe { std::slice::from_raw_parts(trace_host as *const u32, grid as usize * setup.block as usize) };
            let mut counts: std::collections::BTreeMap<u32, usize> = Default::default();
            for &v in t {
                *counts.entry(v).or_default() += 1;
            }
            println!("checkpoint counts: {counts:?}");
            if a.contains_key("dumpw") {
                let f = unsafe { std::slice::from_raw_parts(trace_host as *const f32, 128) };
                let rad: Vec<f32> = (0..8).map(|g| b.lanes[(4 + g % 4) * 2 + g / 4]).collect();
                check_dump_w(f, &rad);
            }
            if a.contains_key("dump") {
                let f = unsafe { std::slice::from_raw_parts(trace_host as *const f32, 64) };
                check_dump(f);
            }
            for (w, chunk) in t.chunks(32).enumerate() {
                if chunk.iter().any(|&v| v != chunk[0]) || counts.get(&chunk[0]).copied().unwrap_or(0) < 64 {
                    println!("warp {w}: {chunk:?}");
                }
            }
            std::process::exit(0);
        }
        cu.check(unsafe { (cu.event_synchronize)(e1) }, "sync")?;
        let mut ms = 0.0f32;
        cu.check(unsafe { (cu.event_elapsed)(&mut ms, e0, e1) }, "elapsed")?;
        Ok(f64::from(ms) / 1000.0)
    };
    run(count.min(16_384) as u32)?;
    let mut best = f64::MAX;
    for r in 0..repeat {
        let secs = run(count as u32)?;
        let mut out = vec![[0.0f32; 8]; count];
        cu.check(unsafe { (cu.dtoh)(out.as_mut_ptr() as *mut c_void, results, count * 32) }, "dtoh")?;
        if a.contains_key("first") {
            println!("creature 0: {:?}", out[0]);
            let bad: Vec<usize> = (0..count).filter(|&i| !(out[i][0] > -1e19) || !out[i][2].is_finite()).collect();
            println!("bad: {} {:?}", bad.len(), &bad[..bad.len().min(40)]);
            for &i in bad.iter().take(3) {
                let tp: Vec<String> = (0..8).map(|g| {
                    let t = b.lanes[(i * RF + 24 + g % 4) * 2 + g / 4].to_bits();
                    format!("{}:{}p{}r{}", g, t & 31, (t >> 5) & 31, (t >> 10) & 3)
                }).collect();
                println!("  creature {i}: steps {} topo {}", out[i][3], tp.join(" "));
            }
        }
        let steps_done: f64 = out.iter().map(|o| f64::from(o[3])).sum();
        let rate = steps_done / secs;
        best = best.min(secs);
        let bad = out.iter().filter(|o| o[0] < -1e19 || !o[0].is_finite()).count();
        let fallen = out.iter().filter(|o| o[1] > 0.5).count();
        let sub = steps_done * f64::from(setup.substeps);
        let rounds: f64 = out.iter().map(|o| f64::from(o[4])).sum::<f64>() / sub;
        let contacts: f64 = out.iter().map(|o| f64::from(o[5])).sum::<f64>() / sub;
        let drift = out.iter().map(|o| o[6]).fold(0.0f32, f32::max);
        let mut drift_sorted: Vec<f32> = out.iter().map(|o| o[6]).collect();
        drift_sorted.sort_by(f32::total_cmp);
        let low: f64 = out.iter().map(|o| f64::from(o[2])).sum::<f64>() / count as f64;
        println!(
            "run {r}: {secs:.3} s, {:.1}M creature-steps/s, {:.0} creatures/s at {steps} steps; \
             rounds/substep {rounds:.2}, contacts/substep {contacts:.2}, lowest point mean {low:.3} m, \
             drift median {:.2e} max {drift:.2e} m, head below neck {fallen}, non-finite {bad}",
            rate / 1e6,
            count as f64 / secs,
            drift_sorted[count / 2],
        );
    }
    println!("best: {:.1}M creature-steps/s", count as f64 * f64::from(steps) / best / 1e6);
    for p in [heads, lanes, msa, msb, mss, roff, mstate, anch, results, counter] {
        unsafe { (cu.mem_free)(p) };
    }
    Ok(())
}
