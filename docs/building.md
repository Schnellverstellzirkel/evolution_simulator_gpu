# Building

The normal `release` profile keeps thin LTO and 16 codegen units. For quicker
iteration, `fast` inherits the release settings but disables LTO, uses 256
codegen units, and enables incremental compilation:

```bash
CARGO_BUILD_JOBS=8 nice -n 10 cargo build --profile fast
CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 nice -n 10 cargo run --profile fast
```

The repository's x86_64 Linux Cargo config already enables `target-cpu=native`.
If `mold` is installed, it can speed up linking:

```bash
CARGO_BUILD_JOBS=8 nice -n 10 env RUSTFLAGS="-C target-cpu=native -C link-arg=-fuse-ld=mold" cargo build --profile fast
```

Supplying `RUSTFLAGS` replaces Cargo's configured rustflags, so the command
repeats `target-cpu=native` explicitly. Omit it when `mold` is unavailable.

For tests, benchmarks, and game runs on this machine, keep evaluation off the
desktop Radeon and cap CPU use. Apply these environment settings to each such
command:

```bash
CARGO_BUILD_JOBS=8 RAYON_NUM_THREADS=8 RUST_TEST_THREADS=1 EVOLUTION_DEVICES=primary EVOLUTION_CPU_THREADS=6 nice -n 10 cargo test --release --all-targets
```

Serial test execution prevents independently created evaluation pools from
running concurrently. `--all-targets` includes the size-report diagnostic tests;
GPU tests remain ignored unless explicitly selected. Use the same resource
settings for other test or benchmark commands. Normal release
builds and runs remain available with `--release`.
