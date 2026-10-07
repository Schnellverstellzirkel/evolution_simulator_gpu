//! exploraMove: 2D creatures of bones, joints and muscles evolve to travel as far
//! as they can in 20 s trials, scored on the GPU and searched with MAP-Elites over
//! island archives. The search lives in `evolution`, `qd` and `storage`.
//! `ring`, `scheduler`, `engine`, `kernel` and `cuda_engine` carry creatures to
//! the GPU and back. `worker` runs the experiment on its own thread, and `ui`
//! draws the worker's snapshots.

pub mod assets;
pub mod block_alloc;
pub mod bounded;
pub mod config;
pub mod creature_kernel;
pub mod cuda_engine;
pub mod dev_pause;
pub mod engine;
pub mod environment;
pub mod evolution;
pub mod gpu;
pub mod kernel;
pub mod loading;
pub mod physics;
pub mod physics2;
pub mod qd;
pub mod replay_forces;
pub mod ring;
pub mod rungs;
pub mod scheduler;
pub mod schematic;
pub mod storage;
pub mod theme;
pub mod threads;
pub mod ui;
pub mod worker;
pub mod world_fx;

/// Large blocks keep their pages for the next block of their size (`block_alloc`).
#[global_allocator]
static ALLOCATOR: block_alloc::BlockAlloc = block_alloc::BlockAlloc;
