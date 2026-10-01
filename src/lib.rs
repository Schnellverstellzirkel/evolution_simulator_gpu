pub mod assets;
pub mod bounded;
pub mod config;
pub mod creature_kernel;
pub mod cuda_engine;
pub mod dev_pause;
pub mod engine;
pub mod environment;
pub mod evolution;
pub mod gpu;
pub mod block_alloc;
pub mod physics;
pub mod physics2;
pub mod qd;
pub mod replay_forces;
pub mod ring;
pub mod scheduler;
pub mod schematic;
pub mod storage;
pub mod theme;
pub mod threads;
pub mod ui;
pub mod warp_kernel;
pub mod worker;
pub mod world_fx;

#[global_allocator]
static ALLOCATOR: block_alloc::BlockAlloc = block_alloc::BlockAlloc;
