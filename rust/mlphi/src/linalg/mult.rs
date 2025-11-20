use rand::prelude::*;
use std::sync::Arc;
use cudarc::driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg, CudaSlice, CudaStream};
use cudarc::nvrtc::compile_ptx;

pub fn F() -> Result<(), DriverError> {
    let start = std::time::Instant::now();

    let ptx = compile_ptx("
    extern \"C\" __global__ void matmul(float* A, float* B, float* C, int N) {
        int ROW = (blockIdx.y * blockDim.y) + threadIdx.y;
        int COL = (blockIdx.x * blockDim.x) + threadIdx.x;

        float tmpSum = 0;

        if (ROW < N && COL < N) {
            // each thread computes one element of the block sub-matrix
            for (int i = 0; i < N; i++) {
                tmpSum += A[ROW * N + i] * B[i * N + COL];
            }
        }
        // printf(\"pos, (%d, %d) - N %d - value %d\\n\", ROW, COL, N, tmpSum);
        C[ROW * N + COL] = tmpSum;
    }
    ").unwrap();
    println!("Compilation succeeded in {:?}", start.elapsed());

    let ctx = CudaContext::new(0)?;
    let stream: Arc<CudaStream> = ctx.default_stream();
    println!("Built in {:?}", start.elapsed());

    let module = ctx.load_module(ptx)?;
    let f = module.load_function("matmul")?;
    println!("Loaded in {:?}", start.elapsed());

    let n: usize = 100;
    let m: u32 = 10;
    let p: u32 = 2;
    let mut a_host: Vec<f32> = vec![0f32; n];
    rand::fill(&mut a_host[..]);
    let mut b_host: Vec<f32> = vec![0f32; n];
    rand::fill(&mut b_host[..]);
    let mut c_host = vec![0.0f32; n];

    println!("a_host: {:?}", &a_host);
    println!("b_host: {:?}", &b_host);
    println!("c_host: {:?}", &c_host);

    let a_dev: CudaSlice<f32> = stream.clone_htod(&a_host)?;
    let b_dev: CudaSlice<f32> = stream.clone_htod(&b_host)?;
    let mut c_dev: CudaSlice<f32> = stream.clone_htod(&c_host)?;

    println!("Copied in {:?}", start.elapsed());

    let mut builder = stream.launch_builder(&f);
    builder.arg(&a_dev);
    builder.arg(&b_dev);
    builder.arg(&mut c_dev);
    builder.arg(&m);
    let cfg = LaunchConfig {
        block_dim: (m/p, m/p, 1),
        grid_dim: (p, p, 1),
        shared_mem_bytes: 0,
    };
    unsafe { builder.launch(cfg) }?;

    stream.memcpy_dtoh(&c_dev, &mut c_host)?;
    println!("Found {:?} in {:?}", c_host, start.elapsed());
    Ok(())
}