use cudarc::driver::{CudaContext, DeviceRepr, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::CompileError;
use cudarc::nvrtc::compile_ptx;
use std::default::Default;
use std::clone::Clone;
use std::sync::Arc;
use crate::linalg::matrix::*;

pub fn matmul<T: Default + Clone + cudarc::driver::DeviceRepr>(
    a: &Matrix<T>,
    b: &Matrix<T>,
) -> Result<Matrix<T>, MatrixError> {
    if a.n_cols != b.n_rows {
        return Err(MatrixError::DimensionMismatch(format!(
            "Dimension mismatch: a.n_cols ({}) != b.n_rows ({})",
            a.n_cols, b.n_rows
        )));
    }
    let ctx = match CudaContext::new(0) {
        Err(e) => return Err(MatrixError::OutOfMemory(format!("Failed to create CUDA context: {}", e))),
        Ok(c) => c,
    };
    let stream: Arc<CudaStream> = ctx.default_stream();

    let n_rows: u32 = a.n_rows as u32;
    let n_cols: u32 = b.n_cols as u32;
    let n_out: u32 = n_rows * n_cols;
    let mut out: Vec<T> = vec![T::default(); n_out as usize];
    let mut out_dev: CudaSlice<T> = match stream.clone_htod(&out) {
        Err(e) => return Err(MatrixError::OutOfMemory(format!("Failed to setup the output matrix from host to device: {}", e))),
        Ok(x) => x,
    };

    // // Multiplication kernel definition and launcher...

    let ptx = match 
        compile_ptx(
            "
            extern \"C\" __global__ void cuMatMul(float* A, float* B, float*C, int n_rows, int n_cols) {
                // Matrix multiplication kernel implementation
                // This is a placeholder; actual implementation needed
            }
            "
        ) {
        Err(e) => return Err(MatrixError::CompileError(format!("Failed to compile PTX: {}", e))),
        Ok(p) => p,
    };
    let module = match ctx.load_module(ptx) {
        Err(e) => return Err(MatrixError::OutOfMemory(format!("Failed to load CUDA module: {}", e))),
        Ok(m) => m,
    };
    let f = match module.load_function("cuMatMul") {
        Err(e) => return Err(MatrixError::OutOfMemory(format!("Failed to load CUDA function: {}", e))),
        Ok(f) => f,
    };
    let mut builder = stream.launch_builder(&f);
    builder.arg(&a.data);
    builder.arg(&b.data);
    builder.arg(&mut out_dev);
    builder.arg(&n_out);
    let block_dim: (u32, u32, u32) = (16, 16, 1); // Each block computes a 16x16 sub-matrix
    let grid_dim: (u32, u32, u32) = (
        (block_dim.0 + 1) / n_cols, // Number of blocks in the x-dimension
        (block_dim.1 + 1) / n_rows, // Number of blocks in the y-dimension
        1,
    );
    let cfg = LaunchConfig {
        block_dim: block_dim,
        grid_dim: grid_dim,
        shared_mem_bytes: 0,
    };
    unsafe { builder.launch(cfg) };




    // match stream.memcpy_dtoh(&out_dev, &mut out) {
    //     Err(e) => return Err(MatrixError::OutOfMemory(format!("Failed to copy data from device to host: {}", e))),
    //     Ok(_) => return Matrix::new(out_dev, n_rows, n_cols),
    // }
    Matrix::new(out_dev, n_rows as usize, n_cols as usize)
}

// use cudarc::driver::{
//     CudaContext, CudaSlice, CudaStream, DriverError, LaunchConfig, PushKernelArg,
// };
// use cudarc::nvrtc::compile_ptx;
// use std::sync::Arc;

// pub fn F() -> Result<(), DriverError> {
//     let start = std::time::Instant::now();

//     let ptx = compile_ptx(
//         "
//     extern \"C\" __global__ void matmul(float* A, float* B, float* C, int N) {
//         int ROW = (blockIdx.y * blockDim.y) + threadIdx.y;
//         int COL = (blockIdx.x * blockDim.x) + threadIdx.x;

//         float tmpSum = 0;

//         if (ROW < N && COL < N) {
//             // each thread computes one element of the block sub-matrix
//             for (int i = 0; i < N; i++) {
//                 tmpSum += A[ROW * N + i] * B[i * N + COL];
//             }
//         }
//         // printf(\"pos, (%d, %d) - N %d - value %d\\n\", ROW, COL, N, tmpSum);
//         C[ROW * N + COL] = tmpSum;
//     }
//     ",
//     )
//     .unwrap();
//     println!("Compilation succeeded in {:?}", start.elapsed());

//     let ctx = CudaContext::new(0)?;
//     let stream: Arc<CudaStream> = ctx.default_stream();
//     println!("Built in {:?}", start.elapsed());

//     let module = ctx.load_module(ptx)?;
//     let f = module.load_function("matmul")?;
//     println!("Loaded in {:?}", start.elapsed());

//     let n: usize = 100;
//     let m: u32 = 10;
//     let p: u32 = 2;
//     let mut a_host: Vec<f32> = vec![0f32; n];
//     rand::fill(&mut a_host[..]);
//     let mut b_host: Vec<f32> = vec![0f32; n];
//     rand::fill(&mut b_host[..]);
//     let mut c_host = vec![0.0f32; n];

//     println!("a_host: {:?}", &a_host);
//     println!("b_host: {:?}", &b_host);
//     println!("c_host: {:?}", &c_host);

//     let a_dev: CudaSlice<f32> = stream.clone_htod(&a_host)?;
//     let b_dev: CudaSlice<f32> = stream.clone_htod(&b_host)?;
//     let mut c_dev: CudaSlice<f32> = stream.clone_htod(&c_host)?;

//     println!("Copied in {:?}", start.elapsed());

//     let mut builder = stream.launch_builder(&f);
//     builder.arg(&a_dev);
//     builder.arg(&b_dev);
//     builder.arg(&mut c_dev);
//     builder.arg(&m);
//     let cfg = LaunchConfig {
//         block_dim: (m / p, m / p, 1),
//         grid_dim: (p, p, 1),
//         shared_mem_bytes: 0,
//     };
//     unsafe { builder.launch(cfg) }?;

//     stream.memcpy_dtoh(&c_dev, &mut c_host)?;
//     println!("Found {:?} in {:?}", c_host, start.elapsed());
//     Ok(())
// }

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use rand::prelude::*;

//     #[test]
//     fn test_f() -> Result<(), DriverError> {
//         F()?;
//         Ok(())
//     }
// }