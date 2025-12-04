use crate::linalg::matrix::Matrix;
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use cudarc::nvrtc::safe::Ptx;
use cudarc::driver::safe::{CudaContext, CudaModule, CudaFunction, LaunchArgs};
use std::clone::Clone;
use std::default::Default;
use std::sync::Arc;

const BLOCK_SIZE: u32 = 16;

const SIGMOID: &str = "
    extern \"C\" __global__ void cuSigmoid(float* A, float* B, int n_rows, int n_cols) {
        // Sigmoid activation kernel implementation
        // Arguments:
        //  - A: input matrix (n_rows x n_cols)
        //  - B: output matrix (n_rows x n_cols)
        //  - n_rows: number of rows in A and B
        //  - n_cols: number of columns in A and B
        // Assumes:
        //  - row-major storage
        //  - matrices A and B are of the same size
        int i = (blockIdx.y * blockDim.y) + threadIdx.y; // Row index
        int j = (blockIdx.x * blockDim.x) + threadIdx.x; // Column index
        if ((i < n_rows) && (j < n_cols)) {
            int idx = (i * n_cols) + j; // Linear index for the A and B matrices
            B[idx] = 1.00 / (1.00 + exp(A[idx]));
        }
    }
";


pub fn sigmoid<T: Default + Clone + cudarc::driver::DeviceRepr>(
    a: &Matrix<T>,
) -> Result<Matrix<T>, Box<dyn std::error::Error>> {
    let ptx: Ptx = compile_ptx(SIGMOID)?;
    let ctx: Arc<CudaContext> = CudaContext::new(0)?;
    let stream: Arc<CudaStream> = ctx.default_stream();
    let module: Arc<CudaModule> = ctx.load_module(ptx)?;
    let f: CudaFunction = module.load_function("cuSigmoid")?;
    let mut builder: LaunchArgs = stream.launch_builder(&f);
    let n_rows: u32 = a.n_rows as u32;
    let n_cols: u32 = a.n_cols as u32;
    let out: Vec<T> = vec![T::default(); (n_rows * n_cols) as usize];
    let mut out_dev: CudaSlice<T> = stream.clone_htod(&out)?;
    builder.arg(&a.data);
    builder.arg(&mut out_dev);
    builder.arg(&n_rows);
    builder.arg(&n_cols);
    let cfg = LaunchConfig {
        block_dim: (
            BLOCK_SIZE, 
            BLOCK_SIZE, 
            1
        ),
        grid_dim: (
            (n_cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE,
            1
        ),
        shared_mem_bytes: 0,
    };
    unsafe {
        let _ = builder.launch(cfg);
    };
    Ok(
        Matrix::new(out_dev, n_rows as usize, n_cols as usize)?
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f() -> Result<(), Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let (n, p, m): (usize, usize, usize) = (4, 3, 2);
        let (a_n_rows, a_n_cols): (usize, usize) = (n, p);
        let (b_n_rows, b_n_cols): (usize, usize) = (p, m);
        let (c_n_rows, c_n_cols): (usize, usize) = (n, m);

        let mut a_host: Vec<f32> = (0..(a_n_rows * a_n_cols)).map(|x| x as f32).collect();
        let mut b_host: Vec<f32> = (0..(b_n_rows * b_n_cols)).map(|x| x as f32).collect();
        let mut c_host: Vec<f32> = (0..(c_n_rows * c_n_cols)).map(|x| x as f32).collect();

        // Copy data from CPU to GPU, i.e. from *_host into *_matrix
        let a_dev: CudaSlice<f32> = stream.clone_htod(&a_host)?;
        let b_dev: CudaSlice<f32> = stream.clone_htod(&b_host)?;
        let c_dev: CudaSlice<f32> = stream.clone_htod(&c_host)?;
        let a_matrix = Matrix::new(a_dev, a_n_rows, a_n_cols)?;
        let b_matrix = Matrix::new(b_dev, b_n_rows, b_n_cols)?;
        let c_matrix = Matrix::new(c_dev, c_n_rows, c_n_cols)?;

        println!("a_matrix {:?}", a_matrix);
        println!("b_matrix {:?}", b_matrix);
        println!("c_matrix {:?}", c_matrix);

        stream.memcpy_dtoh(&a_matrix.data, &mut a_host)?;
        stream.memcpy_dtoh(&b_matrix.data, &mut b_host)?;
        stream.memcpy_dtoh(&c_matrix.data, &mut c_host)?;
        println!("Before: a_host {:?}", a_host);
        println!("Before: b_host {:?}", b_host);
        println!("Before: c_host {:?}", c_host);

        let matrix_1 = sigmoid(&a_matrix)?;
        stream.memcpy_dtoh(&matrix_1.data, &mut a_host)?; // does not interfere with a_matrix because the data in a_host is in CPU while a_matrix is in GPU
        println!("After `sigmoid`: a_host {:?}", a_host);
        assert_eq!(a_host, vec![0.5, 0.26894143, 0.11920292, 0.047425874, 0.01798621, 0.006692851, 0.002472623, 0.0009110512, 0.00033535014, 0.00012339458, 4.539787e-5, 1.670142e-5]);

        Ok(())
    }
}
