use crate::linalg::matrix::{Matrix, MatrixError};
use cudarc::driver::safe::{CudaContext, CudaFunction, LaunchArgs};
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use cudarc::nvrtc::safe::Ptx;
use std::error::Error;
use std::sync::Arc;

const BLOCK_SIZE: u32 = 16;

#[derive(Debug, Clone, PartialEq)]
pub enum Cost {
    MSE,
    MAE,
    HL, // needs work to account for the additional threshold parameter
}

const MSE: &str = "
    extern \"C\" __global__ void cuMSE(float* A, float* B, float* C, int n_rows, int n_cols) {
        // Mean squared error kernel implementation
        // Arguments:
        //  - A: input predicted matrix (n_rows x n_cols)
        //  - B: input target matrix (n_rows x n_cols)
        //  - C: output matrix (n_rows x n_cols)
        //  - n_rows: number of rows in A, B and C
        //  - n_cols: number of columns in A, B and C
        // Assumes:
        //  - row-major storage
        //  - matrices A, B and C are of the same size
        int i = (blockIdx.y * blockDim.y) + threadIdx.y; // Row index
        int j = (blockIdx.x * blockDim.x) + threadIdx.x; // Column index
        if ((i < n_rows) && (j < n_cols)) {
            int idx = (i * n_cols) + j; // Linear index for the A, B and C matrices
            C[idx] = powf(A[idx] - B[idx], 2.0) / 2.0;
        }
    }
";

const MSE_DERIVATIVE: &str = "
    extern \"C\" __global__ void cuMSEDerivative(float* A, float* B, float* C, int n_rows, int n_cols) {
        // Mean squared error derivative kernel implementation
        // Arguments:
        //  - A: input predicted matrix (n_rows x n_cols)
        //  - B: input target matrix (n_rows x n_cols)
        //  - C: output matrix (n_rows x n_cols)
        //  - n_rows: number of rows in A, B and C
        //  - n_cols: number of columns in A, B and C
        // Assumes:
        //  - row-major storage
        //  - matrices A, B and C are of the same size
        int i = (blockIdx.y * blockDim.y) + threadIdx.y; // Row index
        int j = (blockIdx.x * blockDim.x) + threadIdx.x; // Column index
        if ((i < n_rows) && (j < n_cols)) {
            int idx = (i * n_cols) + j; // Linear index for the A, B and C matrices
            C[idx] = A[idx] - B[idx];
        }
    }
";

const MAE: &str = "
    extern \"C\" __global__ void cuMAE(float* A, float* B, float* C, int n_rows, int n_cols) {
        // Mean absolute error kernel implementation
        // Arguments:
        //  - A: input predicted matrix (n_rows x n_cols)
        //  - B: input target matrix (n_rows x n_cols)
        //  - C: output matrix (n_rows x n_cols)
        //  - n_rows: number of rows in A, B and C
        //  - n_cols: number of columns in A, B and C
        // Assumes:
        //  - row-major storage
        //  - matrices A, B and C are of the same size
        int i = (blockIdx.y * blockDim.y) + threadIdx.y; // Row index
        int j = (blockIdx.x * blockDim.x) + threadIdx.x; // Column index
        if ((i < n_rows) && (j < n_cols)) {
            int idx = (i * n_cols) + j; // Linear index for the A, B and C matrices
            C[idx] = abs(A[idx] - B[idx]) / 2.0;
        }
    }
";

const MAE_DERIVATIVE: &str = "
    extern \"C\" __global__ void cuMAEDerivative(float* A, float* B, float* C, int n_rows, int n_cols) {
        // Mean absolute error derivative kernel implementation
        // Arguments:
        //  - A: input predicted matrix (n_rows x n_cols)
        //  - B: input target matrix (n_rows x n_cols)
        //  - C: output matrix (n_rows x n_cols)
        //  - n_rows: number of rows in A, B and C
        //  - n_cols: number of columns in A, B and C
        // Assumes:
        //  - row-major storage
        //  - matrices A, B and C are of the same size
        int i = (blockIdx.y * blockDim.y) + threadIdx.y; // Row index
        int j = (blockIdx.x * blockDim.x) + threadIdx.x; // Column index
        if ((i < n_rows) && (j < n_cols)) {
            int idx = (i * n_cols) + j; // Linear index for the A, B and C matrices
            float x = A[idx] - B[idx];
            if (x < 0.0) {
                C[idx] = -1.00;
            } else {
                C[idx] = 1.00;
            }
        }
    }
";

const HL: &str = "
    extern \"C\" __global__ void cuHL(float* A, float* B, float* C, float d, int n_rows, int n_cols) {
        // Huber loss kernel implementation (combines the best properties of MSE and MAE, using MSE for small errors and MAE for large errors)
        // Arguments:
        //  - A: input predicted matrix (n_rows x n_cols)
        //  - B: input target matrix (n_rows x n_cols)
        //  - C: output matrix (n_rows x n_cols)
        //  - d: threshold parameter that determines the transition between MSE and MAE
        //  - n_rows: number of rows in A, B and C
        //  - n_cols: number of columns in A, B and C
        // Assumes:
        //  - row-major storage
        //  - matrices A, B and C are of the same size
        int i = (blockIdx.y * blockDim.y) + threadIdx.y; // Row index
        int j = (blockIdx.x * blockDim.x) + threadIdx.x; // Column index
        if ((i < n_rows) && (j < n_cols)) {
            int idx = (i * n_cols) + j; // Linear index for the A, B and C matrices
            float x = A[idx] - B[idx];
            if (abs(x) <= d) {
                C[idx] = 0.5 * powf(x, 2.0);
            } else {
                C[idx] = d * (abs(x) - (0.5*d));
            }
        }
    }
";

const HL_DERIVATIVE: &str = "
    extern \"C\" __global__ void cuHLDerivative(float* A, float* B, float* C, float d, int n_rows, int n_cols) {
        // Huber loss derivative kernel implementation
        // Arguments:
        //  - A: input predicted matrix (n_rows x n_cols)
        //  - B: input target matrix (n_rows x n_cols)
        //  - C: output matrix (n_rows x n_cols)
        //  - d: threshold parameter that determines the transition between MSE and MAE
        //  - n_rows: number of rows in A, B and C
        //  - n_cols: number of columns in A, B and C
        // Assumes:
        //  - row-major storage
        //  - matrices A, B and C are of the same size
        int i = (blockIdx.y * blockDim.y) + threadIdx.y; // Row index
        int j = (blockIdx.x * blockDim.x) + threadIdx.x; // Column index
        if ((i < n_rows) && (j < n_cols)) {
            int idx = (i * n_cols) + j; // Linear index for the A, B and C matrices
            float x = A[idx] - B[idx];
            if (abs(x) <= d) {
                C[idx] = -x;
            } else {
                if (x < 0.0) {
                    C[idx] = -d * -1.0;
                } else {
                    C[idx] = -d * 1.0;
                }
            }
        }
    }
";

pub fn mse(a: &Matrix, b: &Matrix) -> Result<Matrix, Box<dyn Error>> {
    let ptx: Ptx = compile_ptx(MSE)?;
    let f: CudaFunction = a.data.context().load_module(ptx)?.load_function("cuMSE")?;
    let stream: Arc<CudaStream> = a.data.context().default_stream();
    let mut builder: LaunchArgs = stream.launch_builder(&f);
    let n_rows: u32 = a.n_rows as u32;
    let n_cols: u32 = a.n_cols as u32;
    let out: Vec<f32> = vec![0.0; (n_rows * n_cols) as usize];
    let mut out_dev: CudaSlice<f32> = stream.clone_htod(&out)?;
    builder.arg(&a.data);
    builder.arg(&b.data);
    builder.arg(&mut out_dev);
    builder.arg(&n_rows);
    builder.arg(&n_cols);
    let cfg = LaunchConfig {
        block_dim: (BLOCK_SIZE, BLOCK_SIZE, 1),
        grid_dim: (
            (n_cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE,
            1,
        ),
        shared_mem_bytes: 0,
    };
    unsafe {
        let _ = builder.launch(cfg);
    };
    Ok(Matrix::new(out_dev, n_rows as usize, n_cols as usize)?)
}

pub fn msederivative(a: &Matrix, b: &Matrix) -> Result<Matrix, Box<dyn Error>> {
    let ptx: Ptx = compile_ptx(MSE_DERIVATIVE)?;
    let f: CudaFunction = a
        .data
        .context()
        .load_module(ptx)?
        .load_function("cuMSEDerivative")?;
    let stream: Arc<CudaStream> = a.data.context().default_stream();
    let mut builder: LaunchArgs = stream.launch_builder(&f);
    let n_rows: u32 = a.n_rows as u32;
    let n_cols: u32 = a.n_cols as u32;
    let out: Vec<f32> = vec![0.0; (n_rows * n_cols) as usize];
    let mut out_dev: CudaSlice<f32> = stream.clone_htod(&out)?;
    builder.arg(&a.data);
    builder.arg(&b.data);
    builder.arg(&mut out_dev);
    builder.arg(&n_rows);
    builder.arg(&n_cols);
    let cfg = LaunchConfig {
        block_dim: (BLOCK_SIZE, BLOCK_SIZE, 1),
        grid_dim: (
            (n_cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE,
            1,
        ),
        shared_mem_bytes: 0,
    };
    unsafe {
        let _ = builder.launch(cfg);
    };
    Ok(Matrix::new(out_dev, n_rows as usize, n_cols as usize)?)
}

pub fn mae(a: &Matrix, b: &Matrix) -> Result<Matrix, Box<dyn Error>> {
    let ptx: Ptx = compile_ptx(MAE)?;
    let f: CudaFunction = a.data.context().load_module(ptx)?.load_function("cuMAE")?;
    let stream: Arc<CudaStream> = a.data.context().default_stream();
    let mut builder: LaunchArgs = stream.launch_builder(&f);
    let n_rows: u32 = a.n_rows as u32;
    let n_cols: u32 = a.n_cols as u32;
    let out: Vec<f32> = vec![0.0; (n_rows * n_cols) as usize];
    let mut out_dev: CudaSlice<f32> = stream.clone_htod(&out)?;
    builder.arg(&a.data);
    builder.arg(&b.data);
    builder.arg(&mut out_dev);
    builder.arg(&n_rows);
    builder.arg(&n_cols);
    let cfg = LaunchConfig {
        block_dim: (BLOCK_SIZE, BLOCK_SIZE, 1),
        grid_dim: (
            (n_cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE,
            1,
        ),
        shared_mem_bytes: 0,
    };
    unsafe {
        let _ = builder.launch(cfg);
    };
    Ok(Matrix::new(out_dev, n_rows as usize, n_cols as usize)?)
}

pub fn maederivative(a: &Matrix, b: &Matrix) -> Result<Matrix, Box<dyn Error>> {
    let ptx: Ptx = compile_ptx(MAE_DERIVATIVE)?;
    let f: CudaFunction = a
        .data
        .context()
        .load_module(ptx)?
        .load_function("cuMAEDerivative")?;
    let stream: Arc<CudaStream> = a.data.context().default_stream();
    let mut builder: LaunchArgs = stream.launch_builder(&f);
    let n_rows: u32 = a.n_rows as u32;
    let n_cols: u32 = a.n_cols as u32;
    let out: Vec<f32> = vec![0.0; (n_rows * n_cols) as usize];
    let mut out_dev: CudaSlice<f32> = stream.clone_htod(&out)?;
    builder.arg(&a.data);
    builder.arg(&b.data);
    builder.arg(&mut out_dev);
    builder.arg(&n_rows);
    builder.arg(&n_cols);
    let cfg = LaunchConfig {
        block_dim: (BLOCK_SIZE, BLOCK_SIZE, 1),
        grid_dim: (
            (n_cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE,
            1,
        ),
        shared_mem_bytes: 0,
    };
    unsafe {
        let _ = builder.launch(cfg);
    };
    Ok(Matrix::new(out_dev, n_rows as usize, n_cols as usize)?)
}

pub fn hl(a: &Matrix, b: &Matrix, d: f32) -> Result<Matrix, Box<dyn Error>> {
    let ptx: Ptx = compile_ptx(HL)?;
    let f: CudaFunction = a.data.context().load_module(ptx)?.load_function("cuHL")?;
    let stream: Arc<CudaStream> = a.data.context().default_stream();
    let mut builder: LaunchArgs = stream.launch_builder(&f);
    let n_rows: u32 = a.n_rows as u32;
    let n_cols: u32 = a.n_cols as u32;
    let out: Vec<f32> = vec![0.0; (n_rows * n_cols) as usize];
    let mut out_dev: CudaSlice<f32> = stream.clone_htod(&out)?;
    builder.arg(&a.data);
    builder.arg(&b.data);
    builder.arg(&mut out_dev);
    builder.arg(&d);
    builder.arg(&n_rows);
    builder.arg(&n_cols);
    let cfg = LaunchConfig {
        block_dim: (BLOCK_SIZE, BLOCK_SIZE, 1),
        grid_dim: (
            (n_cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE,
            1,
        ),
        shared_mem_bytes: 0,
    };
    unsafe {
        let _ = builder.launch(cfg);
    };
    Ok(Matrix::new(out_dev, n_rows as usize, n_cols as usize)?)
}

pub fn hlderivative(a: &Matrix, b: &Matrix, d: f32) -> Result<Matrix, Box<dyn Error>> {
    let ptx: Ptx = compile_ptx(HL_DERIVATIVE)?;
    let f: CudaFunction = a
        .data
        .context()
        .load_module(ptx)?
        .load_function("cuHLDerivative")?;
    let stream: Arc<CudaStream> = a.data.context().default_stream();
    let mut builder: LaunchArgs = stream.launch_builder(&f);
    let n_rows: u32 = a.n_rows as u32;
    let n_cols: u32 = a.n_cols as u32;
    let out: Vec<f32> = vec![0.0; (n_rows * n_cols) as usize];
    let mut out_dev: CudaSlice<f32> = stream.clone_htod(&out)?;
    builder.arg(&a.data);
    builder.arg(&b.data);
    builder.arg(&mut out_dev);
    builder.arg(&d);
    builder.arg(&n_rows);
    builder.arg(&n_cols);
    let cfg = LaunchConfig {
        block_dim: (BLOCK_SIZE, BLOCK_SIZE, 1),
        grid_dim: (
            (n_cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE,
            1,
        ),
        shared_mem_bytes: 0,
    };
    unsafe {
        let _ = builder.launch(cfg);
    };
    Ok(Matrix::new(out_dev, n_rows as usize, n_cols as usize)?)
}

impl Cost {
    pub fn cost(&self, a: &Matrix, b: &Matrix) -> Result<Matrix, Box<dyn Error>> {
        match self {
            Cost::MSE => mse(a, b),
            Cost::MAE => mae(a, b),
            _ => {
                return Err(Box::new(MatrixError::DimensionMismatch(format!(
                    "Unimplemented cost function."
                ))));
            }
        }
    }
    pub fn derivative(&self, a: &Matrix, b: &Matrix) -> Result<Matrix, Box<dyn Error>> {
        match self {
            Cost::MSE => msederivative(a, b),
            Cost::MAE => maederivative(a, b),
            _ => {
                return Err(Box::new(MatrixError::DimensionMismatch(format!(
                    "Unimplemented cost function."
                ))));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_costs() -> Result<(), Box<dyn Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let (n, p): (usize, usize) = (4, 3);
        let (a_n_rows, a_n_cols): (usize, usize) = (n, p);
        let (b_n_rows, b_n_cols): (usize, usize) = (n, p);

        let mut a_host: Vec<f32> = (0..(a_n_rows * a_n_cols)).map(|x| x as f32).collect();
        let mut b_host: Vec<f32> = ((b_n_rows * b_n_cols)..(2 * b_n_rows * b_n_cols))
            .map(|x| x as f32)
            .collect();

        // Copy data from CPU to GPU, i.e. from *_host into *_matrix
        let a_dev: CudaSlice<f32> = stream.clone_htod(&a_host)?;
        let b_dev: CudaSlice<f32> = stream.clone_htod(&b_host)?;
        let a_matrix = Matrix::new(a_dev, a_n_rows, a_n_cols)?;
        let b_matrix = Matrix::new(b_dev, b_n_rows, b_n_cols)?;

        println!("a_matrix {:?}", a_matrix);
        println!("b_matrix {:?}", b_matrix);

        stream.memcpy_dtoh(&a_matrix.data, &mut a_host)?;
        stream.memcpy_dtoh(&b_matrix.data, &mut b_host)?;
        println!("Before: a_host {:?}", a_host);
        println!("Before: b_host {:?}", b_host);

        let cost_1 = Cost::MSE;
        let cost_2 = Cost::MAE;
        let _cost_3 = Cost::HL;

        let matrix_1 = cost_1.cost(&a_matrix, &b_matrix)?;
        stream.memcpy_dtoh(&matrix_1.data, &mut a_host)?; // does not interfere with a_matrix because the data in a_host is in CPU while a_matrix is in GPU
        println!("After `mse`: a_host {:?}", a_host);
        assert_eq!(
            a_host,
            vec![
                72.0, 72.0, 72.0, 72.0, 72.0, 72.0, 72.0, 72.0, 72.0, 72.0, 72.0, 72.0
            ]
        );

        let matrix_2 = cost_1.derivative(&a_matrix, &b_matrix)?;
        stream.memcpy_dtoh(&matrix_2.data, &mut a_host)?; // does not interfere with a_matrix because the data in a_host is in CPU while a_matrix is in GPU
        println!("After `msederivative`: a_host {:?}", a_host);
        assert_eq!(
            a_host,
            vec![
                -12.0, -12.0, -12.0, -12.0, -12.0, -12.0, -12.0, -12.0, -12.0, -12.0, -12.0, -12.0
            ]
        );

        let matrix_3 = cost_2.cost(&a_matrix, &b_matrix)?;
        stream.memcpy_dtoh(&matrix_3.data, &mut a_host)?; // does not interfere with a_matrix because the data in a_host is in CPU while a_matrix is in GPU
        println!("After `mae`: a_host {:?}", a_host);
        assert_eq!(
            a_host,
            vec![6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 6.0]
        );

        let matrix_4 = cost_2.derivative(&a_matrix, &b_matrix)?;
        stream.memcpy_dtoh(&matrix_4.data, &mut a_host)?; // does not interfere with a_matrix because the data in a_host is in CPU while a_matrix is in GPU
        println!("After `maederivative`: a_host {:?}", a_host);
        assert_eq!(
            a_host,
            vec![
                -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0
            ]
        );

        // Needs work because of the additional threshold parameter
        let matrix_5 = hl(&a_matrix, &b_matrix, 1.0)?;
        stream.memcpy_dtoh(&matrix_5.data, &mut a_host)?; // does not interfere with a_matrix because the data in a_host is in CPU while a_matrix is in GPU
        println!("After `hl`: a_host {:?}", a_host);
        assert_eq!(
            a_host,
            vec![
                11.5, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5, 11.5
            ]
        );

        let matrix_6 = hlderivative(&a_matrix, &b_matrix, 1.0)?;
        stream.memcpy_dtoh(&matrix_6.data, &mut a_host)?; // does not interfere with a_matrix because the data in a_host is in CPU while a_matrix is in GPU
        println!("After `hlderivative`: a_host {:?}", a_host);
        assert_eq!(
            a_host,
            vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        );

        Ok(())
    }
}
