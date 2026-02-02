use crate::linalg::matrix::{Matrix, MatrixError};
use cudarc::driver::safe::{CudaContext, CudaFunction, LaunchArgs};
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use cudarc::nvrtc::safe::Ptx;
use std::error::Error;
use std::sync::Arc;

const BLOCK_SIZE: u32 = 16;

#[derive(Debug, Clone, PartialEq)]
pub enum Activation {
    Sigmoid,
    HyperbolicTangent,
    ReLU,
    LeakyReLU, // needs work to account for the additional slope parameter
}

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

const SIGMOID_DERIVATIVE: &str = "
    extern \"C\" __global__ void cuSigmoidDerivative(float* A, float* B, int n_rows, int n_cols) {
        // Sigmoid activation derivative kernel implementation
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
            float s = 1.00 / (1.00 + exp(A[idx]));
            B[idx] = s * (1.00 - s);
        }
    }
";

const HYPERBOLICTANGENT: &str = "
    extern \"C\" __global__ void cuHyperbolicTangent(float* A, float* B, int n_rows, int n_cols) {
        // Hyperbolic tangent activation kernel implementation
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
            float a = exp(A[idx]);
            float b = exp(-A[idx]);
            B[idx] = (a - b) / (a + b);
        }
    }
";

const HYPERBOLICTANGENT_DERIVATIVE: &str = "
    extern \"C\" __global__ void cuHyperbolicTangentDerivative(float* A, float* B, int n_rows, int n_cols) {
        // Hyperbolic tangent derivative activation kernel implementation
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
            float a = exp(A[idx]);
            float b = exp(-A[idx]);
            float t = (a - b) / (a + b);
            B[idx] = 1.00 - pow(t, 2);
        }
    }
";

const RELU: &str = "
    extern \"C\" __global__ void cuReLU(float* A, float* B, int n_rows, int n_cols) {
        // Rectified linear unit activation kernel implementation
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
            if (A[idx] > 0) {
                B[idx] = A[idx];
            } else {
                B[idx] = 0;
            }
        }
    }
";

const RELU_DERIVATIVE: &str = "
    extern \"C\" __global__ void cuReLUDerivative(float* A, float* B, int n_rows, int n_cols) {
        // Rectified linear unit activation derivative kernel implementation
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
            if (A[idx] > 0) {
                B[idx] = 1.0;
            } else {
                B[idx] = 0.0;
            }
        }
    }
";

const LEAKYRELU: &str = "
    extern \"C\" __global__ void cuLeakyReLU(float a, float* A, float* B, int n_rows, int n_cols) {
        // Leaky rectified linear unit activation kernel implementation
        // Arguments:
        //  - a: slope of the function
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
            if (A[idx] > 0) {
                B[idx] = A[idx];
            } else {
                B[idx] = a * A[idx];
            }
        }
    }
";

const LEAKYRELU_DERIVATIVE: &str = "
    extern \"C\" __global__ void cuLeakyReLUDerivative(float a, float* A, float* B, int n_rows, int n_cols) {
        // Leaky rectified linear unit activation derivative kernel implementation
        // Arguments:
        //  - a: slope of the function
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
            if (A[idx] > 0) {
                B[idx] = 1.0;
            } else {
                B[idx] = a;
            }
        }
    }
";

pub fn sigmoid(a: &Matrix) -> Result<Matrix, Box<dyn Error>> {
    let ptx: Ptx = compile_ptx(SIGMOID)?;
    let f: CudaFunction = a
        .data
        .context()
        .load_module(ptx)?
        .load_function("cuSigmoid")?;
    let stream: Arc<CudaStream> = a.data.context().default_stream();
    let mut builder: LaunchArgs = stream.launch_builder(&f);
    let n_rows: u32 = a.n_rows as u32;
    let n_cols: u32 = a.n_cols as u32;
    let out: Vec<f32> = vec![0.0; (n_rows * n_cols) as usize];
    let mut out_dev: CudaSlice<f32> = stream.clone_htod(&out)?;
    builder.arg(&a.data);
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

pub fn sigmoidderivative(a: &Matrix) -> Result<Matrix, Box<dyn Error>> {
    let ptx: Ptx = compile_ptx(SIGMOID_DERIVATIVE)?;
    let f: CudaFunction = a
        .data
        .context()
        .load_module(ptx)?
        .load_function("cuSigmoidDerivative")?;
    let stream: Arc<CudaStream> = a.data.context().default_stream();
    let mut builder: LaunchArgs = stream.launch_builder(&f);
    let n_rows: u32 = a.n_rows as u32;
    let n_cols: u32 = a.n_cols as u32;
    let out: Vec<f32> = vec![0.0; (n_rows * n_cols) as usize];
    let mut out_dev: CudaSlice<f32> = stream.clone_htod(&out)?;
    builder.arg(&a.data);
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

pub fn hyperbolictangent(a: &Matrix) -> Result<Matrix, Box<dyn Error>> {
    let ptx: Ptx = compile_ptx(HYPERBOLICTANGENT)?;
    let f: CudaFunction = a
        .data
        .context()
        .load_module(ptx)?
        .load_function("cuHyperbolicTangent")?;
    let stream: Arc<CudaStream> = a.data.context().default_stream();
    let mut builder: LaunchArgs = stream.launch_builder(&f);
    let n_rows: u32 = a.n_rows as u32;
    let n_cols: u32 = a.n_cols as u32;
    let out: Vec<f32> = vec![0.0; (n_rows * n_cols) as usize];
    let mut out_dev: CudaSlice<f32> = stream.clone_htod(&out)?;
    builder.arg(&a.data);
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

pub fn hyperbolictangentderivative(a: &Matrix) -> Result<Matrix, Box<dyn Error>> {
    let ptx: Ptx = compile_ptx(HYPERBOLICTANGENT_DERIVATIVE)?;
    let f: CudaFunction = a
        .data
        .context()
        .load_module(ptx)?
        .load_function("cuHyperbolicTangentDerivative")?;
    let stream: Arc<CudaStream> = a.data.context().default_stream();
    let mut builder: LaunchArgs = stream.launch_builder(&f);
    let n_rows: u32 = a.n_rows as u32;
    let n_cols: u32 = a.n_cols as u32;
    let out: Vec<f32> = vec![0.0; (n_rows * n_cols) as usize];
    let mut out_dev: CudaSlice<f32> = stream.clone_htod(&out)?;
    builder.arg(&a.data);
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

pub fn relu(a: &Matrix) -> Result<Matrix, Box<dyn Error>> {
    let ptx: Ptx = compile_ptx(RELU)?;
    let f: CudaFunction = a.data.context().load_module(ptx)?.load_function("cuReLU")?;
    let stream: Arc<CudaStream> = a.data.context().default_stream();
    let mut builder: LaunchArgs = stream.launch_builder(&f);
    let n_rows: u32 = a.n_rows as u32;
    let n_cols: u32 = a.n_cols as u32;
    let out: Vec<f32> = vec![0.0; (n_rows * n_cols) as usize];
    let mut out_dev: CudaSlice<f32> = stream.clone_htod(&out)?;
    builder.arg(&a.data);
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

pub fn reluderivative(a: &Matrix) -> Result<Matrix, Box<dyn Error>> {
    let ptx: Ptx = compile_ptx(RELU_DERIVATIVE)?;
    let f: CudaFunction = a
        .data
        .context()
        .load_module(ptx)?
        .load_function("cuReLUDerivative")?;
    let stream: Arc<CudaStream> = a.data.context().default_stream();
    let mut builder: LaunchArgs = stream.launch_builder(&f);
    let n_rows: u32 = a.n_rows as u32;
    let n_cols: u32 = a.n_cols as u32;
    let out: Vec<f32> = vec![0.0; (n_rows * n_cols) as usize];
    let mut out_dev: CudaSlice<f32> = stream.clone_htod(&out)?;
    builder.arg(&a.data);
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

// Different function signatures as above:

pub fn leakyrelu(a: &Matrix, s: f32) -> Result<Matrix, Box<dyn Error>> {
    let ptx: Ptx = compile_ptx(LEAKYRELU)?;
    let f: CudaFunction = a
        .data
        .context()
        .load_module(ptx)?
        .load_function("cuLeakyReLU")?;
    let stream: Arc<CudaStream> = a.data.context().default_stream();
    let mut builder: LaunchArgs = stream.launch_builder(&f);
    let n_rows: u32 = a.n_rows as u32;
    let n_cols: u32 = a.n_cols as u32;
    let out: Vec<f32> = vec![0.0; (n_rows * n_cols) as usize];
    let mut out_dev: CudaSlice<f32> = stream.clone_htod(&out)?;
    builder.arg(&s);
    builder.arg(&a.data);
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

pub fn leakyreluderivative(a: &Matrix, s: f32) -> Result<Matrix, Box<dyn Error>> {
    let ptx: Ptx = compile_ptx(LEAKYRELU_DERIVATIVE)?;
    let f: CudaFunction = a
        .data
        .context()
        .load_module(ptx)?
        .load_function("cuLeakyReLUDerivative")?;
    let stream: Arc<CudaStream> = a.data.context().default_stream();
    let mut builder: LaunchArgs = stream.launch_builder(&f);
    let n_rows: u32 = a.n_rows as u32;
    let n_cols: u32 = a.n_cols as u32;
    let out: Vec<f32> = vec![0.0; (n_rows * n_cols) as usize];
    let mut out_dev: CudaSlice<f32> = stream.clone_htod(&out)?;
    builder.arg(&s);
    builder.arg(&a.data);
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

impl Activation {
    pub fn activate(&self, a: &Matrix) -> Result<Matrix, Box<dyn Error>> {
        match self {
            Activation::Sigmoid => sigmoid(a),
            Activation::HyperbolicTangent => hyperbolictangent(a),
            Activation::ReLU => relu(a),
            _ => {
                return Err(Box::new(MatrixError::DimensionMismatch(format!(
                    "Unimplemented activation function."
                ))));
            }
        }
    }
    pub fn derivative(&self, a: &Matrix) -> Result<Matrix, Box<dyn Error>> {
        match self {
            Activation::Sigmoid => sigmoidderivative(a),
            Activation::HyperbolicTangent => hyperbolictangentderivative(a),
            Activation::ReLU => reluderivative(a),
            _ => {
                return Err(Box::new(MatrixError::DimensionMismatch(format!(
                    "Unimplemented activation function."
                ))));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_activations() -> Result<(), Box<dyn Error>> {
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

        let activation_1 = Activation::Sigmoid;
        let activation_2 = Activation::HyperbolicTangent;
        let activation_3 = Activation::ReLU;
        let _activation_4 = Activation::LeakyReLU;

        let matrix_1 = activation_1.activate(&a_matrix)?;
        stream.memcpy_dtoh(&matrix_1.data, &mut a_host)?; // does not interfere with a_matrix because the data in a_host is in CPU while a_matrix is in GPU
        println!("After `sigmoid`: a_host {:?}", a_host);
        assert_eq!(
            a_host,
            vec![
                0.5,
                0.26894143,
                0.11920292,
                0.047425874,
                0.01798621,
                0.006692851,
                0.002472623,
                0.0009110512,
                0.00033535014,
                0.00012339458,
                4.539787e-5,
                1.670142e-5
            ]
        );

        let matrix_2 = activation_1.derivative(&a_matrix)?;
        stream.memcpy_dtoh(&matrix_2.data, &mut a_host)?; // does not interfere with a_matrix because the data in a_host is in CPU while a_matrix is in GPU
        println!("After `sigmoidderivative`: a_host {:?}", a_host);
        assert_eq!(
            a_host,
            vec![
                0.25,
                0.19661194,
                0.10499358,
                0.04517666,
                0.017662706,
                0.0066480567,
                0.0024665091,
                0.00091022113,
                0.00033523768,
                0.00012337936,
                4.5395806e-5,
                1.6701142e-5
            ]
        );

        let matrix_3 = activation_2.activate(&a_matrix)?;
        stream.memcpy_dtoh(&matrix_3.data, &mut a_host)?; // does not interfere with a_matrix because the data in a_host is in CPU while a_matrix is in GPU
        println!("After `hyperbolictangent`: a_host {:?}", a_host);
        assert_eq!(
            a_host,
            vec![
                0.0, 0.7615942, 0.9640275, 0.9950547, 0.9993293, 0.9999091, 0.9999877, 0.99999845,
                0.9999998, 1.0, 1.0, 1.0
            ]
        );

        let matrix_4 = activation_2.derivative(&a_matrix)?;
        stream.memcpy_dtoh(&matrix_4.data, &mut a_host)?; // does not interfere with a_matrix because the data in a_host is in CPU while a_matrix is in GPU
        println!("After `hyperbolictangentderivative`: a_host {:?}", a_host);
        assert_eq!(
            a_host,
            vec![
                1.0,
                0.41997433,
                0.070650935,
                0.009866118,
                0.0013408661,
                0.00018179417,
                2.4557114e-5,
                3.0994415e-6,
                3.5762787e-7,
                0.0,
                0.0,
                0.0
            ]
        );

        let matrix_5 = activation_3.activate(&a_matrix)?;
        stream.memcpy_dtoh(&matrix_5.data, &mut a_host)?; // does not interfere with a_matrix because the data in a_host is in CPU while a_matrix is in GPU
        println!("After `relu`: a_host {:?}", a_host);
        assert_eq!(
            a_host,
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        );

        let matrix_6 = activation_3.derivative(&a_matrix)?;
        stream.memcpy_dtoh(&matrix_6.data, &mut a_host)?; // does not interfere with a_matrix because the data in a_host is in CPU while a_matrix is in GPU
        println!("After `reluderivative`: a_host {:?}", a_host);
        assert_eq!(
            a_host,
            vec![0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        );

        // Needs work because of the additional slope parameter
        let matrix_7 = leakyrelu(&a_matrix, 0.1)?;
        stream.memcpy_dtoh(&matrix_7.data, &mut a_host)?; // does not interfere with a_matrix because the data in a_host is in CPU while a_matrix is in GPU
        println!("After `leakyrelu`: a_host {:?}", a_host);
        assert_eq!(
            a_host,
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        );

        let matrix_8 = leakyreluderivative(&a_matrix, 0.1)?;
        stream.memcpy_dtoh(&matrix_8.data, &mut a_host)?; // does not interfere with a_matrix because the data in a_host is in CPU while a_matrix is in GPU
        println!("After `leakyreluderivative`: a_host {:?}", a_host);
        assert_eq!(
            a_host,
            vec![0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        );

        Ok(())
    }
}
