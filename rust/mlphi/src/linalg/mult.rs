use crate::linalg::matrix::{Matrix, MatrixError};
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use cudarc::nvrtc::safe::Ptx;
use cudarc::driver::safe::{CudaContext, CudaModule, CudaFunction, LaunchArgs};
use std::clone::Clone;
use std::default::Default;
use std::sync::Arc;

const BLOCK_SIZE: u32 = 16;

const SCALARMATMUL: &str = "
    extern \"C\" __global__ void cuScalarMatMul(float scalar, float* A, float* B, int n_rows, int n_cols) {
        // Scalar-matrix multiplication kernel implementation
        // Arguments:
        //  - scalar: scalar multiplier
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
            B[idx] = scalar * A[idx];
        }
    }
";

const ELEMENTWISEMATMUL: &str = "
    extern \"C\" __global__ void cuScalarMatMul(float* A, float* B, float* C, int n_rows, int n_cols) {
        // Scalar-matrix multiplication kernel implementation
        // Arguments:
        //  - scalar: scalar multiplier
        //  - A: input matrix 1 (n_rows x n_cols)
        //  - B: input matrix 2 (n_rows x n_cols)
        //  - C: output matrix (n_rows x n_cols)
        //  - n_rows: number of rows in all 3 matrices
        //  - n_cols: number of columns in all 3 matrices
        // Assumes:
        //  - row-major storage
        //  - matrices A, B and C are of the same size
        int i = (blockIdx.y * blockDim.y) + threadIdx.y; // Row index
        int j = (blockIdx.x * blockDim.x) + threadIdx.x; // Column index
        if ((i < n_rows) && (j < n_cols)) {
            int idx = (i * n_cols) + j; // Linear index for the A and B matrices
            C[idx] = A[idx] * B[idx];
        }
    }
";

const MATMUL: &str = "
    extern \"C\" __global__ void cuMatMul(float* A, float* B, float*C, int n_rows, int n_cols, int p) {
        // Matrix multiplication kernel implementation
        // Arguments:
        //  - A: left matrix (n_rows x p)
        //  - B: right matrix (p x n_cols)
        //  - C: output matrix (n_rows x n_cols)
        //  - n_rows: number of rows in A and C
        //  - n_cols: number of columns in B and C
        //  - p: number of columns in A and rows in B
        // Assumes:
        //  - row-major storage
        //  - matrices A and B are of compatible
        //  - matrix C is the correct size to hold the result
        int i = (blockIdx.y * blockDim.y) + threadIdx.y; // Row index of C to compute
        int j = (blockIdx.x * blockDim.x) + threadIdx.x; // Column index of C to compute
        if ((i < n_rows) && (j < n_cols)) {
            float s = 0.0f;
            for (int k=0; k<p; k++) {
                s += A[(i*p)+k] * B[(k*n_cols)+j];
            }
            C[(i*n_cols)+j] = s;
        }
    }
";

pub struct MatMulPrepResult<T: std::clone::Clone + std::default::Default + cudarc::driver::DeviceRepr> {
    pub ctx: Arc<CudaContext>,
    pub stream: Arc<CudaStream>,
    pub ptx: Ptx,
    pub module: Arc<CudaModule>,
    pub function: CudaFunction,
    pub launch_args: LaunchArgs<'static>,
    pub launch_config: LaunchConfig,
    pub out_host: Vec<T>,
    pub out_dev: CudaSlice<T>,
}

pub fn prep_cuda_kernel<T: std::clone::Clone + std::default::Default + cudarc::driver::DeviceRepr>(
    n_rows: u32,
    n_cols: u32,
    kernel_string: &str,
    function_name: &str,
) -> Result<MatMulPrepResult<T>, MatrixError>
{
    let ctx: Arc<CudaContext> = match CudaContext::new(0) {
        Err(e) => {
            return Err(MatrixError::OutOfMemory(format!(
                "Failed to create CUDA context: {}",
                e
            )));
        }
        Ok(c) => c,
    };
    let stream: Arc<CudaStream> = ctx.default_stream();
    let ptx: Ptx = match compile_ptx(kernel_string) {
        Err(e) => {
            return Err(MatrixError::CompileError(format!(
                "Failed to compile PTX: {}",
                e
            )));
        }
        Ok(p) => p,
    };
    let module: Arc<CudaModule> = match ctx.load_module(ptx.clone()) {
        Err(e) => {
            return Err(MatrixError::OutOfMemory(format!(
                "Failed to load CUDA module: {}",
                e
            )));
        }
        Ok(m) => m,
    };
    let f: CudaFunction = match module.load_function(function_name) {
        Err(e) => {
            return Err(MatrixError::OutOfMemory(format!(
                "Failed to load CUDA function: {}",
                e
            )));
        }
        Ok(f) => f,
    };
    let mut builder = stream.launch_builder(&f.clone());
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
    let out: Vec<T> = vec![T::default(); (n_rows * n_cols) as usize];
    let mut out_dev: CudaSlice<T> = match stream.clone_htod(&out) {
        Err(e) => {
            return Err(MatrixError::OutOfMemory(format!(
                "Failed to setup the output matrix from host to device: {}",
                e
            )));
        }
        Ok(x) => x,
    };
    Ok(MatMulPrepResult {
        ctx,
        stream,
        ptx,
        module,
        function: f,
        launch_args: builder,
        launch_config: cfg,
        out_host: out,
        out_dev,
    })
}

pub fn scalarmatmul<T: Default + Clone + cudarc::driver::DeviceRepr>(
    s: T,
    a: &Matrix<T>,
) -> Result<Matrix<T>, MatrixError> {
    // let ctx = match CudaContext::new(0) {
    //     Err(e) => {
    //         return Err(MatrixError::OutOfMemory(format!(
    //             "Failed to create CUDA context: {}",
    //             e
    //         )));
    //     }
    //     Ok(c) => c,
    // };
    // let stream: Arc<CudaStream> = ctx.default_stream();

    // let n_rows: u32 = a.n_rows as u32;
    // let n_cols: u32 = a.n_cols as u32;
    // let out: Vec<T> = vec![T::default(); (n_rows * n_cols) as usize];
    // let mut out_dev: CudaSlice<T> = match stream.clone_htod(&out) {
    //     Err(e) => {
    //         return Err(MatrixError::OutOfMemory(format!(
    //             "Failed to setup the output matrix from host to device: {}",
    //             e
    //         )));
    //     }
    //     Ok(x) => x,
    // };
    // let ptx = match compile_ptx(SCALARMATMUL) {
    //     Err(e) => {
    //         return Err(MatrixError::CompileError(format!(
    //             "Failed to compile PTX: {}",
    //             e
    //         )));
    //     }
    //     Ok(p) => p,
    // };
    // let module = match ctx.load_module(ptx) {
    //     Err(e) => {
    //         return Err(MatrixError::OutOfMemory(format!(
    //             "Failed to load CUDA module: {}",
    //             e
    //         )));
    //     }
    //     Ok(m) => m,
    // };
    // let f = match module.load_function("cuScalarMatMul") {
    //     Err(e) => {
    //         return Err(MatrixError::OutOfMemory(format!(
    //             "Failed to load CUDA function: {}",
    //             e
    //         )));
    //     }
    //     Ok(f) => f,
    // };
    // let mut builder = stream.launch_builder(&f);
    let n_rows: u32 = a.n_rows as u32;
    let n_cols: u32 = a.n_cols as u32;

    let Q = prep_cuda_kernel(
        n_rows,
        n_cols,
        SCALARMATMUL,
        "cuScalarMatMul",
    )?;


    

    

    Q.builder.arg(&s);
    Q.builder.arg(&a.data);
    Q.builder.arg(&mut out_dev);
    Q.builder.arg(&n_rows);
    Q.builder.arg(&n_cols);
    // let cfg = LaunchConfig {
    //     block_dim: (
    //         BLOCK_SIZE, 
    //         BLOCK_SIZE, 
    //         1
    //     ),
    //     grid_dim: (
    //         (n_cols + BLOCK_SIZE - 1) / BLOCK_SIZE,
    //         (n_rows + BLOCK_SIZE - 1) / BLOCK_SIZE,
    //         1
    //     ),
    //     shared_mem_bytes: 0,
    // };
    unsafe {
        let _ = Q.builder.launch(Q.cfg);
    };
    Matrix::new(out_dev, n_rows as usize, n_cols as usize)
}


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
        Err(e) => {
            return Err(MatrixError::OutOfMemory(format!(
                "Failed to create CUDA context: {}",
                e
            )));
        }
        Ok(c) => c,
    };
    let stream: Arc<CudaStream> = ctx.default_stream();

    let n_rows: u32 = a.n_rows as u32;
    let n_cols: u32 = b.n_cols as u32;
    let p: u32 = a.n_cols as u32;
    let out: Vec<T> = vec![T::default(); (n_rows * n_cols) as usize];
    let mut out_dev: CudaSlice<T> = match stream.clone_htod(&out) {
        Err(e) => {
            return Err(MatrixError::OutOfMemory(format!(
                "Failed to setup the output matrix from host to device: {}",
                e
            )));
        }
        Ok(x) => x,
    };
    let ptx = match compile_ptx(MATMUL) {
        Err(e) => {
            return Err(MatrixError::CompileError(format!(
                "Failed to compile PTX: {}",
                e
            )));
        }
        Ok(p) => p,
    };
    let module = match ctx.load_module(ptx) {
        Err(e) => {
            return Err(MatrixError::OutOfMemory(format!(
                "Failed to load CUDA module: {}",
                e
            )));
        }
        Ok(m) => m,
    };
    let f = match module.load_function("cuMatMul") {
        Err(e) => {
            return Err(MatrixError::OutOfMemory(format!(
                "Failed to load CUDA function: {}",
                e
            )));
        }
        Ok(f) => f,
    };
    let mut builder = stream.launch_builder(&f);
    builder.arg(&a.data);
    builder.arg(&b.data);
    builder.arg(&mut out_dev);
    builder.arg(&n_rows);
    builder.arg(&n_cols);
    builder.arg(&p);
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
    Matrix::new(out_dev, n_rows as usize, n_cols as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_f() -> Result<(), MatrixError> {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let (n, p, m): (usize, usize, usize) = (4, 3, 2);
        let (a_n_rows, a_n_cols): (usize, usize) = (n, p);
        let (b_n_rows, b_n_cols): (usize, usize) = (p, m);
        let (c_n_rows, c_n_cols): (usize, usize) = (n, m);

        let mut a_host: Vec<f32> = (0..(a_n_rows * a_n_cols)).map(|x| x as f32).collect();
        let mut b_host: Vec<f32> = (0..(b_n_rows * b_n_cols)).map(|x| x as f32).collect();
        let mut c_host: Vec<f32> = (0..(c_n_rows * c_n_cols)).map(|x| x as f32).collect();

        let a_dev: CudaSlice<f32> = stream.clone_htod(&a_host).unwrap();
        let b_dev: CudaSlice<f32> = stream.clone_htod(&b_host).unwrap();
        let c_dev: CudaSlice<f32> = stream.clone_htod(&c_host).unwrap();
        let a_matrix = Matrix::new(a_dev, a_n_rows, a_n_cols)?;
        let b_matrix = Matrix::new(b_dev, b_n_rows, b_n_cols)?;
        let c_matrix = Matrix::new(c_dev, c_n_rows, c_n_cols)?;

        println!("a_matrix {:?}", a_matrix);
        println!("b_matrix {:?}", b_matrix);
        println!("c_matrix {:?}", c_matrix);

        stream.memcpy_dtoh(&a_matrix.data, &mut a_host).unwrap();
        stream.memcpy_dtoh(&b_matrix.data, &mut b_host).unwrap();
        stream.memcpy_dtoh(&c_matrix.data, &mut c_host).unwrap();
        println!("Before: a_host {:?}", a_host);
        println!("Before: b_host {:?}", b_host);
        println!("Before: c_host {:?}", c_host);

        let matrix_1 = scalarmatmul(2.0, &a_matrix)?;
        stream.memcpy_dtoh(&matrix_1.data, &mut a_host).unwrap();
        println!("After `scalarmatmul`: a_host {:?}", a_host);


        let out_matrix = matmul(&a_matrix, &b_matrix)?;
        stream.memcpy_dtoh(&out_matrix.data, &mut c_host).unwrap();
        println!("After `matmul`: c_host {:?}", c_host);

        assert_eq!(c_host, vec![10.0, 13.0, 28.0, 40.0, 46.0, 67.0, 64.0, 94.0]);

        Ok(())
    }
}
