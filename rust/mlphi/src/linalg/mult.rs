use crate::linalg::matrix::{Matrix, MatrixError};
use cudarc::driver::safe::{CudaContext, CudaFunction, LaunchArgs};
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use cudarc::nvrtc::safe::Ptx;
use std::error::Error;
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

const ELEMENTWISEMATINVERSE: &str = "
    extern \"C\" __global__ void cuElementwiseMatInverse(float* A, float* B, int n_rows, int n_cols) {
        // Element-wise multiplicative inverse kernel implementation
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
            B[idx] = 1.0f / A[idx];
        }
    }
";

const ELEMENTWISEMATPOWER: &str = "
    extern \"C\" __global__ void cuElementwiseMatPower(float power, float* A, float* B, int n_rows, int n_cols) {
        // Element-wise power kernel implementation
        // Arguments:
        //  - power: exponent to raise each element of A
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
            B[idx] = powf(A[idx], power);
        }
    }
";

const ELEMENTWISEMATMUL: &str = "
    extern \"C\" __global__ void cuElementwiseMatMul(float* A, float* B, float* C, int n_rows, int n_cols) {
        // Elementwise matrix multiplication kernel implementation
        // Arguments:
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

const ROWMATMUL: &str = "
    extern \"C\" __global__ void cuRowMatMul(float* A, float* B, float* C, int n_rows, int n_cols) {
        // Row-by-row: matrix by column vector elementwise multiplication kernel implementation
        // Arguments:
        //  - A: input matrix (n_rows x n_cols)
        //  - B: input column vector (n_rows x 1)
        //  - C: output matrix (n_rows x n_cols)
        //  - n_rows: number of rows in A and B matrices
        //  - n_cols: number of columns in A and C matrices
        // Assumes:
        //  - row-major storage
        int i = (blockIdx.y * blockDim.y) + threadIdx.y; // Row index
        int j = (blockIdx.x * blockDim.x) + threadIdx.x; // Column index
        if ((i < n_rows) && (j < n_cols)) {
            int idx = (i * n_cols) + j; // Linear index for the A and B matrices
            C[idx] = A[idx] * B[i];
        }
    }
";

const COLMATMUL: &str = "
    extern \"C\" __global__ void cuColMatMul(float* A, float* B, float* C, int n_rows, int n_cols) {
        // Column-by-column: matrix by row vector elementwise multiplication kernel implementation
        // Arguments:
        //  - A: input matrix 1 (n_rows x n_cols)
        //  - B: input row vector (1 x n_cols)
        //  - C: output matrix (n_rows x n_cols)
        //  - n_rows: number of rows in A and C matrices
        //  - n_cols: number of columns in A and B matrices
        // Assumes:
        //  - row-major storage
        int i = (blockIdx.y * blockDim.y) + threadIdx.y; // Row index
        int j = (blockIdx.x * blockDim.x) + threadIdx.x; // Column index
        if ((i < n_rows) && (j < n_cols)) {
            int idx = (i * n_cols) + j; // Linear index for the A and B matrices
            C[idx] = A[idx] * B[j];
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
        //  - matrices A and B are compatible
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

const MATMULT0: &str = "
    extern \"C\" __global__ void cuMatMulT0(float* A, float* B, float*C, int p_a, int p_b, int n) {
        // Matrix multiplication kernel implementation for X'X
        // Arguments:
        //  - A: left matrix (n x p_a)
        //  - B: right matrix (n x p_b)
        //  - C: output matrix (p_a x p_b)
        //  - n: number of rows in A and B
        //  - p_a: number of columns in A and number of rows in C
        //  - p_b: number of columns in B and C
        // Assumes:
        //  - row-major storage
        //  - transpose of matrix A and the matrix B are compatible
        //  - matrix C is the correct size to hold the result
        int i = (blockIdx.y * blockDim.y) + threadIdx.y; // Row index of C to compute
        int j = (blockIdx.x * blockDim.x) + threadIdx.x; // Column index of C to compute
        if ((i < p_a) && (j < p_b)) {
            float s = 0.0f;
            for (int k=0; k<n; k++) {
                s += A[(k*p_a)+i] * B[(k*p_b)+j];
            }
            C[(i*p_b)+j] = s;
        }
    }
";

const MATMUL0T: &str = "
    extern \"C\" __global__ void cuMatMul0T(float* A, float* B, float*C, int n_a, int n_b, int p) {
        // Matrix multiplication kernel implementation for XX'
        // Arguments:
        //  - A: left matrix (n_a x p)
        //  - B: right matrix (n_b x p)
        //  - C: output matrix (n_a x n_b)
        //  - n_a: number of rows in A and C
        //  - n_b: number of rows in B and number of columns in C
        //  - p: number of columns in A and B
        // Assumes:
        //  - row-major storage
        //  - matrix A and the transpose of matrix B are compatible
        //  - matrix C is the correct size to hold the result
        int i = (blockIdx.y * blockDim.y) + threadIdx.y; // Row index of C to compute
        int j = (blockIdx.x * blockDim.x) + threadIdx.x; // Column index of C to compute
        if ((i < n_a) && (j < n_b)) {
            float s = 0.0f;
            for (int k=0; k<p; k++) {
                s += A[(i*p)+k] * B[(j*p)+k];
            }
            C[(i*n_b)+j] = s;
        }
    }
";

const MATMULTT: &str = "
    extern \"C\" __global__ void cuMatMulTT(float* A, float* B, float*C, int p_a, int n_b, int m) {
        // Matrix multiplication kernel implementation for X'X
        // Arguments:
        //  - A: left matrix (m x p_a)
        //  - B: right matrix (n_b x m)
        //  - C: output matrix (p_a x n_b)
        //  - m: number of rows in A and number of columns B
        //  - p_a: number of columns in A and number of rows in C
        //  - n_b: number of rows in B and number of columns C
        // Assumes:
        //  - row-major storage
        //  - transpose of matrix A and the transpose of matrix B are compatible
        //  - matrix C is the correct size to hold the result
        int i = (blockIdx.y * blockDim.y) + threadIdx.y; // Row index of C to compute
        int j = (blockIdx.x * blockDim.x) + threadIdx.x; // Column index of C to compute
        if ((i < p_a) && (j < n_b)) {
            float s = 0.0f;
            for (int k=0; k<m; k++) {
                s += A[(k*p_a)+i] * B[(j*m)+k];
            }
            C[(i*n_b)+j] = s;
        }
    }
";

impl Matrix {
    pub fn scalarmatmul(self: &Self, s: f32) -> Result<Self, Box<dyn Error>> {
        let ptx: Ptx = compile_ptx(SCALARMATMUL)?;
        let f: CudaFunction = self
            .data
            .context()
            .load_module(ptx)?
            .load_function("cuScalarMatMul")?;
        let stream: Arc<CudaStream> = self.data.context().default_stream();
        let mut builder: LaunchArgs = stream.launch_builder(&f);
        let n_rows: u32 = self.n_rows as u32;
        let n_cols: u32 = self.n_cols as u32;
        let out: Vec<f32> = vec![0.0; (n_rows * n_cols) as usize];
        let mut out_dev: CudaSlice<f32> = stream.clone_htod(&out)?;
        builder.arg(&s);
        builder.arg(&self.data);
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
        Ok(Self::new(out_dev, n_rows as usize, n_cols as usize)?)
    }

    pub fn elementwisematinverse(self: &Self) -> Result<Self, Box<dyn Error>> {
        let ptx: Ptx = compile_ptx(ELEMENTWISEMATINVERSE)?;
        let f: CudaFunction = self
            .data
            .context()
            .load_module(ptx)?
            .load_function("cuElementwiseMatInverse")?;
        let stream: Arc<CudaStream> = self.data.context().default_stream();
        let mut builder: LaunchArgs = stream.launch_builder(&f);
        let n_rows: u32 = self.n_rows as u32;
        let n_cols: u32 = self.n_cols as u32;
        let out: Vec<f32> = vec![0.0; (n_rows * n_cols) as usize];
        let mut out_dev: CudaSlice<f32> = stream.clone_htod(&out)?;
        builder.arg(&self.data);
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
        Ok(Self::new(out_dev, n_rows as usize, n_cols as usize)?)
    }

    pub fn elementwisematpower(self: &Self, power: f32) -> Result<Self, Box<dyn Error>> {
        let ptx: Ptx = compile_ptx(ELEMENTWISEMATPOWER)?;
        let f: CudaFunction = self
            .data
            .context()
            .load_module(ptx)?
            .load_function("cuElementwiseMatPower")?;
        let stream: Arc<CudaStream> = self.data.context().default_stream();
        let mut builder: LaunchArgs = stream.launch_builder(&f);
        let n_rows: u32 = self.n_rows as u32;
        let n_cols: u32 = self.n_cols as u32;
        let out: Vec<f32> = vec![0.0; (n_rows * n_cols) as usize];
        let mut out_dev: CudaSlice<f32> = stream.clone_htod(&out)?;
        builder.arg(&power);
        builder.arg(&self.data);
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
        Ok(Self::new(out_dev, n_rows as usize, n_cols as usize)?)
    }

    pub fn elementwisematmul(self: &Self, b: &Self) -> Result<Self, Box<dyn Error>> {
        if (self.n_rows != b.n_rows) | (self.n_cols != b.n_cols) {
            return Err(Box::new(MatrixError::DimensionMismatch(format!(
                "in `elementwisematmul`: self.n_rows ({}) != b.n_rows ({}) and/or self.n_cols ({}) != b.n_cols ({})",
                self.n_rows, b.n_rows, self.n_cols, b.n_cols
            ))));
        }
        let ptx: Ptx = compile_ptx(ELEMENTWISEMATMUL)?;
        let f: CudaFunction = self
            .data
            .context()
            .load_module(ptx)?
            .load_function("cuElementwiseMatMul")?;
        let stream: Arc<CudaStream> = self.data.context().default_stream();
        let mut builder: LaunchArgs = stream.launch_builder(&f);
        let n_rows: u32 = self.n_rows as u32;
        let n_cols: u32 = self.n_cols as u32;
        let out: Vec<f32> = vec![0.0; (n_rows * n_cols) as usize];
        let mut out_dev: CudaSlice<f32> = stream.clone_htod(&out)?;
        builder.arg(&self.data);
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
        Ok(Self::new(out_dev, n_rows as usize, n_cols as usize)?)
    }

    pub fn rowmatmul(self: &Self, b: &Self) -> Result<Self, Box<dyn Error>> {
        if (self.n_rows != b.n_rows) | (b.n_cols != 1) {
            return Err(Box::new(MatrixError::DimensionMismatch(format!(
                "in `rowmatmul`: self.n_rows ({}) != b.n_rows ({}) and/or b.n_cols ({}) != 1",
                self.n_rows, b.n_rows, b.n_cols
            ))));
        }
        let ptx: Ptx = compile_ptx(ROWMATMUL)?;
        let f: CudaFunction = self
            .data
            .context()
            .load_module(ptx)?
            .load_function("cuRowMatMul")?;
        let stream: Arc<CudaStream> = self.data.context().default_stream();
        let mut builder: LaunchArgs = stream.launch_builder(&f);
        let n_rows: u32 = self.n_rows as u32;
        let n_cols: u32 = self.n_cols as u32;
        let out: Vec<f32> = vec![0.0; (n_rows * n_cols) as usize];
        let mut out_dev: CudaSlice<f32> = stream.clone_htod(&out)?;
        builder.arg(&self.data);
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
        Ok(Self::new(out_dev, n_rows as usize, n_cols as usize)?)
    }

    pub fn colmatmul(self: &Self, b: &Self) -> Result<Self, Box<dyn Error>> {
        if (self.n_cols != b.n_cols) | (b.n_rows != 1) {
            return Err(Box::new(MatrixError::DimensionMismatch(format!(
                "in `colmatmul`: self.n_cols ({}) != b.n_cols ({}) and/or b.n_rows ({}) != 1",
                self.n_cols, b.n_cols, b.n_rows
            ))));
        }
        let ptx: Ptx = compile_ptx(COLMATMUL)?;
        let f: CudaFunction = self
            .data
            .context()
            .load_module(ptx)?
            .load_function("cuColMatMul")?;
        let stream: Arc<CudaStream> = self.data.context().default_stream();
        let mut builder: LaunchArgs = stream.launch_builder(&f);
        let n_rows: u32 = self.n_rows as u32;
        let n_cols: u32 = self.n_cols as u32;
        let out: Vec<f32> = vec![0.0; (n_rows * n_cols) as usize];
        let mut out_dev: CudaSlice<f32> = stream.clone_htod(&out)?;
        builder.arg(&self.data);
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
        Ok(Self::new(out_dev, n_rows as usize, n_cols as usize)?)
    }

    pub fn matmul(self: &Self, b: &Self) -> Result<Self, Box<dyn Error>> {
        if self.n_cols != b.n_rows {
            return Err(Box::new(MatrixError::DimensionMismatch(format!(
                "in `matmul`: self.n_cols ({}) != b.n_rows ({})",
                self.n_cols, b.n_rows
            ))));
        }
        let ptx: Ptx = compile_ptx(MATMUL)?;
        let f: CudaFunction = self
            .data
            .context()
            .load_module(ptx)?
            .load_function("cuMatMul")?;
        let stream: Arc<CudaStream> = self.data.context().default_stream();
        let mut builder: LaunchArgs = stream.launch_builder(&f);
        let n_rows: u32 = self.n_rows as u32;
        let n_cols: u32 = b.n_cols as u32;
        let p: u32 = self.n_cols as u32;
        let out: Vec<f32> = vec![0.0; (n_rows * n_cols) as usize];
        let mut out_dev: CudaSlice<f32> = stream.clone_htod(&out)?;
        builder.arg(&self.data);
        builder.arg(&b.data);
        builder.arg(&mut out_dev);
        builder.arg(&n_rows);
        builder.arg(&n_cols);
        builder.arg(&p);
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
        Ok(Self::new(out_dev, n_rows as usize, n_cols as usize)?)
    }

    pub fn matmult0(self: &Self, b: &Self) -> Result<Self, Box<dyn Error>> {
        if self.n_rows != b.n_rows {
            return Err(Box::new(MatrixError::DimensionMismatch(format!(
                "in `matmult0`: self.n_rows ({}) != b.n_rows ({})",
                self.n_rows, b.n_rows
            ))));
        }
        let ptx: Ptx = compile_ptx(MATMULT0)?;
        let f: CudaFunction = self
            .data
            .context()
            .load_module(ptx)?
            .load_function("cuMatMulT0")?;
        let stream: Arc<CudaStream> = self.data.context().default_stream();
        let mut builder: LaunchArgs = stream.launch_builder(&f);
        let p_a: u32 = self.n_cols as u32;
        let p_b: u32 = b.n_cols as u32;
        let n: u32 = self.n_rows as u32;
        let out: Vec<f32> = vec![0.0; (p_a * p_b) as usize];
        let mut out_dev: CudaSlice<f32> = stream.clone_htod(&out)?;
        builder.arg(&self.data);
        builder.arg(&b.data);
        builder.arg(&mut out_dev);
        builder.arg(&p_a);
        builder.arg(&p_b);
        builder.arg(&n);
        let cfg = LaunchConfig {
            block_dim: (BLOCK_SIZE, BLOCK_SIZE, 1),
            grid_dim: (
                (p_b + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (p_a + BLOCK_SIZE - 1) / BLOCK_SIZE,
                1,
            ),
            shared_mem_bytes: 0,
        };
        unsafe {
            let _ = builder.launch(cfg);
        };
        Ok(Self::new(out_dev, p_a as usize, p_b as usize)?)
    }

    pub fn matmul0t(self: &Self, b: &Self) -> Result<Self, Box<dyn Error>> {
        if self.n_cols != b.n_cols {
            return Err(Box::new(MatrixError::DimensionMismatch(format!(
                "in `matmul0t`: self.n_cols ({}) != b.n_cols ({})",
                self.n_cols, b.n_cols
            ))));
        }
        let ptx: Ptx = compile_ptx(MATMUL0T)?;
        let f: CudaFunction = self
            .data
            .context()
            .load_module(ptx)?
            .load_function("cuMatMul0T")?;
        let stream: Arc<CudaStream> = self.data.context().default_stream();
        let mut builder: LaunchArgs = stream.launch_builder(&f);
        let n_a: u32 = self.n_rows as u32;
        let n_b: u32 = b.n_rows as u32;
        let p: u32 = self.n_cols as u32;
        let out: Vec<f32> = vec![0.0; (n_a * n_b) as usize];
        let mut out_dev: CudaSlice<f32> = stream.clone_htod(&out)?;
        builder.arg(&self.data);
        builder.arg(&b.data);
        builder.arg(&mut out_dev);
        builder.arg(&n_a);
        builder.arg(&n_b);
        builder.arg(&p);
        let cfg = LaunchConfig {
            block_dim: (BLOCK_SIZE, BLOCK_SIZE, 1),
            grid_dim: (
                (n_b + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (n_a + BLOCK_SIZE - 1) / BLOCK_SIZE,
                1,
            ),
            shared_mem_bytes: 0,
        };
        unsafe {
            let _ = builder.launch(cfg);
        };
        Ok(Self::new(out_dev, n_a as usize, n_b as usize)?)
    }

    pub fn matmultt(self: &Self, b: &Self) -> Result<Self, Box<dyn Error>> {
        if self.n_rows != b.n_cols {
            return Err(Box::new(MatrixError::DimensionMismatch(format!(
                "in `matmultt`: self.n_rows ({}) != b.n_cols ({})",
                self.n_rows, b.n_cols
            ))));
        }
        let ptx: Ptx = compile_ptx(MATMULTT)?;
        let f: CudaFunction = self
            .data
            .context()
            .load_module(ptx)?
            .load_function("cuMatMulTT")?;
        let stream: Arc<CudaStream> = self.data.context().default_stream();
        let mut builder: LaunchArgs = stream.launch_builder(&f);
        let p_a: u32 = self.n_cols as u32;
        let n_b: u32 = b.n_rows as u32;
        let m: u32 = self.n_rows as u32;
        let out: Vec<f32> = vec![0.0; (p_a * n_b) as usize];
        let mut out_dev: CudaSlice<f32> = stream.clone_htod(&out)?;
        builder.arg(&self.data);
        builder.arg(&b.data);
        builder.arg(&mut out_dev);
        builder.arg(&p_a);
        builder.arg(&n_b);
        builder.arg(&m);
        let cfg = LaunchConfig {
            block_dim: (BLOCK_SIZE, BLOCK_SIZE, 1),
            grid_dim: (
                (n_b + BLOCK_SIZE - 1) / BLOCK_SIZE,
                (p_a + BLOCK_SIZE - 1) / BLOCK_SIZE,
                1,
            ),
            shared_mem_bytes: 0,
        };
        unsafe {
            let _ = builder.launch(cfg);
        };
        Ok(Self::new(out_dev, p_a as usize, n_b as usize)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_mult() -> Result<(), Box<dyn Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let (n, p, m): (usize, usize, usize) = (4, 3, 2);
        let (a_n_rows, a_n_cols): (usize, usize) = (n, p);
        let (b_n_rows, b_n_cols): (usize, usize) = (p, m);
        let (c_n_rows, c_n_cols): (usize, usize) = (n, m);
        let (d_n_rows, d_n_cols): (usize, usize) = (n, 1);
        let (e_n_rows, e_n_cols): (usize, usize) = (1, p);
        let (f_n_rows, f_n_cols): (usize, usize) = (p, m);
        let (g_n_rows, g_n_cols): (usize, usize) = (m, n);

        let mut a_host: Vec<f32> = (0..(a_n_rows * a_n_cols)).map(|x| x as f32).collect();
        let mut b_host: Vec<f32> = (0..(b_n_rows * b_n_cols)).map(|x| x as f32).collect();
        let mut c_host: Vec<f32> = (0..(c_n_rows * c_n_cols)).map(|x| x as f32).collect();
        let mut d_host: Vec<f32> = (0..(d_n_rows * d_n_cols)).map(|x| x as f32).collect();
        let mut e_host: Vec<f32> = (0..(e_n_rows * e_n_cols)).map(|x| x as f32).collect();
        let mut f_host: Vec<f32> = (0..(f_n_rows * f_n_cols)).map(|x| x as f32).collect();
        let mut g_host: Vec<f32> = (0..(g_n_rows * g_n_cols)).map(|x| x as f32).collect();

        // Copy data from CPU to GPU, i.e. from *_host into *_matrix
        let a_dev: CudaSlice<f32> = stream.clone_htod(&a_host)?;
        let b_dev: CudaSlice<f32> = stream.clone_htod(&b_host)?;
        let c_dev: CudaSlice<f32> = stream.clone_htod(&c_host)?;
        let d_dev: CudaSlice<f32> = stream.clone_htod(&d_host)?;
        let e_dev: CudaSlice<f32> = stream.clone_htod(&e_host)?;
        let f_dev: CudaSlice<f32> = stream.clone_htod(&f_host)?;
        let g_dev: CudaSlice<f32> = stream.clone_htod(&g_host)?;

        let a_matrix = Matrix::new(a_dev, a_n_rows, a_n_cols)?;
        let b_matrix = Matrix::new(b_dev, b_n_rows, b_n_cols)?;
        let c_matrix = Matrix::new(c_dev, c_n_rows, c_n_cols)?;
        let d_matrix = Matrix::new(d_dev, d_n_rows, d_n_cols)?;
        let e_matrix = Matrix::new(e_dev, e_n_rows, e_n_cols)?;
        let f_matrix = Matrix::new(f_dev, f_n_rows, f_n_cols)?;
        let g_matrix = Matrix::new(g_dev, g_n_rows, g_n_cols)?;

        println!("a_matrix {:?}", a_matrix);
        println!("b_matrix {:?}", b_matrix);
        println!("c_matrix {:?}", c_matrix);
        println!("d_matrix {:?}", d_matrix);
        println!("e_matrix {:?}", e_matrix);
        println!("f_matrix {:?}", f_matrix);
        println!("g_matrix {:?}", g_matrix);

        stream.memcpy_dtoh(&a_matrix.data, &mut a_host)?;
        stream.memcpy_dtoh(&b_matrix.data, &mut b_host)?;
        stream.memcpy_dtoh(&c_matrix.data, &mut c_host)?;
        stream.memcpy_dtoh(&d_matrix.data, &mut d_host)?;
        stream.memcpy_dtoh(&e_matrix.data, &mut e_host)?;
        stream.memcpy_dtoh(&f_matrix.data, &mut f_host)?;
        stream.memcpy_dtoh(&g_matrix.data, &mut g_host)?;

        println!("Before: a_host {:?}", a_host);
        println!("Before: b_host {:?}", b_host);
        println!("Before: c_host {:?}", c_host);
        println!("Before: d_host {:?}", d_host);
        println!("Before: e_host {:?}", e_host);
        println!("Before: f_host {:?}", f_host);
        println!("Before: g_host {:?}", g_host);

        let matrix_0 = a_matrix.scalarmatmul(2.0)?;
        stream.memcpy_dtoh(&matrix_0.data, &mut a_host)?; // does not interfere with a_matrix because the data in a_host is in CPU while a_matrix is in GPU
        println!("After `scalarmatmul`: a_host {:?}", a_host);
        assert_eq!(
            a_host,
            vec![
                2. * 0.0,
                2. * 1.0,
                2. * 2.0,
                2. * 3.0,
                2. * 4.0,
                2. * 5.0,
                2. * 6.0,
                2. * 7.0,
                2. * 8.0,
                2. * 9.0,
                2. * 10.0,
                2. * 11.0
            ]
        );

        let matrix_1 = a_matrix.elementwisematinverse()?;
        stream.memcpy_dtoh(&matrix_1.data, &mut a_host)?; // does not interfere with a_matrix because the data in a_host is in CPU while a_matrix is in GPU
        println!("After `elementwisematinverse`: a_host {:?}", a_host);
        assert_eq!(
            a_host,
            vec![
                1. / 0.0,
                1. / 1.0,
                1. / 2.0,
                1. / 3.0,
                1. / 4.0,
                1. / 5.0,
                1. / 6.0,
                1. / 7.0,
                1. / 8.0,
                1. / 9.0,
                1. / 10.0,
                1. / 11.0
            ]
        );

        let matrix_p = a_matrix.elementwisematpower(2.0)?;
        stream.memcpy_dtoh(&matrix_p.data, &mut a_host)?; // does not interfere with a_matrix because the data in a_host is in CPU while a_matrix is in GPU
        println!("After `elementwisematinverse`: a_host {:?}", a_host);
        assert_eq!(
            a_host,
            vec![
                0.0_f32.powf(2.0),
                1.0_f32.powf(2.0),
                2.0_f32.powf(2.0),
                3.0_f32.powf(2.0),
                4.0_f32.powf(2.0),
                5.0_f32.powf(2.0),
                6.0_f32.powf(2.0),
                7.0_f32.powf(2.0),
                8.0_f32.powf(2.0),
                9.0_f32.powf(2.0),
                10.0_f32.powf(2.0),
                11.0_f32.powf(2.0)
            ]
        );

        let matrix_2 = a_matrix.elementwisematmul(&a_matrix)?;
        stream.memcpy_dtoh(&matrix_2.data, &mut a_host)?; // does not interfere with a_matrix because the data in a_host is in CPU while a_matrix is in GPU
        println!("After `elementwisematmul`: a_host {:?}", a_host);
        assert_eq!(
            a_host,
            vec![
                0.0 * 0.0,
                1.0 * 1.0,
                2.0 * 2.0,
                3.0 * 3.0,
                4.0 * 4.0,
                5.0 * 5.0,
                6.0 * 6.0,
                7.0 * 7.0,
                8.0 * 8.0,
                9.0 * 9.0,
                10.0 * 10.0,
                11.0 * 11.0
            ]
        );

        let matrix_3 = a_matrix.rowmatmul(&d_matrix)?;
        stream.memcpy_dtoh(&matrix_3.data, &mut a_host)?; // does not interfere with a_matrix because the data in a_host is in CPU while a_matrix is in GPU
        println!("After `rowmatmul`: a_host {:?}", a_host);
        assert_eq!(
            a_host,
            vec![
                0.0 * 0.0,
                1.0 * 0.0,
                2.0 * 0.0,
                3.0 * 1.0,
                4.0 * 1.0,
                5.0 * 1.0,
                6.0 * 2.0,
                7.0 * 2.0,
                8.0 * 2.0,
                9.0 * 3.0,
                10.0 * 3.0,
                11.0 * 3.0
            ]
        );

        let matrix_4 = a_matrix.colmatmul(&e_matrix)?;
        stream.memcpy_dtoh(&matrix_4.data, &mut a_host)?; // does not interfere with a_matrix because the data in a_host is in CPU while a_matrix is in GPU
        println!("After `colmatmul`: a_host {:?}", a_host);
        assert_eq!(
            a_host,
            vec![
                0.0 * 0.0,
                1.0 * 1.0,
                2.0 * 2.0,
                3.0 * 0.0,
                4.0 * 1.0,
                5.0 * 2.0,
                6.0 * 0.0,
                7.0 * 1.0,
                8.0 * 2.0,
                9.0 * 0.0,
                10.0 * 1.0,
                11.0 * 2.0
            ]
        );

        let matrix_5 = a_matrix.matmul(&b_matrix)?;
        stream.memcpy_dtoh(&matrix_5.data, &mut c_host)?;
        println!("After `matmul`: c_host {:?}", c_host);
        assert_eq!(c_host, vec![10.0, 13.0, 28.0, 40.0, 46.0, 67.0, 64.0, 94.0]);

        let matrix_6 = a_matrix.matmult0(&c_matrix)?;
        stream.memcpy_dtoh(&matrix_6.data, &mut f_host)?;
        println!("After `matmult0`: f_host {:?}", f_host);
        assert_eq!(f_host, vec![84.0, 102.0, 96.0, 118.0, 108.0, 134.0]);

        let matrix_7 = c_matrix.matmul0t(&f_matrix)?;
        stream.memcpy_dtoh(&matrix_7.data, &mut a_host)?;
        println!("After `matmul0t`: a_host {:?}", a_host);
        assert_eq!(
            a_host,
            vec![
                1.0, 3.0, 5.0, 3.0, 13.0, 23.0, 5.0, 23.0, 41.0, 7.0, 33.0, 59.0
            ]
        );

        let matrix_8 = b_matrix.matmultt(&a_matrix)?;
        stream.memcpy_dtoh(&matrix_8.data, &mut g_host)?;
        println!("After `matmultt`: g_host {:?}", g_host);
        assert_eq!(g_host, vec![10.0, 28.0, 46.0, 64.0, 13.0, 40.0, 67.0, 94.0]);

        Ok(())
    }
}
