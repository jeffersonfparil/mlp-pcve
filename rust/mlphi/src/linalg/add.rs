use crate::linalg::matrix::{Matrix, MatrixError};
use cudarc::driver::safe::{CudaContext, CudaFunction, LaunchArgs};
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use cudarc::nvrtc::safe::Ptx;
use std::error::Error;
use std::sync::Arc;

const BLOCK_SIZE: u32 = 16;

const SCALARMATADD: &str = "
    extern \"C\" __global__ void cuScalarMatAdd(float scalar, float* A, float* B, int n_rows, int n_cols) {
        // Scalar-matrix addition kernel implementation
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
            B[idx] = scalar + A[idx];
        }
    }
";

const ELEMENTWISEMATADD: &str = "
    extern \"C\" __global__ void cuElementwiseMatAdd(float* A, float* B, float* C, int n_rows, int n_cols) {
        // Matrix addition kernel implementation
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
            C[idx] = A[idx] + B[idx];
        }
    }
";

const ROWMATADD: &str = "
    extern \"C\" __global__ void cuRowMatAdd(float* A, float* B, float* C, int n_rows, int n_cols) {
        // Row-by-row addition: matrix and column vector addition kernel implementation
        // Arguments:
        //  - A: input matrix (n_rows x n_cols)
        //  - B: input column vector (n_rows x 1)
        //  - C: output matrix (n_rows x n_cols)
        //  - n_rows: number of rows in all A and B matrices
        //  - n_cols: number of columns in A and C matrices
        // Assumes:
        //  - row-major storage
        int i = (blockIdx.y * blockDim.y) + threadIdx.y; // Row index (index of B matrix)
        int j = (blockIdx.x * blockDim.x) + threadIdx.x; // Column index
        if ((i < n_rows) && (j < n_cols)) {
            int idx = (i * n_cols) + j; // Linear index for the A and C matrices
            C[idx] = A[idx] + B[i];
        }
    }
";

const COLMATADD: &str = "
    extern \"C\" __global__ void cuColMatAdd(float* A, float* B, float* C, int n_rows, int n_cols) {
        // Column-by-column addition: matrix and row vector addition kernel implementation
        // Arguments:
        //  - A: input matrix (n_rows x n_cols)
        //  - B: input row vector (1 x n_cols)
        //  - C: output matrix (n_rows x n_cols)
        //  - n_rows: number of rows in all A and C matrices
        //  - n_cols: number of columns in A and B matrices
        // Assumes:
        //  - row-major storage
        int i = (blockIdx.y * blockDim.y) + threadIdx.y; // Row index (index of B matrix)
        int j = (blockIdx.x * blockDim.x) + threadIdx.x; // Column index
        if ((i < n_rows) && (j < n_cols)) {
            int idx = (i * n_cols) + j; // Linear index for the A and C matrices
            C[idx] = A[idx] + B[j];
        }
    }
";

impl Matrix {
    pub fn scalarmatadd(self: &Self, s: f32) -> Result<Self, Box<dyn Error>> {
        let ptx: Ptx = compile_ptx(SCALARMATADD)?;
        let f: CudaFunction = self
            .data
            .context()
            .load_module(ptx)?
            .load_function("cuScalarMatAdd")?;
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

    pub fn elementwisematadd(self: &Self, b: &Self) -> Result<Self, Box<dyn Error>> {
        if (self.n_rows != b.n_rows) | (self.n_cols != b.n_cols) {
            return Err(Box::new(MatrixError::DimensionMismatch(format!(
                "in `elementwisematadd`: self.n_rows ({}) != b.n_rows ({}) and/or self.n_cols ({}) != b.n_cols ({})",
                self.n_rows, b.n_rows, self.n_cols, b.n_cols
            ))));
        }
        let ptx: Ptx = compile_ptx(ELEMENTWISEMATADD)?;
        let f: CudaFunction = self
            .data
            .context()
            .load_module(ptx)?
            .load_function("cuElementwiseMatAdd")?;
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

    pub fn rowmatadd(self: &Self, b: &Self) -> Result<Self, Box<dyn Error>> {
        if (self.n_rows != b.n_rows) | (b.n_cols != 1) {
            return Err(Box::new(MatrixError::DimensionMismatch(format!(
                "in `rowmatadd`: self.n_rows ({}) != b.n_rows ({}) and/or b.n_cols ({}) != 1",
                self.n_rows, b.n_rows, b.n_cols
            ))));
        }
        let ptx: Ptx = compile_ptx(ROWMATADD)?;
        let f: CudaFunction = self
            .data
            .context()
            .load_module(ptx)?
            .load_function("cuRowMatAdd")?;
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

    pub fn colmatadd(self: &Self, b: &Self) -> Result<Self, Box<dyn Error>> {
        if (self.n_cols != b.n_cols) | (b.n_rows != 1) {
            return Err(Box::new(MatrixError::DimensionMismatch(format!(
                "in `colmatadd`: self.n_cols ({}) != b.n_cols ({}) and/or b.n_rows ({}) != 1",
                self.n_cols, b.n_cols, b.n_rows
            ))));
        }
        let ptx: Ptx = compile_ptx(COLMATADD)?;
        let f: CudaFunction = self
            .data
            .context()
            .load_module(ptx)?
            .load_function("cuColMatAdd")?;
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
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_add() -> Result<(), Box<dyn Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let (n, p, m): (usize, usize, usize) = (4, 3, 2);
        let (a_n_rows, a_n_cols): (usize, usize) = (n, p);
        let (b_n_rows, b_n_cols): (usize, usize) = (p, m);
        let (c_n_rows, c_n_cols): (usize, usize) = (n, m);
        let (d_n_rows, d_n_cols): (usize, usize) = (n, 1);
        let (e_n_rows, e_n_cols): (usize, usize) = (1, p);

        let mut a_host: Vec<f32> = (0..(a_n_rows * a_n_cols)).map(|x| x as f32).collect();
        let mut b_host: Vec<f32> = (0..(b_n_rows * b_n_cols)).map(|x| x as f32).collect();
        let mut c_host: Vec<f32> = (0..(c_n_rows * c_n_cols)).map(|x| x as f32).collect();
        let mut d_host: Vec<f32> = (0..(d_n_rows * d_n_cols)).map(|x| x as f32).collect();
        let mut e_host: Vec<f32> = (0..(e_n_rows * e_n_cols)).map(|x| x as f32).collect();

        // Copy data from CPU to GPU, i.e. from *_host into *_matrix
        let a_dev: CudaSlice<f32> = stream.clone_htod(&a_host)?;
        let b_dev: CudaSlice<f32> = stream.clone_htod(&b_host)?;
        let c_dev: CudaSlice<f32> = stream.clone_htod(&c_host)?;
        let d_dev: CudaSlice<f32> = stream.clone_htod(&d_host)?;
        let e_dev: CudaSlice<f32> = stream.clone_htod(&e_host)?;
        let a_matrix = Matrix::new(a_dev, a_n_rows, a_n_cols)?;
        let b_matrix = Matrix::new(b_dev, b_n_rows, b_n_cols)?;
        let c_matrix = Matrix::new(c_dev, c_n_rows, c_n_cols)?;
        let d_matrix = Matrix::new(d_dev, d_n_rows, d_n_cols)?;
        let e_matrix = Matrix::new(e_dev, e_n_rows, e_n_cols)?;

        println!("a_matrix {:?}", a_matrix);
        println!("b_matrix {:?}", b_matrix);
        println!("c_matrix {:?}", c_matrix);
        println!("d_matrix {:?}", d_matrix);
        println!("e_matrix {:?}", c_matrix);

        stream.memcpy_dtoh(&a_matrix.data, &mut a_host)?;
        stream.memcpy_dtoh(&b_matrix.data, &mut b_host)?;
        stream.memcpy_dtoh(&c_matrix.data, &mut c_host)?;
        stream.memcpy_dtoh(&d_matrix.data, &mut d_host)?;
        stream.memcpy_dtoh(&e_matrix.data, &mut e_host)?;
        println!("Before: a_host {:?}", a_host);
        println!("Before: b_host {:?}", b_host);
        println!("Before: c_host {:?}", c_host);
        println!("Before: d_host {:?}", d_host);
        println!("Before: e_host {:?}", e_host);

        let matrix_1 = a_matrix.scalarmatadd(2.0)?;
        stream.memcpy_dtoh(&matrix_1.data, &mut a_host)?; // does not interfere with a_matrix because the data in a_host is in CPU while a_matrix is in GPU
        println!("After `scalarmatadd`: a_host {:?}", a_host);
        assert_eq!(
            a_host,
            vec![
                2. + 0.0,
                2. + 1.0,
                2. + 2.0,
                2. + 3.0,
                2. + 4.0,
                2. + 5.0,
                2. + 6.0,
                2. + 7.0,
                2. + 8.0,
                2. + 9.0,
                2. + 10.0,
                2. + 11.0
            ]
        );

        let matrix_2 = a_matrix.elementwisematadd(&a_matrix)?;
        stream.memcpy_dtoh(&matrix_2.data, &mut a_host)?; // does not interfere with a_matrix because the data in a_host is in CPU while a_matrix is in GPU
        println!("After `elementwisematadd`: a_host {:?}", a_host);
        assert_eq!(
            a_host,
            vec![
                0.0 + 0.0,
                1.0 + 1.0,
                2.0 + 2.0,
                3.0 + 3.0,
                4.0 + 4.0,
                5.0 + 5.0,
                6.0 + 6.0,
                7.0 + 7.0,
                8.0 + 8.0,
                9.0 + 9.0,
                10.0 + 10.0,
                11.0 + 11.0
            ]
        );

        let matrix_3 = a_matrix.rowmatadd(&d_matrix)?;
        stream.memcpy_dtoh(&matrix_3.data, &mut a_host)?; // does not interfere with a_matrix because the data in a_host is in CPU while a_matrix is in GPU
        println!("After `rowmatadd`: a_host {:?}", a_host);
        assert_eq!(
            a_host,
            vec![
                0.0 + 0.0,
                1.0 + 0.0,
                2.0 + 0.0,
                3.0 + 1.0,
                4.0 + 1.0,
                5.0 + 1.0,
                6.0 + 2.0,
                7.0 + 2.0,
                8.0 + 2.0,
                9.0 + 3.0,
                10.0 + 3.0,
                11.0 + 3.0
            ]
        );

        let matrix_4 = a_matrix.colmatadd(&e_matrix)?;
        stream.memcpy_dtoh(&matrix_4.data, &mut a_host)?; // does not interfere with a_matrix because the data in a_host is in CPU while a_matrix is in GPU
        println!("After `colmatadd`: a_host {:?}", a_host);
        assert_eq!(
            a_host,
            vec![
                0.0 + 0.0,
                1.0 + 1.0,
                2.0 + 2.0,
                3.0 + 0.0,
                4.0 + 1.0,
                5.0 + 2.0,
                6.0 + 0.0,
                7.0 + 1.0,
                8.0 + 2.0,
                9.0 + 0.0,
                10.0 + 1.0,
                11.0 + 2.0
            ]
        );

        Ok(())
    }
}
