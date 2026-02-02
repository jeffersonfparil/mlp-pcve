use crate::linalg::matrix::Matrix;
use cudarc::driver::safe::{CudaContext, CudaFunction, LaunchArgs};
use cudarc::driver::{CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::compile_ptx;
use cudarc::nvrtc::safe::Ptx;
use std::error::Error;
use std::sync::Arc;

const BLOCK_SIZE: u32 = 16;

const CLAMP: &str = "
    extern \"C\" __global__ void cuElementwiseClamp(float* A, float* B, float lower, float upper, int n_rows, int n_cols) {
        // Element-wise clamp kernel implementation
        // Arguments:
        //  - A: input matrix (n_rows x n_cols)
        //  - B: output matrix (n_rows x n_cols)
        //  - lower: lower bound for clamping
        //  - upper: upper bound for clamping
        //  - n_rows: number of rows in A and B
        //  - n_cols: number of columns in A and B
        // Assumes:
        //  - row-major storage
        //  - matrices A and B are of the same size
        int i = (blockIdx.y * blockDim.y) + threadIdx.y; // Row index
        int j = (blockIdx.x * blockDim.x) + threadIdx.x; // Column index
        if ((i < n_rows) && (j < n_cols)) {
            int idx = (i * n_cols) + j; // Linear index for the A and B matrices
            B[idx] = fmaxf(A[idx], lower);
            B[idx] = fminf(B[idx], upper);
        }
    }
";

impl Matrix {
    pub fn clamp(self: &Self, lower: f32, upper: f32) -> Result<Self, Box<dyn Error>> {
        let ptx: Ptx = compile_ptx(CLAMP)?;
        let f: CudaFunction = self
            .data
            .context()
            .load_module(ptx)?
            .load_function("cuElementwiseClamp")?;
        let stream: Arc<CudaStream> = self.data.context().default_stream();
        let mut builder: LaunchArgs = stream.launch_builder(&f);
        let n_rows: u32 = self.n_rows as u32;
        let n_cols: u32 = self.n_cols as u32;
        let out: Vec<f32> = vec![0.0; (n_rows * n_cols) as usize];
        let mut out_dev: CudaSlice<f32> = stream.clone_htod(&out)?;
        builder.arg(&self.data);
        builder.arg(&mut out_dev);
        builder.arg(&lower);
        builder.arg(&upper);
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
    fn test_clamp() -> Result<(), Box<dyn Error>> {
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

        let matrix_1 = a_matrix.clamp(2.0, 5.0)?;
        stream.memcpy_dtoh(&matrix_1.data, &mut a_host)?; // does not interfere with a_matrix because the data in a_host is in CPU while a_matrix is in GPU
        println!("After `clamp`: a_host {:?}", a_host);
        assert_eq!(
            a_host,
            vec![
                2.0_f32, 2.0_f32, 2.0_f32, 3.0_f32, 4.0_f32, 5.0_f32, 5.0_f32, 5.0_f32, 5.0_f32,
                5.0_f32, 5.0_f32, 5.0_f32
            ]
        );

        Ok(())
    }
}
