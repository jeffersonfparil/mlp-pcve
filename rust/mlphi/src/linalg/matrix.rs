use cudarc::driver::{CudaContext, CudaSlice};
use std::error::Error;
use std::fmt;

/// Matrix is stored in GPU memory
/// I have decided to store data in a row-major format just because I feel like it :-P
/// Also all floats are stored as f32
#[repr(C)]
#[derive(Debug, Clone)]
pub struct Matrix {
    pub n_rows: usize,
    pub n_cols: usize,
    pub data: CudaSlice<f32>,
}

/// Implement methods for Matrix
impl Matrix {
    pub fn new(data: CudaSlice<f32>, n_rows: usize, n_cols: usize) -> Result<Self, MatrixError> {
        let n = data.len();
        if n != n_rows * n_cols {
            return Err(MatrixError::DimensionMismatch(format!(
                "initialising matrix: Data length {} does not match matrix dimensions {}x{}={}",
                n,
                n_rows,
                n_cols,
                n_rows * n_cols
            )));
        }
        let out = Self {
            n_rows: n_rows,
            n_cols: n_cols,
            data: data,
        };
        Ok(out)
    }
    pub fn slice(
        self: &Self,
        row_indexes: &Vec<usize>,
        col_indexes: &Vec<usize>,
    ) -> Result<Self, Box<dyn Error>> {
        let n_rows = row_indexes.len();
        let n_cols = col_indexes.len();
        let mut destination: Vec<f32> = vec![0.0; n_rows * n_cols];
        let stream = self.data.context().default_stream();
        let mut source: Vec<f32> = vec![0.0; self.n_rows * self.n_cols];
        stream.memcpy_dtoh(&self.data, &mut source)?;
        for (i, &row_idx) in row_indexes.iter().enumerate() {
            for (j, &col_idx) in col_indexes.iter().enumerate() {
                destination[i * n_cols + j] = source[row_idx * self.n_cols + col_idx];
            }
        }
        let data_dev: CudaSlice<f32> = stream.clone_htod(&destination)?;
        Ok(Matrix::new(data_dev, n_rows, n_cols)?)
    }
    pub fn to_host(&self) -> Result<Vec<f32>, Box<dyn Error>> {
        let mut vec_host: Vec<f32> = vec![0.0; self.n_rows * self.n_cols];
        let stream = self.data.context().default_stream();
        stream.memcpy_dtoh(&self.data, &mut vec_host)?;
        Ok(vec_host)
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.n_rows == 0 || self.n_cols == 0 {
            write!(
                f,
                "[] (rows={}; cols={}; len={} length; bytes={})",
                self.n_rows,
                self.n_cols,
                self.data.len(),
                self.data.num_bytes(),
            )
        } else if (self.n_rows >= 4) & (self.n_cols >= 4) {
            let idx_rows: Vec<usize> = vec![0, 1, 2, self.n_rows - 1];
            let idx_cols: Vec<usize> = vec![0, 1, 2, self.n_cols - 1];
            let for_printing: Matrix = match self.slice(&idx_rows, &idx_cols) {
                Ok(mat) => mat,
                Err(_) => return write!(f, "Matrix data could not be sliced for display."),
            };
            let vec_host = match for_printing.to_host() {
                Ok(vec) => vec,
                Err(_) => return write!(f, "Matrix data could not be copied to host for display."),
            };
            write!(
                f,
                "Dimension: rows={}; cols={}; len={} length; bytes={}\n⎡ {:.5} {:.5} {:.5} ... {:.5} ⎤\n⎢ {:.5} {:.5} {:.5} ... {:.5} ⎥\n⎢ {:.5} {:.5} {:.5} ... {:.5} ⎥\n⎢ ....... ....... ....... ... ....... ⎥\n⎣ {:.5} {:.5} {:.5} ... {:.5} ⎦\n",
                self.n_rows,
                self.n_cols,
                self.data.len(),
                self.data.num_bytes(),
                vec_host[0],
                vec_host[1],
                vec_host[2],
                vec_host[3],
                vec_host[4],
                vec_host[5],
                vec_host[6],
                vec_host[7],
                vec_host[8],
                vec_host[9],
                vec_host[10],
                vec_host[11],
                vec_host[12],
                vec_host[13],
                vec_host[14],
                vec_host[15],
            )
        } else if (self.n_rows >= 4) & (self.n_cols == 1) {
            let idx_rows: Vec<usize> = vec![0, 1, 2, self.n_rows - 1];
            let idx_cols: Vec<usize> = vec![0];
            let for_printing: Matrix = match self.slice(&idx_rows, &idx_cols) {
                Ok(mat) => mat,
                Err(_) => return write!(f, "Matrix data could not be sliced for display."),
            };
            let vec_host = match for_printing.to_host() {
                Ok(vec) => vec,
                Err(_) => return write!(f, "Matrix data could not be copied to host for display."),
            };
            write!(
                f,
                "Dimension: rows={}; cols={}; len={} length; bytes={}\n⎡ {:.5} ⎤\n⎢ {:.5} ⎥\n⎢ {:.5} ⎥\n⎢ ....... ⎥\n⎣ {:.5} ⎦\n",
                self.n_rows,
                self.n_cols,
                self.data.len(),
                self.data.num_bytes(),
                vec_host[0],
                vec_host[1],
                vec_host[2],
                vec_host[3],
            )
        } else if (self.n_rows == 1) & (self.n_cols >= 4) {
            let idx_rows: Vec<usize> = vec![0];
            let idx_cols: Vec<usize> = vec![0, 1, 2, self.n_cols - 1];
            let for_printing: Matrix = match self.slice(&idx_rows, &idx_cols) {
                Ok(mat) => mat,
                Err(_) => return write!(f, "Matrix data could not be sliced for display."),
            };
            let vec_host = match for_printing.to_host() {
                Ok(vec) => vec,
                Err(_) => return write!(f, "Matrix data could not be copied to host for display."),
            };
            write!(
                f,
                "Dimension: rows={}; cols={}; len={} length; bytes={}\n[ {:.5} {:.5} {:.5} ... {:.5} ]\n",
                self.n_rows,
                self.n_cols,
                self.data.len(),
                self.data.num_bytes(),
                vec_host[0],
                vec_host[1],
                vec_host[2],
                vec_host[3],
            )
        } else if (self.n_rows == 3) & (self.n_cols == 3) {
            let vec_host = match self.to_host() {
                Ok(vec) => vec,
                Err(_) => return write!(f, "Matrix data could not be copied to host for display."),
            };
            write!(
                f,
                "Dimension: rows={}; cols={}; len={} length; bytes={}\n⎡ {:.5} {:.5} {:.5} ⎤\n⎢ {:.5} {:.5} {:.5} ⎥\n⎣ {:.5} {:.5} {:.5} ⎦\n",
                self.n_rows,
                self.n_cols,
                self.data.len(),
                self.data.num_bytes(),
                vec_host[0],
                vec_host[1],
                vec_host[2],
                vec_host[3],
                vec_host[4],
                vec_host[5],
                vec_host[6],
                vec_host[7],
                vec_host[8],
            )
        } else if (self.n_rows == 3) & (self.n_cols == 2) {
            let vec_host = match self.to_host() {
                Ok(vec) => vec,
                Err(_) => return write!(f, "Matrix data could not be copied to host for display."),
            };
            write!(
                f,
                "Dimension: rows={}; cols={}; len={} length; bytes={}\n⎡ {:.5} {:.5} ⎤\n⎢ {:.5} {:.5} ⎥\n⎣ {:.5} {:.5} ⎦\n",
                self.n_rows,
                self.n_cols,
                self.data.len(),
                self.data.num_bytes(),
                vec_host[0],
                vec_host[1],
                vec_host[2],
                vec_host[3],
                vec_host[4],
                vec_host[5],
            )
        } else if (self.n_rows == 3) & (self.n_cols == 1) {
            let vec_host = match self.to_host() {
                Ok(vec) => vec,
                Err(_) => return write!(f, "Matrix data could not be copied to host for display."),
            };
            write!(
                f,
                "Dimension: rows={}; cols={}; len={} length; bytes={}\n⎡ {:.5} ⎤\n⎢ {:.5} ⎥\n⎣ {:.5} ⎦\n",
                self.n_rows,
                self.n_cols,
                self.data.len(),
                self.data.num_bytes(),
                vec_host[0],
                vec_host[1],
                vec_host[2],
            )
        } else if (self.n_rows == 2) & (self.n_cols == 3) {
            let vec_host = match self.to_host() {
                Ok(vec) => vec,
                Err(_) => return write!(f, "Matrix data could not be copied to host for display."),
            };
            write!(
                f,
                "Dimension: rows={}; cols={}; len={} length; bytes={}\n⎡ {:.5} {:.5} {:.5} ⎤\n⎣ {:.5} {:.5} {:.5} ⎦\n",
                self.n_rows,
                self.n_cols,
                self.data.len(),
                self.data.num_bytes(),
                vec_host[0],
                vec_host[1],
                vec_host[2],
                vec_host[3],
                vec_host[4],
                vec_host[5],
            )
        } else if (self.n_rows == 1) & (self.n_cols == 3) {
            let vec_host = match self.to_host() {
                Ok(vec) => vec,
                Err(_) => return write!(f, "Matrix data could not be copied to host for display."),
            };
            write!(
                f,
                "Dimension: rows={}; cols={}; len={} length; bytes={}\n[ {:.5} {:.5} {:.5} ]\n",
                self.n_rows,
                self.n_cols,
                self.data.len(),
                self.data.num_bytes(),
                vec_host[0],
                vec_host[1],
                vec_host[2],
            )
        } else if (self.n_rows == 2) & (self.n_cols == 2) {
            let vec_host = match self.to_host() {
                Ok(vec) => vec,
                Err(_) => return write!(f, "Matrix data could not be copied to host for display."),
            };
            write!(
                f,
                "Dimension: rows={}; cols={}; len={} length; bytes={}\n⎡ {:.5} {:.5} ⎤\n⎣ {:.5} {:.5} ⎦\n",
                self.n_rows,
                self.n_cols,
                self.data.len(),
                self.data.num_bytes(),
                vec_host[0],
                vec_host[1],
                vec_host[2],
                vec_host[3],
            )
        } else if (self.n_rows == 2) & (self.n_cols == 1) {
            let vec_host = match self.to_host() {
                Ok(vec) => vec,
                Err(_) => return write!(f, "Matrix data could not be copied to host for display."),
            };
            write!(
                f,
                "Dimension: rows={}; cols={}; len={} length; bytes={}\n⎡ {:.5} ⎤\n⎣ {:.5} ⎦\n",
                self.n_rows,
                self.n_cols,
                self.data.len(),
                self.data.num_bytes(),
                vec_host[0],
                vec_host[1],
            )
        } else if (self.n_rows == 1) & (self.n_cols == 2) {
            let vec_host = match self.to_host() {
                Ok(vec) => vec,
                Err(_) => return write!(f, "Matrix data could not be copied to host for display."),
            };
            write!(
                f,
                "Dimension: rows={}; cols={}; len={} length; bytes={}\n[ {:.5} {:.5} ]\n",
                self.n_rows,
                self.n_cols,
                self.data.len(),
                self.data.num_bytes(),
                vec_host[0],
                vec_host[1],
            )
        } else {
            // 1 x 1 matrix
            let vec_host = match self.to_host() {
                Ok(vec) => vec,
                Err(_) => return write!(f, "Matrix data could not be copied to host for display."),
            };
            write!(
                f,
                "Dimension: rows={}; cols={}; len={} length; bytes={}\n[ {:.5} ]\n",
                self.n_rows,
                self.n_cols,
                self.data.len(),
                self.data.num_bytes(),
                vec_host[0],
            )
        }
    }
}

/// Matrix errors
#[derive(Debug, PartialEq)]
pub enum MatrixError {
    DimensionMismatch(String),
    TypeMismatch(String),
    OutOfBounds(String),
    OutOfMemory(String),
    CompileError(String),
    OtherError(String),
}

/// Implement Error for MatrixError
impl Error for MatrixError {}

/// Implement std::fmt::Display for MatrixError
impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MatrixError::DimensionMismatch(msg) => write!(f, "Dimension Mismatch in {}", msg),
            MatrixError::TypeMismatch(msg) => write!(f, "Type Mismatch in {}", msg),
            MatrixError::OutOfBounds(msg) => write!(f, "Out of Bounds in {}", msg),
            MatrixError::OutOfMemory(msg) => write!(f, "Out of Memory in {}", msg),
            MatrixError::CompileError(msg) => write!(f, "Compiler Error in {}", msg),
            MatrixError::OtherError(msg) => write!(f, "Other Error in {}", msg),
        }
    }
}

// Implement From trait from Box<dyn Error> to MatrixError
// This is to simplify error handling when using the cudarc error types and our MatrixError
impl From<Box<dyn Error>> for MatrixError {
    fn from(err: Box<dyn Error>) -> MatrixError {
        MatrixError::CompileError(err.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_matrix() -> Result<(), Box<dyn Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let n_rows: usize = 10;
        let n_cols: usize = 50;
        let mut a_host: Vec<f32> = (0..n_rows * n_cols).map(|x| x as f32).collect();
        rand::fill(&mut a_host[..]);
        let a_dev: CudaSlice<f32> = stream.clone_htod(&a_host)?;
        let a_matrix = Matrix::new(a_dev, n_rows, n_cols)?;
        println!("a_matrix\n{}", a_matrix);
        let mut b_host: Vec<f32> = a_host.clone();
        stream.memcpy_dtoh(&a_matrix.data, &mut b_host)?;
        assert_eq!(a_host, b_host);

        let row_indexes: Vec<usize> = (0..n_rows).step_by(2).collect();
        let col_indexes: Vec<usize> = vec![0];
        let b_matrix = a_matrix.slice(&row_indexes, &col_indexes)?;
        println!("b_matrix\n{}", b_matrix);

        let row_indexes: Vec<usize> = vec![0];
        let col_indexes: Vec<usize> = (0..n_cols).step_by(2).collect();
        let c_matrix = a_matrix.slice(&row_indexes, &col_indexes)?;
        println!("c_matrix\n{}", c_matrix);

        let row_indexes: Vec<usize> = (0..3).collect();
        let col_indexes: Vec<usize> = (0..3).collect();
        let d_matrix = a_matrix.slice(&row_indexes, &col_indexes)?;
        println!("d_matrix\n{}", d_matrix);

        let row_indexes: Vec<usize> = (0..3).collect();
        let col_indexes: Vec<usize> = (0..2).collect();
        let e_matrix = a_matrix.slice(&row_indexes, &col_indexes)?;
        println!("e_matrix\n{}", e_matrix);

        let row_indexes: Vec<usize> = (0..3).collect();
        let col_indexes: Vec<usize> = (0..1).collect();
        let f_matrix = a_matrix.slice(&row_indexes, &col_indexes)?;
        println!("f_matrix\n{}", f_matrix);

        let row_indexes: Vec<usize> = (0..2).collect();
        let col_indexes: Vec<usize> = (0..3).collect();
        let g_matrix = a_matrix.slice(&row_indexes, &col_indexes)?;
        println!("g_matrix\n{}", g_matrix);

        let row_indexes: Vec<usize> = (0..1).collect();
        let col_indexes: Vec<usize> = (0..3).collect();
        let h_matrix = a_matrix.slice(&row_indexes, &col_indexes)?;
        println!("h_matrix\n{}", h_matrix);

        let row_indexes: Vec<usize> = (0..2).collect();
        let col_indexes: Vec<usize> = (0..2).collect();
        let i_matrix = a_matrix.slice(&row_indexes, &col_indexes)?;
        println!("i_matrix\n{}", i_matrix);

        let row_indexes: Vec<usize> = (0..2).collect();
        let col_indexes: Vec<usize> = (0..1).collect();
        let j_matrix = a_matrix.slice(&row_indexes, &col_indexes)?;
        println!("j_matrix\n{}", j_matrix);

        let row_indexes: Vec<usize> = (0..1).collect();
        let col_indexes: Vec<usize> = (0..2).collect();
        let k_matrix = a_matrix.slice(&row_indexes, &col_indexes)?;
        println!("k_matrix\n{}", k_matrix);

        let row_indexes: Vec<usize> = (0..1).collect();
        let col_indexes: Vec<usize> = (0..1).collect();
        let l_matrix = a_matrix.slice(&row_indexes, &col_indexes)?;
        println!("l_matrix\n{}", l_matrix);

        let vec_host = a_matrix.to_host()?;
        assert_eq!(a_host, vec_host);

        MatrixError::DimensionMismatch("Test error".to_string());
        MatrixError::TypeMismatch("Test error".to_string());
        MatrixError::OutOfBounds("Test error".to_string());
        MatrixError::OutOfMemory("Test error".to_string());
        MatrixError::CompileError("Test error".to_string());
        MatrixError::OtherError("Test error".to_string());

        Ok(())
    }
}
