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
                "Data length {} does not match matrix dimensions {}x{}={}",
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
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{} rows x {} columns ({} length; {} bytes)",
            self.n_rows,
            self.n_cols,
            self.data.len(),
            self.data.num_bytes(),
        )
    }
}

/// Matrix errors
#[derive(Debug)]
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
            MatrixError::DimensionMismatch(msg) => write!(f, "Dimension Mismatch: {}", msg),
            MatrixError::TypeMismatch(msg) => write!(f, "Type Mismatch: {}", msg),
            MatrixError::OutOfBounds(msg) => write!(f, "Out of Bounds: {}", msg),
            MatrixError::OutOfMemory(msg) => write!(f, "Out of Memory: {}", msg),
            MatrixError::CompileError(msg) => write!(f, "Compiler Error: {}", msg),
            MatrixError::OtherError(msg) => write!(f, "Other Error: {}", msg),
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
        let mut a_host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        println!("a_host {:?}", a_host);

        rand::fill(&mut a_host[..]);
        println!("a_host {:?}", a_host);

        let a_dev: CudaSlice<f32> = stream.clone_htod(&a_host)?;
        println!("a_dev {:?}", a_dev);
        println!("a_dev.len() {:?}", a_dev.len());
        let a_matrix = Matrix::new(a_dev, 2, 3)?;

        // println!("a_dev {:?}", a_dev); // error because a_dev is now owned by a_matrix

        let mut b_host: Vec<f32> = a_host.clone();
        stream.memcpy_dtoh(&a_matrix.data, &mut b_host)?;
        println!("b_host {:?}", b_host);

        MatrixError::DimensionMismatch("Test error".to_string());
        MatrixError::TypeMismatch("Test error".to_string());
        MatrixError::OutOfBounds("Test error".to_string());
        MatrixError::OutOfMemory("Test error".to_string());
        MatrixError::CompileError("Test error".to_string());

        Ok(())
    }
}
