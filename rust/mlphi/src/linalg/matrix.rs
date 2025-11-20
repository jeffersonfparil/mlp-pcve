use std::sync::Arc;
use cudarc::driver::{DriverError, CudaContext, CudaSlice, CudaStream};


pub struct MatrixF32 {
    pub n_rows: usize,
    pub n_cols: usize,
    pub data: CudaSlice<f32>,
}

impl MatrixF32 {
    pub fn new(
        stream: Arc<CudaStream>, 
        data: &Vec<f32>, 
        n_rows: usize, 
        n_cols: usize
    ) -> Result<Self, DriverError> {
        let d = stream.clone_htod(data)?;
        let out = Self {
            n_rows: n_rows,
            n_cols: n_cols,
            data: d,
        };
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_matrix_new() -> Result<(), Box<dyn std::error::Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let a_host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a_matrix = MatrixF32::new(stream, &a_host, 2, 3);

        // stream.memcpy_dtoh(&a_matrix, &mut a_host)?;
        println!("a_host {:?}", a_host);
        
        Ok(())
    }
}