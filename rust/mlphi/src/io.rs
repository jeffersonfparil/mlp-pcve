use crate::linalg::matrix::{Matrix, MatrixError};
use std;
use std::env;
use std::error::Error;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use cudarc::driver::{CudaSlice, CudaContext, CudaStream};
use rand_distr::{Normal, Distribution};

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Data {
    pub features: Matrix,
    pub targets: Matrix,
}

impl Data {
    pub fn simulate(n: usize, p: usize, k: usize, seed: usize) -> Result<Self, Box<dyn Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let mut rng = ChaCha12Rng::seed_from_u64(seed as u64);
        let normal = Normal::new(0.0, 1.0)?;
        let features = {
            let features_host: Vec<f32> = (&mut rng).sample_iter(normal).take(n * p).collect();
            let features_dev: CudaSlice<f32> = stream.clone_htod(&features_host)?;
            Matrix::new(features_dev, n, p)?
        };
        let targets = {
            let targets_host: Vec<f32> = (&mut rng).sample_iter(normal).take(n * k).collect();
            let targets_dev: CudaSlice<f32> = stream.clone_htod(&targets_host)?;
            Matrix::new(targets_dev, n, k)?
        };
        Ok(
            Data { features, targets }
        )
    }
    pub fn check_dimensions(&self) -> Result<(), MatrixError> {
        if self.features.n_rows != self.targets.n_rows {
            return Err(MatrixError::DimensionMismatch(format!(
                "Number of rows in features ({}) does not match number of rows in targets ({}).",
                self.features.n_rows, self.targets.n_rows
            )));
        }
        if self.features.n_cols == 0 {
            return Err(MatrixError::DimensionMismatch(format!(
                "Number of columns in features is zero."
            )));
        }
        if self.targets.n_cols == 0 {
            return Err(MatrixError::DimensionMismatch(format!(
                "Number of columns in targets is zero."
            )));
        }
        Ok(())
    }
    pub fn write_csv(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let features = self.features.to_host()?;
        let targets = self.targets.to_host()?;
        let file = std::fs::File::create(path)?;
        let mut writer = std::io::BufWriter::new(file);
        Ok(())
    }
}