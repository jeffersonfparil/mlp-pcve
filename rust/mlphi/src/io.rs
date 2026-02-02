use crate::linalg::matrix::{Matrix, MatrixError};
use crate::network::Network;
use cudarc::driver::{CudaContext, CudaSlice};
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use rand_distr::Normal;
// use std::path::PathBuf;
// use std::env::current_dir;
use std::fmt;
use std::fs::File;
use std::error::Error;
use std::io::{BufWriter, BufReader};
use std::io::{Write, BufRead};

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Data {
    pub features: Matrix, // p x n: p features, n samples
    pub targets: Matrix,  // k x n: k targets, n samples
    pub feature_names: Vec<String>,
    pub target_names: Vec<String>,
}

/// Implement std::fmt::Display for MatrixError
impl fmt::Display for Data {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Data {{\n  features: {},\n  targets: {}\n}}", self.features, self.targets)
    }
}


impl Data {
    pub fn new(n: usize, p: usize, k: usize) -> Result<Self, Box<dyn Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let features_dev: CudaSlice<f32> = stream.clone_htod(&vec![0.0f32; p * n])?;
        let targets_dev: CudaSlice<f32> = stream.clone_htod(&vec![0.0f32; k * n])?;
        let features = Matrix::new(features_dev, p, n)?;
        let targets = Matrix::new(targets_dev, k, n)?;
        let feature_names: Vec<String> = (0..p).map(|i| format!("feature_{}", i)).collect();
        let target_names: Vec<String> = (0..k).map(|i| format!("target_{}", i)).collect();
        Ok(Data { features, targets, feature_names, target_names })
    }
    
    pub fn simulate(
        n: usize,
        p: usize,
        k: usize,
        d: usize,
        seed: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let mut data = Data::new(n, p, k)?;
        let stream = data.features.data.context().default_stream();
        let mut rng = ChaCha12Rng::seed_from_u64(seed as u64);
        let normal = Normal::new(0.0, 1.0)?;
        let features = {
            let features_host: Vec<f32> = (&mut rng).sample_iter(normal).take(p * n).collect();
            let features_dev: CudaSlice<f32> = stream.clone_htod(&features_host)?;
            Matrix::new(features_dev, p, n)?
        };
        let targets = {
            let targets_host: Vec<f32> = (&mut rng).sample_iter(normal).take(k * n).collect();
            let targets_dev: CudaSlice<f32> = stream.clone_htod(&targets_host)?;
            Matrix::new(targets_dev, k, n)?
        };
        let n_hidden_layers: usize = d;
        let n_hidden_nodes: Vec<usize> = vec![(p as f64 / 2.0).ceil() as usize; n_hidden_layers];
        let dropout_rates: Vec<f32> = vec![0.0; n_hidden_layers];
        let mut network = Network::new(
            &stream,
            features,
            targets,
            n_hidden_layers,
            n_hidden_nodes,
            dropout_rates,
            seed,
        )?;
        network.forwardpass()?;
        data.features = network.activations_per_layer[0].clone();
        data.targets = network.predictions.clone();
        Ok(data)
    }
    
    pub fn check_dimensions(&self) -> Result<(), MatrixError> {
        if self.features.n_cols != self.targets.n_cols {
            return Err(MatrixError::DimensionMismatch(format!(
                "Number of observations (n_cols) in features ({}) does not match number of observations (n_cols) in targets ({}).",
                self.features.n_cols, self.targets.n_cols
            )));
        }
        if self.features.n_rows == 0 {
            return Err(MatrixError::DimensionMismatch(format!(
                "Number of features (n_rows) is zero."
            )));
        }
        if self.targets.n_rows == 0 {
            return Err(MatrixError::DimensionMismatch(format!(
                "Number of target variable/s (n_rows) is zero."
            )));
        }
        if self.feature_names.len() != self.features.n_rows {
            return Err(MatrixError::DimensionMismatch(format!(
                "Number of feature names ({}) does not match number of features ({}).",
                self.feature_names.len(),
                self.features.n_rows
            )));
        }
        if self.target_names.len() != self.targets.n_rows {
            return Err(MatrixError::DimensionMismatch(format!(
                "Number of target names ({}) does not match number of target variable/s ({}).",
                self.target_names.len(),
                self.targets.n_rows
            )));
        }
        Ok(())
    }

    pub fn write_delimited(&self, path: &str, delim: &str) -> Result<(), Box<dyn Error>> {
        self.check_dimensions()?;
        let features = self.features.to_host()?;
        let targets = self.targets.to_host()?;
        let file = File::create_new(path)?; // makes sure not to overwrite existing files, i.e. using create_new() instead of just create()
        let mut writer = BufWriter::new(file);
        let n = self.features.n_cols;
        let p = self.features.n_rows;
        let k = self.targets.n_rows;
        // Write header
        let mut header: Vec<String> = Vec::with_capacity(p + k);
        for t in &self.target_names {
            header.push(t.clone());
        }
        for f in &self.feature_names {
            header.push(f.clone());
        }
        writeln!(writer, "{}", header.join(delim))?;
        // Write data
        for i in 0..n {
            let mut row: Vec<String> = Vec::with_capacity(p + k);
            for j in 0..k {
                row.push(format!("{}", targets[(j*n)+i]));
            }
            for j in 0..p {
                row.push(format!("{}", features[(j*n)+i]));
            }
            writeln!(writer, "{}", row.join(delim))?;
        }
        Ok(())
    }

    pub fn read_delimited(path: &str, delim:&str, column_indices_of_targets: Vec<usize>) -> Result<Self, Box<dyn Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        // Read header
        let header: Vec<String> = if let Some(header_line) = lines.next() {
            let header = header_line?;
            header.trim().split(delim).map(|s| s.to_string()).collect()
        } else {
            return Err(Box::new(MatrixError::DimensionMismatch(
                "File is empty.".to_string(),
            )))
        };
        if column_indices_of_targets.is_empty() {
            return Err(Box::new(MatrixError::DimensionMismatch(
                "No column indices of targets provided.".to_string(),
            )));
        }
        if column_indices_of_targets.iter().any(|&idx| idx >= header.len()) {
            return Err(Box::new(MatrixError::DimensionMismatch(
                "One or more column indices of targets are out of bounds.".to_string(),
            )));
        }
        let column_indices_features: Vec<usize> = (0..header.len())
            .filter(|idx| !column_indices_of_targets.contains(idx))
            .collect();
        let target_names: Vec<String> = column_indices_of_targets
            .iter()
            .map(|&idx| header[idx].clone())
            .collect();
        let feature_names: Vec<String> = column_indices_features
            .iter()
            .map(|&idx| header[idx].clone())
            .collect();
        let mut features_data: Vec<f32> = Vec::new();
        let mut targets_data: Vec<f32> = Vec::new();
        for line in lines {
            let line = line?;
            let values: Vec<&str> = line.trim().split(delim).collect();
            if values.len() != header.len() {
                return Err(Box::new(MatrixError::DimensionMismatch(
                    "Number of values in a row does not match number of columns in header.".to_string(),
                )));
            }
            for &idx in &column_indices_of_targets {
                let value: f32 = values[idx].parse()?;
                targets_data.push(value);
            }
            for &idx in &column_indices_features {
                let value: f32 = values[idx].parse()?;
                features_data.push(value);
            }
        }
        let n = targets_data.len() / column_indices_of_targets.len();
        let p = feature_names.len();
        let k = target_names.len();
        let mut data = Data::new(n, p, k)?;
        let stream = data.features.data.context().default_stream();
        let features_dev: CudaSlice<f32> = stream.clone_htod(&features_data)?;
        let targets_dev: CudaSlice<f32> = stream.clone_htod(&targets_data)?;
        data.features = Matrix::new(features_dev, p, n)?;
        data.targets = Matrix::new(targets_dev, k, n)?;
        data.feature_names = feature_names;
        data.target_names = target_names;
        Ok(data)
    }

    pub fn init_network(
        &self,
        n_hidden_layers: usize,
        n_hidden_nodes: Vec<usize>,
        dropout_rates: Vec<f32>,
        seed: usize,
    ) -> Result<Network, Box<dyn Error>> {
        self.check_dimensions()?;
        let stream = self.features.data.context().default_stream();
        let network = Network::new(
            &stream,
            self.features.clone(),
            self.targets.clone(),
            n_hidden_layers,
            n_hidden_nodes,
            dropout_rates,
            seed,
        )?;
        Ok(network)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs::{exists, remove_file};
    #[test]
    fn test_io() -> Result<(), Box<dyn Error>> {
        let data = Data::new(100, 10, 1)?;
        let data_simulated = Data::simulate(100, 10, 1, 2, 42)?;
        assert_eq!(data.features.n_rows, data_simulated.features.n_rows);
        assert!(data.targets.summat()? == 0.0);
        assert!(data_simulated.targets.summat()? != 0.0);
        assert!(data.features.summat()? == 0.0);
        assert!(data_simulated.features.summat()? != 0.0);

        assert_eq!(data.check_dimensions(), Ok(()));

        println!("data: {}", data);
        println!("data_simulated: {}", data_simulated);

        if exists("test_data.csv")? {
            remove_file("test_data.csv")?;
        }
        if exists("test_data_simulated.tsv")? {
            remove_file("test_data_simulated.tsv")?;
        }
        data.write_delimited("test_data.csv", ",")?;
        data_simulated.write_delimited("test_data_simulated.tsv", "\t")?;

        let data_reloaded = Data::read_delimited("test_data.csv", ",", vec![0])?;
        let data_simulated_reloaded = Data::read_delimited("test_data_simulated.tsv", "\t", vec![0])?;

        assert!(data.features.summat()? - data_reloaded.features.summat()? < 1e-5);
        assert!(data_simulated.features.summat()? - data_simulated_reloaded.features.summat()? < 1e-5);

        println!("data_reloaded: {}", data_reloaded);
        println!("data_simulated_reloaded: {}", data_simulated_reloaded);

        // Initialise the network from reloaded data
        let network = data_simulated_reloaded.init_network(
            2,
            vec![5; 2],
            vec![0.0; 2],
            42,
        )?;
        assert!(network.targets.summat()? - data_simulated_reloaded.targets.summat()? < 1e-5);
        assert!(network.activations_per_layer[0].summat()? - data_simulated_reloaded.features.summat()? < 1e-5);
        assert_eq!(network.n_hidden_layers, 2);
        println!("network: {}", network);


        // Clean-up
        remove_file("test_data.csv")?;
        remove_file("test_data_simulated.tsv")?;

        Ok(())
    }
}
