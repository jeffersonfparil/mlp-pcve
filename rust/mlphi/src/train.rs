use crate::linalg::matrix::{Matrix, MatrixError};
use crate::network::Network;
use crate::optimisers::{OptimisationParameters, Optimiser};
use chrono::Utc;
use cudarc::driver::CudaSlice;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use ruviz::core::{Plot, PlottingError};
use ruviz::prelude::LegendPosition;
use std::env;
use std::error::Error;
impl From<PlottingError> for MatrixError {
    fn from(err: PlottingError) -> Self {
        MatrixError::CompileError(err.to_string())
    }
}

impl Network {
    pub fn shufflesplit(self: &Self, n_batches: usize) -> Result<Vec<Vec<usize>>, Box<dyn Error>> {
        let n: usize = self.targets.n_cols; // number of observations
        if n_batches == 0 {
            return Err(Box::new(MatrixError::OtherError(
                "Number of batches must be greater than zero.".to_string(),
            )));
        }
        if n_batches > n {
            return Err(Box::new(MatrixError::OtherError(
                "Number of batches cannot be greater than number of observations.".to_string(),
            )));
        }
        let mut rng = ChaCha12Rng::seed_from_u64(self.seed as u64);
        let mut indexes: Vec<usize> = (0..n).collect();
        indexes.shuffle(&mut rng);
        let batch_size = (n + n_batches - 1) / n_batches;
        let mut col_indexes_per_batch: Vec<Vec<usize>> = Vec::new();
        for i in 0..n_batches {
            let start = i * batch_size;
            let end = match (i + 1) * batch_size {
                x if x > n => n,
                x => x,
            };
            if start >= end {
                break;
            }
            col_indexes_per_batch.push(indexes[start..end].to_vec());
        }
        Ok(col_indexes_per_batch)
    }

    pub fn predict(self: &mut Self) -> Result<(), Box<dyn Error>> {
        // Different from forwardpass in that it does not apply dropout.
        let n = self.n_hidden_layers;
        for i in 0..n {
            let weights_x_activations =
                self.weights_per_layer[i].matmul(&self.activations_per_layer[i])?;
            self.weights_x_biases_per_layer[i] =
                weights_x_activations.rowmatadd(&self.biases_per_layer[i])?;
            self.activations_per_layer[i + 1] = self
                .activation
                .activate(&self.weights_x_biases_per_layer[i])?;
        }
        let weights_x_dropout = self.weights_per_layer[n].matmul(&self.activations_per_layer[n])?;
        self.weights_x_biases_per_layer[n] =
            weights_x_dropout.rowmatadd(&self.biases_per_layer[n])?;
        self.predictions = self.weights_x_biases_per_layer[n].clone();
        Ok(())
    }

    pub fn train_per_batch(
        self: &mut Self,
        optimisation_parameters: &mut OptimisationParameters,
    ) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
        let mut epochs: Vec<f64> = Vec::new();
        let mut costs: Vec<f64> = Vec::new();
        for epoch in 0..optimisation_parameters.n_epochs {
            self.forwardpass()?;
            self.backpropagation()?;
            self.optimise(optimisation_parameters)?;
            self.predict()?;
            let e: f64 = epoch as f64;
            let c: f64 = (self
                .cost
                .cost(&self.predictions, &self.targets)?
                .summat(&self.targets.data.context().default_stream())? as f64)
                / (self.targets.n_cols as f64);
            epochs.push(e);
            costs.push(c);
        }
        Ok((epochs, costs))
    }

    pub fn train(
        self: &mut Self,
        optimisation_parameters: &mut OptimisationParameters,
        verbose: bool,
    ) -> Result<(), Box<dyn Error>> {
        if optimisation_parameters.n_batches == 0 {
            return Err(Box::new(MatrixError::OtherError(
                "Number of batches must be greater than zero.".to_string(),
            )));
        }
        let stream = self.targets.data.context().default_stream();
        let (epochs, costs): (Vec<Vec<f64>>, Vec<Vec<f64>>) = if optimisation_parameters.n_batches
            == 1
        {
            // Only one batch, train on the whole dataset
            let (epochs, costs) = self.train_per_batch(optimisation_parameters)?; // performs the prediction internally after every epoch, hence no need to call self.predict() again
            (vec![epochs], vec![costs])
        } else {
            // Multiple batches, split the dataset then average the parameters after training on each batch
            let col_indexes_per_batch: Vec<Vec<usize>> =
                self.shufflesplit(optimisation_parameters.n_batches)?;
            let mut networks_per_batch: Vec<Network> =
                Vec::with_capacity(optimisation_parameters.n_batches);
            for col_indexes in col_indexes_per_batch {
                // indexes for each batch, i.e. for observations
                let network = self.slice(&col_indexes)?;
                networks_per_batch.push(network);
            }
            let mut epochs: Vec<Vec<f64>> = Vec::new();
            let mut costs: Vec<Vec<f64>> = Vec::new();
            for (i, network) in networks_per_batch.iter_mut().enumerate() {
                if verbose {
                    println!(
                        "Training on batch {} with {} observations.",
                        i, network.targets.n_cols
                    );
                }
                let (epochs_batch, costs_batch) =
                    network.train_per_batch(optimisation_parameters)?;
                epochs.push(epochs_batch.clone());
                costs.push(costs_batch.clone());
            }
            // Merge the parameters from each batch network back into the original network
            // TODO: update this simple averaging with a better method
            for i in 0..self.n_hidden_layers + 1 {
                let zeros_weights_host: Vec<f32> =
                    vec![0.0; self.weights_per_layer[i].n_rows * self.weights_per_layer[i].n_cols];
                let zeros_biases_host: Vec<f32> =
                    vec![0.0; self.biases_per_layer[i].n_rows * self.biases_per_layer[i].n_cols];
                let zeros_weights_dev: CudaSlice<f32> =
                    stream.clone_htod(&zeros_weights_host)?;
                let zeros_biases_dev: CudaSlice<f32> =
                    stream.clone_htod(&zeros_biases_host)?;
                let mut summed_weights = Matrix::new(
                    zeros_weights_dev,
                    self.weights_per_layer[i].n_rows,
                    self.weights_per_layer[i].n_cols,
                )?;
                let mut summed_biases = Matrix::new(
                    zeros_biases_dev,
                    self.biases_per_layer[i].n_rows,
                    self.biases_per_layer[i].n_cols,
                )?;
                for network in &networks_per_batch {
                    summed_weights =
                        summed_weights.elementwisematadd(&network.weights_per_layer[i])?;
                    summed_biases =
                        summed_biases.elementwisematadd(&network.biases_per_layer[i])?;
                }
                self.weights_per_layer[i] =
                    summed_weights.scalarmatmul(1.00 / networks_per_batch.len() as f32)?;
                self.biases_per_layer[i] =
                    summed_biases.scalarmatmul(1.00 / networks_per_batch.len() as f32)?;
            }
            // Update predictions using the merged parameters
            self.predict()?;
            // Return epochs, costs
            (epochs, costs)
        };
        if verbose {
            // Assess cost after training
            let final_cost = self.cost.cost(&self.predictions, &self.targets)?;
            let final_cost_value =
                final_cost.summat(&stream)? as f32 / self.targets.n_cols as f32;
            // Plot loss curve
            let dir: std::path::PathBuf = env::current_dir()?;
            let fname_svg = &format!(
                "{}/Loss_curve-{:?}-Time{}.svg",
                dir.display(),
                optimisation_parameters.optimiser,
                Utc::now().format("%Y%m%d%H%M%S")
            );
            let mut ylabel = String::from("Cost");
            ylabel.push_str(&format!(
                " ({:?}; {:?})",
                self.cost, optimisation_parameters.optimiser
            ));
            let mut plot_vec = vec![
                Plot::new()
                    .title("Training Cost over Epochs")
                    .legend_position(LegendPosition::Best)
                    .xlabel("Epochs")
                    .ylabel(&ylabel)
                    .line(&epochs[0], &costs[0])
                    .label("Batch 0")
                    .size(4.0, 3.0),
            ];
            for i in 1..epochs.len() {
                plot_vec[0] = plot_vec[0]
                    .clone()
                    .line(&epochs[i], &costs[i])
                    .label(&format!("Batch {}", i));
            }
            // plot_vec[0].clone().save(fname_png)?;
            plot_vec[0].clone().export_svg(fname_svg)?;
            // Messages
            println!("Final cost after training: {}", final_cost_value);
            println!("Find the loss curve saved as: {}", fname_svg);
        }
        Ok(())
    }
    // pub fn optim(self: &Self) -> Result<(), Box<dyn Error>> {
    //     Ok(())
    // }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::{CudaContext, CudaSlice};
    #[test]
    fn test_train() -> Result<(), Box<dyn Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let n: usize = 3_000_123; // number of observations
        let p: usize = 314; // number of input features
        let k: usize = 1; // number of output features
        let n_hidden_layers: usize = 2;
        let n_hidden_layer_nodes: usize = 5;
        let mut input_host: Vec<f32> = vec![0.0f32; p * n]; // p x n
        let mut output_host: Vec<f32> = vec![0.0f32; k * n]; // k x n
        rand::fill(&mut input_host[..]);
        rand::fill(&mut output_host[..]);
        let input_dev: CudaSlice<f32> = stream.clone_htod(&input_host)?;
        let output_dev: CudaSlice<f32> = stream.clone_htod(&output_host)?;
        let input_matrix = Matrix::new(input_dev, p, n)?; // p x n matrix
        println!("input_matrix: {}", input_matrix);
        let output_matrix = Matrix::new(output_dev, k, n)?; // k x n matrix
        println!("output_matrix: {}", output_matrix);
        let mut network: Network = Network::new(
            &stream,
            input_matrix,
            output_matrix,
            n_hidden_layers,
            vec![n_hidden_layer_nodes; n_hidden_layers],
            vec![0.0f32; n_hidden_layers],
            42,
        )?;
        let mut optimisation_parameters = OptimisationParameters::new(&network)?;
        println!("Network:\n{}\n\n", network);
        println!("Optimisation Parameters:\n{}\n\n", optimisation_parameters);
        // optimisation_parameters.learning_rate = 0.00001f32;
        // optimisation_parameters.optimiser = Optimiser::GradientDescent;
        optimisation_parameters.optimiser = Optimiser::Adam;
        // optimisation_parameters.optimiser = Optimiser::AdamMax;

        // Tests

        let indexes: Vec<Vec<usize>> = network.shufflesplit(5)?;
        // println!("indexes: {:?}", indexes);
        println!("Number of batches: {:?}", indexes.len());
        let mut total_len: usize = 0;
        for i in 0..indexes.len() {
            println!(
                "indexes[{}]: [{}, {}, ...{}] length: {:?}",
                i,
                indexes[i][0],
                indexes[i][1],
                indexes[i][indexes[i].len() - 1],
                indexes[i].len()
            );
            total_len += indexes[i].len();
        }
        println!("Total length: {:?}", total_len);
        assert!(total_len == network.targets.n_cols);

        let stream = network.targets.data.context().default_stream();
        let mut a_host = vec![0.0f32; network.targets.n_rows * network.targets.n_cols];
        stream.memcpy_dtoh(&network.targets.data, &mut a_host)?;
        println!(
            "targets: [{}, {}, {}, ..., {}]",
            a_host[0],
            a_host[1],
            a_host[2],
            a_host[a_host.len() - 1]
        );

        stream.memcpy_dtoh(&network.predictions.data, &mut a_host)?;
        println!(
            "predictions (before predict()): [{}, {}, {}, ..., {}]",
            a_host[0],
            a_host[1],
            a_host[2],
            a_host[a_host.len() - 1]
        );
        network.train_per_batch(&mut optimisation_parameters)?;
        optimisation_parameters.n_epochs = 20;
        optimisation_parameters.n_batches = 2;
        network.train(&mut optimisation_parameters, true)?;

        Ok(())
    }
}
