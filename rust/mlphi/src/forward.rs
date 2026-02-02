use crate::linalg::matrix::Matrix;
use crate::network::Network;
use cudarc::driver::{CudaContext, CudaSlice};
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use std::error::Error;

// Notes:
// Matrices are in shape (n_rows, n_cols) = (nodes, samples), i.e.:
//      - activations_per_layer[i]: activations of layer i, shape (n_hidden_nodes[i-1], n_samples).
//      - weights_per_layer[i]: weights connecting layer i to layer i+1, shape (n_hidden_nodes[i], n_hidden_nodes[i-1]).
//      - biases_per_layer[i]: biases of layer i+1, shape (n_hidden_nodes[i], 1).
//      - weights_x_biases_per_layer[i]: pre-activation values of layer i+1, shape (n_hidden_nodes[i], n_samples).

impl Network {
    pub fn forwardpass(&mut self) -> Result<(), Box<dyn Error>> {
        // Updates:
        //  1. weights_x_biases_per_layer, and
        //  2. activations_per_layer. Predictions are not updated here as they are only
        //     updated in the predict() method (no dropout there).
        let mut rng = ChaCha12Rng::seed_from_u64(self.seed as u64);
        for i in 0..self.n_hidden_layers {
            let n_nodes = self.n_hidden_nodes[i];
            let n_dropped_nodes =
                (self.dropout_rates[i] * self.n_hidden_nodes[i] as f32).round() as usize;
            let weights_x_dropout = if n_dropped_nodes > 0 {
                let idx_dropped_nodes = (0..n_nodes).choose_multiple(&mut rng, n_dropped_nodes);
                let mut d = vec![1.0f32; n_nodes];
                for i in idx_dropped_nodes {
                    d[i] = 0.0;
                }
                let d_dev: CudaSlice<f32> = self
                    .targets
                    .data
                    .context()
                    .default_stream()
                    .clone_htod(&d)?;
                let d_matrix = Matrix::new(d_dev, n_nodes, 1)?;
                let x = self.weights_per_layer[i].rowmatmul(&d_matrix)?;
                x.matmul(&self.activations_per_layer[i])?
            } else {
                self.weights_per_layer[i].matmul(&self.activations_per_layer[i])?
            };
            self.weights_x_biases_per_layer[i] =
                weights_x_dropout.rowmatadd(&self.biases_per_layer[i])?;
            self.activations_per_layer[i + 1] = self
                .activation
                .activate(&self.weights_x_biases_per_layer[i])?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_forward() -> Result<(), Box<dyn Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let n: usize = 100;
        let p: usize = 17;
        let k: usize = 1;
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
            10,
            vec![256; 10],
            vec![0.0f32; 10],
            42,
        )?;
        let i: usize = network.n_hidden_layers - 1;
        println!(
            "layer {} activations (before forward pass):\n{}",
            i, network.activations_per_layer[i],
        );
        // Forward pass
        network.forwardpass()?;
        println!(
            "layer {} activations (after forward pass):\n{}",
            i, network.activations_per_layer[i],
        );
        let sum_1: f32 = network.activations_per_layer[i].summat()?;
        // Modify weights (set all weights to 1.00 instead of random values between 0 and 1) and forward pass
        let b_host = vec![
            1.0f32;
            network.weights_per_layer[i - 1].n_rows
                * network.weights_per_layer[i - 1].n_cols
        ];
        let b_dev = stream.clone_htod(&b_host)?;
        let b_matrix = Matrix::new(
            b_dev,
            network.weights_per_layer[i - 1].n_rows,
            network.weights_per_layer[i - 1].n_cols,
        )?;
        network.weights_per_layer[i - 1] = b_matrix;
        network.forwardpass()?;
        println!(
            "layer {} activations (after forward pass with weights all set to 1):\n{}",
            i, network.activations_per_layer[i],
        );
        let sum_2: f32 = network.activations_per_layer[i].summat()?;
        // We expect the sum of activations using weights between 0 and 1 is lower the that using weights all set to 1.
        assert!(sum_1 < sum_2);
        Ok(())
    }
}
