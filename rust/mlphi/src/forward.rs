use crate::linalg::matrix::Matrix;
use crate::linalg::mult::{matmul, rowmatmul};
use crate::network::Network;
use cudarc::driver::{CudaContext, CudaSlice};
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use std::error::Error;

impl Network {
    pub fn forwardpass(&mut self) -> Result<(), Box<dyn Error>> {
        // Updates:
        //  1. weights_x_biases_per_layer,
        //  2. activations_per_layer, and
        //  3. predictions.
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
                let d_dev: CudaSlice<f32> = self.stream.clone_htod(&d)?;
                let d_matrix = Matrix::new(d_dev, n_nodes, 1)?;
                let x = rowmatmul(&self.weights_per_layer[i], &d_matrix)?;
                matmul(&x, &self.activations_per_layer[i])?
            } else {
                matmul(&self.weights_per_layer[i], &self.activations_per_layer[i])?
            };
            // assert!((rowmatadd(&weights_x_dropout, &self.biases_per_layer[i])?).n_rows == self.weights_x_biases_per_layer[i].n_rows);
            // assert!((rowmatadd(&weights_x_dropout, &self.biases_per_layer[i])?).n_cols == self.weights_x_biases_per_layer[i].n_cols);
            self.weights_x_biases_per_layer[i] =
                weights_x_dropout.rowmatadd(&self.biases_per_layer[i])?;
            // assert!(self.weights_x_biases_per_layer[i].n_rows == self.activations_per_layer[i + 1].n_rows);
            // assert!(self.weights_x_biases_per_layer[i].n_cols == self.activations_per_layer[i + 1].n_cols);
            self.activations_per_layer[i + 1] = self
                .activation
                .activate(&self.weights_x_biases_per_layer[i])?;

            // let mut a_host =
            //     vec![0.0f32; self.activations_per_layer[i].n_rows * self.activations_per_layer[i].n_cols];
            // self
            //     .stream
            //     .memcpy_dtoh(&self.activations_per_layer[i].data, &mut a_host)?;
            // println!(
            //     "??????? layer {}: [{}, {}, {}, ..., {}]",
            //     i,
            //     a_host[0],
            //     a_host[1],
            //     a_host[2],
            //     a_host[a_host.len() - 1]
            // );
        }
        let n = self.n_hidden_layers;
        let weights_x_dropout = matmul(&self.weights_per_layer[n], &self.activations_per_layer[n])?;

        // let mut a_host =
        //     vec![0.0f32; self.activations_per_layer[n].n_rows * self.activations_per_layer[n].n_cols];
        // self
        //     .stream
        //     .memcpy_dtoh(&self.activations_per_layer[n].data, &mut a_host)?;
        // println!(
        //     "???????: [{}, {}, {}, ..., {}]",
        //     a_host[0],
        //     a_host[1],
        //     a_host[2],
        //     a_host[a_host.len() - 1]
        // );

        self.weights_x_biases_per_layer[n] =
            weights_x_dropout.rowmatadd(&self.biases_per_layer[n])?;
        // assert!(self.predictions.n_rows == self.weights_x_biases_per_layer[n].n_rows);
        // assert!(self.predictions.n_cols == self.weights_x_biases_per_layer[n].n_cols);
        self.predictions = self.weights_x_biases_per_layer[n].clone();

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
            stream,
            input_matrix,
            output_matrix,
            10,
            vec![256; 10],
            vec![0.0f32; 10],
            42,
        )?;
        let i: usize = network.n_hidden_layers - 1;
        let mut a_host = vec![
            0.0f32;
            network.activations_per_layer[i].n_rows
                * network.activations_per_layer[i].n_cols
        ];
        network
            .stream
            .memcpy_dtoh(&network.activations_per_layer[i].data, &mut a_host)?;
        println!(
            "layer {} activations (before forward pass): [{}, {}, {}, ..., {}]",
            i,
            a_host[0],
            a_host[1],
            a_host[2],
            a_host[a_host.len() - 1]
        );
        // Forward pass
        network.forwardpass()?;
        network
            .stream
            .memcpy_dtoh(&network.activations_per_layer[i].data, &mut a_host)?;
        println!(
            "layer {} activations (after forward pass with random weights between 0 and 1): [{}, {}, {}, ..., {}]",
            i,
            a_host[0],
            a_host[1],
            a_host[2],
            a_host[a_host.len() - 1]
        );
        let sum_1 = a_host.iter().fold(0.0, |sum, x| sum + x);
        // Modify weights (set all weights to 1.00 instead of random values between 0 and 1) and forward pass
        let b_host = vec![
            1.0f32;
            network.weights_per_layer[i - 1].n_rows
                * network.weights_per_layer[i - 1].n_cols
        ];
        let b_dev = network.stream.clone_htod(&b_host)?;
        let b_matrix = Matrix::new(
            b_dev,
            network.weights_per_layer[i - 1].n_rows,
            network.weights_per_layer[i - 1].n_cols,
        )?;
        network.weights_per_layer[i - 1] = b_matrix;
        network.forwardpass()?;
        network
            .stream
            .memcpy_dtoh(&network.activations_per_layer[i].data, &mut a_host)?;
        println!(
            "layer {} activations (after forward pass with weights all set to 1): [{}, {}, {}, ..., {}]",
            i,
            a_host[0],
            a_host[1],
            a_host[2],
            a_host[a_host.len() - 1]
        );
        let sum_2 = a_host.iter().fold(0.0, |sum, x| sum + x);
        println!("sum_1={}; sum_2={}", sum_1, sum_2);
        // We expect the sum of activations using weights between 0 and 1 is lower the that using weights all set to 1.
        assert!(sum_1 < sum_2);
        Ok(())
    }
}
