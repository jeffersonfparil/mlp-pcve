use std::error::Error;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use cudarc::driver::{CudaContext, CudaSlice};
use crate::linalg::matrix::Matrix;
use crate::linalg::add::rowmatadd;
use crate::linalg::mult::{rowmatmul, matmul};
use crate::network::Network;

impl Network{
    pub fn forwardpass(
        &mut self,
    ) -> Result<(), Box<dyn Error>> {
        let mut rng = ChaCha12Rng::seed_from_u64(self.seed as u64);
        for i in 0..self.n_hidden_layers {
            let n_nodes = self.n_hidden_nodes[i];
            let n_dropped_nodes = (self.dropout_rates[i] * self.n_hidden_nodes[i] as f32).round() as usize;
            let idx_dropped_nodes = (0..n_nodes).choose_multiple(&mut rng, n_dropped_nodes);
            let mut d = vec![1.0f32; n_nodes];
            for i in idx_dropped_nodes {
                d[i] = 0.0;
            }
            let d_dev: CudaSlice<f32> = self.stream.clone_htod(&d)?;
            let d_matrix = Matrix::new(d_dev, n_nodes, 1)?;
            let x = rowmatmul(&self.weights_per_layer[i], &d_matrix)?;
            let y = matmul(&x, &self.activations_per_layer[i])?;
            let z = rowmatadd(&y, &self.biases_per_layer[i])?;
            let a = self.activation.activate(&z)?;
            self.weights_x_biases_per_layer[i] = z;
            self.activations_per_layer[i+1] = a;
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
        let mut input_host: Vec<f32> = vec![0.0f32; p*n]; // p x n
        let mut output_host: Vec<f32> = vec![0.0f32; k*n]; // k x n
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
        let i: usize = 1;
        let mut a_host = vec![0.0f32; network.activations_per_layer[i].n_rows * network.activations_per_layer[i].n_cols];
        network.stream.memcpy_dtoh(&network.activations_per_layer[i].data, &mut a_host)?;
        println!("weights_0 (before forward pass): [{}, {}, {}, ..., {}]", a_host[0], a_host[1], a_host[2], a_host[a_host.len()-1]);
        // Forward pass
        network.forwardpass()?;
        network.stream.memcpy_dtoh(&network.activations_per_layer[i].data, &mut a_host)?;
        println!("weights_0 (after forward pass): [{}, {}, {}, ..., {}]", a_host[0], a_host[1], a_host[2], a_host[a_host.len()-1]);
        let sum_1 = a_host.iter().fold(0.0, |sum, x| sum + x);
        // Modify weights and forward pass
        let b_host = vec![1.0f32; network.weights_per_layer[i-1].n_rows * network.weights_per_layer[i-1].n_cols];
        let b_dev = network.stream.clone_htod(&b_host)?;
        let b_matrix = Matrix::new(b_dev, network.weights_per_layer[i-1].n_rows, network.weights_per_layer[i-1].n_cols)?;
        network.weights_per_layer[i-1] = b_matrix;
        network.forwardpass()?;
        network.stream.memcpy_dtoh(&network.activations_per_layer[i].data, &mut a_host)?;
        println!("weights_0 (after forward pass): [{}, {}, {}, ..., {}]", a_host[0], a_host[1], a_host[2], a_host[a_host.len()-1]);
        let sum_2 = a_host.iter().fold(0.0, |sum, x| sum + x);
        println!("sum_1={}; sum_2={}", sum_1, sum_2);
        assert!(sum_1 < sum_2);
        Ok(())
    }
}
