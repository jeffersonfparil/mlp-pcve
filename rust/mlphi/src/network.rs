use std::error::Error;
use std::sync::Arc;
use std::fmt;
use cudarc::driver::{CudaContext, CudaStream, CudaSlice};
use crate::linalg::matrix::{Matrix, MatrixError};
use crate::linalg::activations;
use crate::linalg::costs;
// use rand::{Rng, SeedableRng, rngs::StdRng};

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Network {
    pub stream: Arc<CudaStream>, // CUDA device stream
    pub n_hidden_layers: usize, // number of hidden layers
    pub n_hidden_nodes: Vec<usize>, // number of nodes per hidden layer (k)
    pub dropout_rates: Vec<f32>, // soft dropout rates per hidden layer (k)
    pub targets: Matrix, // observed values (n x 1)
    pub predictions: Matrix, // predictions (n x 1)
    pub weights_per_layer: Vec<Matrix>, // weights ((n_hidden_nodes[i+1] x n_hidden_nodes[i]) for i in 0:(k-1))
    pub biases_per_layer: Vec<Matrix>, // biases ((n_hidden_nodes[i+1] x 1) for i in 0:(k-1))
    pub weights_x_biases_per_layer: Vec<Matrix>, // summed weights (i.e. prior to activation function) ((n_hidden_nodes[i+1] x 1) for i in 0:(k-1))
    pub activations_per_layer: Vec<Matrix>, // activation function output including the input layer as the first element ((n_hidden_nodes[i+1] x 1) for i in 0:(k-1))
    pub weights_gradients_per_layers: Vec<Matrix>, // gradients of the weights ((n_hidden_nodes[i+1] x n_hidden_nodes[i]) for i in 0:(k-1))
    pub biases_gradients_per_layer: Vec<Matrix>, // gradients of the biases ((n_hidden_nodes[i+1] x 1) for i in 0:(k-1))
    pub activation: activations::Activation, // activation function enum (includes derivative)
    pub cost: costs::Cost, // cost function
    pub seed: usize, // random seed for dropouts
}

impl fmt::Display for Network {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f, 
            "
            stream = {:?}
            n_hidden_layers = {}
            n_hidden_nodes = [{}, ..., {}]
            dropout_rates = [{}, ..., {}]
            targets = {}
            predictions = {}
            activation = {:?}
            cost = {:?}
            weights_per_layer = [
                {}, 
                ..., 
                {}
            ]
            biases_per_layer = [
                {}, 
                ..., 
                {}
            ]
            weights_x_biases_per_layer = [
                {}, 
                ..., 
                {}
            ]
            activations_per_layer = [
                {}, 
                ..., 
                {}
            ]
            weights_gradients_per_layers = [
                {}, 
                ..., 
                {}
            ]
            biases_gradients_per_layer = [
                {}, 
                ..., 
                {}
            ]
            ",
            *self.stream,
            self.n_hidden_layers,
            self.n_hidden_nodes[0], self.n_hidden_nodes[self.n_hidden_nodes.len()-1],
            self.dropout_rates[0], self.dropout_rates[self.dropout_rates.len()-1],
            self.targets,
            self.predictions,
            self.activation,
            self.cost,
            self.weights_per_layer[0], self.weights_per_layer[self.weights_per_layer.len()-1],
            self.biases_per_layer[0], self.biases_per_layer[self.biases_per_layer.len()-1],
            self.weights_x_biases_per_layer[0], self.weights_x_biases_per_layer[self.weights_x_biases_per_layer.len()-1],
            self.activations_per_layer[0], self.activations_per_layer[self.activations_per_layer.len()-1],
            self.weights_gradients_per_layers[0], self.weights_gradients_per_layers[self.weights_gradients_per_layers.len()-1],
            self.biases_gradients_per_layer[0], self.biases_gradients_per_layer[self.biases_gradients_per_layer.len()-1],
        )
    }
}

impl Network {
    pub fn new(
        n_observations: usize,
        n_input_nodes: usize,
        n_output_nodes: usize,
        n_hidden_layers: usize,
        n_hidden_nodes: Vec<usize>,
        dropout_rates: Vec<f32>,
        seed: usize,
    ) -> Result<Self,  Box<dyn Error>> {
        if n_observations < 1 {
            return Err(Box::new(MatrixError::DimensionMismatch("We require observations.".to_string())));
        }
        if n_input_nodes < 2 {
            return Err(Box::new(MatrixError::DimensionMismatch("We require at leat 2 input nodes.".to_string())));
        }
        if n_output_nodes < 1 {
            return Err(Box::new(MatrixError::DimensionMismatch("We require at leat 1 output nodes.".to_string())));
        }
        if n_hidden_layers != n_hidden_nodes.len() {
            return Err(Box::new(MatrixError::DimensionMismatch(format!(
                "Number of hidden layers ({}) does not match the number of elements in the vector of number of nodes per hidden layer ({}).",
                n_hidden_layers,
                n_hidden_nodes.len(),
            ))));
        }
        if n_hidden_layers != dropout_rates.len() {
            return Err(Box::new(MatrixError::DimensionMismatch(format!(
                "Number of hidden layers ({}) does not match the number of elements in the vector of dropout rates per hidden layer ({}).",
                n_hidden_layers,
                dropout_rates.len(),
            ))));
        }
        // Number of nodes for all layers, i.e. input node, hiddent-
        let mut n_nodes: Vec<usize> = vec![n_input_nodes];
        for i in 0..n_hidden_layers {
            n_nodes.push(n_hidden_nodes[i]);
        }
        n_nodes.push(n_output_nodes);
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let mut targets_host: Vec<f32> = vec![0f32; n_nodes[0]];
        let mut predictions_host: Vec<f32> = vec![0f32; n_nodes[0]];
        rand::fill(&mut targets_host[..]);
        rand::fill(&mut predictions_host[..]);
        let targets_dev: CudaSlice<f32> = stream.clone_htod(&targets_host)?;
        let predictions_dev: CudaSlice<f32> = stream.clone_htod(&predictions_host)?;
        let predictions: Matrix = Matrix::new(predictions_dev, n_nodes[0], 1)?;
        let targets: Matrix = Matrix::new(targets_dev, n_nodes[0], 1)?;
        let mut weights_per_layer: Vec<Matrix> = vec![];
        let mut weights_gradients_per_layers: Vec<Matrix> = vec![];
        let mut biases_per_layer: Vec<Matrix> = vec![];
        let mut biases_gradients_per_layer: Vec<Matrix> = vec![];
        let mut weights_x_biases_per_layer: Vec<Matrix> = vec![];
        let mut activations_per_layer: Vec<Matrix> = vec![];
        for i in 0..(n_nodes.len()-1) {
            let n: usize = n_nodes[i+1];
            let p: usize = n_nodes[i];
            let mut weights_host: Vec<f32> = vec![0f32; n*p];
            let mut dweights_host: Vec<f32> = vec![0f32; n*p];
            let mut biases_host: Vec<f32> = vec![0f32; n*1];
            let mut dbiases_host: Vec<f32> = vec![0f32; n*1];
            let mut weights_x_biases_host: Vec<f32> = vec![0f32; n*1];
            let mut activations_host: Vec<f32> = vec![0f32; n*1];
            rand::fill(&mut weights_host[..]);
            rand::fill(&mut dweights_host[..]);
            rand::fill(&mut biases_host[..]);
            rand::fill(&mut dbiases_host[..]);
            rand::fill(&mut weights_x_biases_host[..]);
            rand::fill(&mut activations_host[..]);
            let weights_dev: CudaSlice<f32> = stream.clone_htod(&weights_host)?;
            let dweights_dev: CudaSlice<f32> = stream.clone_htod(&dweights_host)?;
            let biases_dev: CudaSlice<f32> = stream.clone_htod(&biases_host)?;
            let dbiases_dev: CudaSlice<f32> = stream.clone_htod(&dbiases_host)?;
            let weights_x_biases_dev: CudaSlice<f32> = stream.clone_htod(&weights_x_biases_host)?;
            let activations_dev: CudaSlice<f32> = stream.clone_htod(&activations_host)?;
            let weights_matrix: Matrix = Matrix::new(weights_dev, n, p)?;
            let dweights_matrix: Matrix = Matrix::new(dweights_dev, n, p)?;
            let biases_matrix: Matrix = Matrix::new(biases_dev, n, 1)?;
            let dbiases_matrix: Matrix = Matrix::new(dbiases_dev, n, 1)?;
            let weights_x_biases_matrix: Matrix = Matrix::new(weights_x_biases_dev, n, 1)?;
            let activations_matrix: Matrix = Matrix::new(activations_dev, n, 1)?;
            weights_per_layer.push(weights_matrix);
            weights_gradients_per_layers.push(dweights_matrix);
            biases_per_layer.push(biases_matrix);
            biases_gradients_per_layer.push(dbiases_matrix);
            weights_x_biases_per_layer.push(weights_x_biases_matrix);
            activations_per_layer.push(activations_matrix);
        }
        let out = Self {
            stream: stream,
            n_hidden_layers: n_hidden_layers,
            n_hidden_nodes: n_hidden_nodes,
            dropout_rates: dropout_rates,
            targets: targets,
            predictions: predictions,
            weights_per_layer: weights_per_layer,
            weights_gradients_per_layers: weights_gradients_per_layers,
            biases_per_layer: biases_per_layer,
            biases_gradients_per_layer: biases_gradients_per_layer,
            weights_x_biases_per_layer: weights_x_biases_per_layer,
            activations_per_layer: activations_per_layer,
            activation: activations::Activation::ReLU,
            cost: costs::Cost::MSE,
            seed: seed,
        };
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_network() -> Result<(), Box<dyn std::error::Error>> {
        let mut network: Network = Network::new(
            10_000,
            1_000,
            1,
            10,
            vec![256; 10],
            vec![0.0f32; 10],
            42,
        )?;
        println!("network: {}", network);
        // Update activation and cost functions
        network.activation = activations::Activation::Sigmoid;
        network.cost = costs::Cost::MAE;
        println!("network: {}", network);

        Ok(())
    }
}