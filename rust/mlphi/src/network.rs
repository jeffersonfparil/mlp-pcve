use crate::activations;
use crate::costs;
use crate::linalg::matrix::{Matrix, MatrixError};
use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use rand_distr::Normal;
use std::error::Error;
use std::fmt;
use std::sync::Arc;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Network {
    pub stream: Arc<CudaStream>,                  // CUDA device stream
    pub n_hidden_layers: usize,                   // number of hidden layers
    pub n_hidden_nodes: Vec<usize>,               // number of nodes per hidden layer (k)
    pub dropout_rates: Vec<f32>,                  // soft dropout rates per hidden layer (k)
    pub targets: Matrix,                          // observed values (n x 1)
    pub predictions: Matrix,                      // predictions (n x 1)
    pub weights_per_layer: Vec<Matrix>, // weights ((n_hidden_nodes[i+1] x n_hidden_nodes[i]) for i in 0:(k-1))
    pub biases_per_layer: Vec<Matrix>,  // biases ((n_hidden_nodes[i+1] x 1) for i in 0:(k-1))
    pub weights_x_biases_per_layer: Vec<Matrix>, // summed weights (i.e. prior to activation function) ((n_hidden_nodes[i+1] x 1) for i in 0:(k-1))
    pub activations_per_layer: Vec<Matrix>, // activation function output including the input layer as the first element ((n_hidden_nodes[i+1] x 1) for i in 0:(k-1))
    pub weights_gradients_per_layer: Vec<Matrix>, // gradients of the weights ((n_hidden_nodes[i+1] x n_hidden_nodes[i]) for i in 0:(k-1))
    pub biases_gradients_per_layer: Vec<Matrix>, // gradients of the biases ((n_hidden_nodes[i+1] x 1) for i in 0:(k-1))
    pub activation: activations::Activation,     // activation function enum (includes derivative)
    pub cost: costs::Cost,                       // cost function
    pub seed: usize,                             // random seed for dropouts
}

pub fn printcost(network: &Network) -> Result<(), Box<dyn Error>> {
    let mse = network
        .cost
        .cost(&network.predictions, &network.targets)?
        .summat(&network.stream)? / (network.targets.n_cols as f32);
    println!("{:?} = {}", network.cost, mse);
    Ok(())
}
pub fn printpredictions(network: &Network) -> Result<(), Box<dyn Error>> {
    let n = network.predictions.n_rows * network.predictions.n_cols;
    let mut a_host = vec![0.0f32; n];
    network
        .stream
        .memcpy_dtoh(&network.predictions.data, &mut a_host)?;
    if n < 4 {
        println!(
            "predictions (n={}): [{}, ..., {}]",
            n,
            a_host[0],
            a_host[a_host.len() - 1]
        );
    } else {
        println!(
            "predictions (n={}): [{}, {}, {}, ..., {}]",
            n,
            a_host[0],
            a_host[1],
            a_host[2],
            a_host[a_host.len() - 1]
        );
    }
    Ok(())
}
pub fn printactivations(network: &Network, layer: usize) -> Result<(), Box<dyn Error>> {
    let n = network.activations_per_layer[layer].n_rows * network.activations_per_layer[layer].n_cols;
    let mut a_host = vec![0.0f32; n];
    network
        .stream
        .memcpy_dtoh(&network.activations_per_layer[layer].data, &mut a_host)?;
    if n < 4 {
        println!(
            "activations (layer={}; n={}): [{}, ..., {}]",
            layer,
            n,
            a_host[0],
            a_host[a_host.len() - 1]
        );
    } else {
        println!(
            "activations (layer={}; n={}): [{}, {}, {}, ..., {}]",
            layer,
            n,
            a_host[0],
            a_host[1],
            a_host[2],
            a_host[a_host.len() - 1]
        );
    }
    Ok(())
}
pub fn printweights(network: &Network, layer: usize) -> Result<(), Box<dyn Error>> {
    let n = network.weights_per_layer[layer].n_rows * network.weights_per_layer[layer].n_cols;
    let mut a_host = vec![0.0f32; n];
    network
        .stream
        .memcpy_dtoh(&network.weights_per_layer[layer].data, &mut a_host)?;
    if n < 4 {
        println!(
            "weights (layer={}; n={}): [{}, ..., {}]",
            layer,
            n,
            a_host[0],
            a_host[a_host.len() - 1]
        );
    } else {
        println!(
            "weights (layer={}; n={}): [{}, {}, {}, ..., {}]",
            layer,
            n,
            a_host[0],
            a_host[1],
            a_host[2],
            a_host[a_host.len() - 1]
        );
    }
    Ok(())
}
pub fn printbiases(network: &Network, layer: usize) -> Result<(), Box<dyn Error>> {
    let n = network.biases_per_layer[layer].n_rows * network.biases_per_layer[layer].n_cols;
    let mut a_host = vec![0.0f32; n];
    network
        .stream
        .memcpy_dtoh(&network.biases_per_layer[layer].data, &mut a_host)?;
    if n < 4 {
        println!(
            "biases (layer={}; n={}): [{}, ..., {}]",
            layer,
            n,
            a_host[0],
            a_host[a_host.len() - 1]
        );
    } else {
        println!(
            "biases (layer={}; n={}): [{}, {}, {}, ..., {}]",
            layer,
            n,
            a_host[0],
            a_host[1],
            a_host[2],
            a_host[a_host.len() - 1]
        );
    }
    Ok(())
}
pub fn printweightsgradients(network: &Network, layer: usize) -> Result<(), Box<dyn Error>> {
    let n = network.weights_gradients_per_layer[layer].n_rows * network.weights_gradients_per_layer[layer].n_cols;
    let mut a_host = vec![0.0f32; n];
    network
        .stream
        .memcpy_dtoh(&network.weights_gradients_per_layer[layer].data, &mut a_host)?;
    if n < 4 {
        println!(
            "weights gradients (layer={}; n={}): [{}, ..., {}]",
            layer,
            n,
            a_host[0],
            a_host[a_host.len() - 1]
        );
    } else {
        println!(
            "weights gradients (layer={}; n={}): [{}, {}, {}, ..., {}]",
            layer,
            n,
            a_host[0],
            a_host[1],
            a_host[2],
            a_host[a_host.len() - 1]
        );
    }
    Ok(())
}
pub fn printbiasesgradients(network: &Network, layer: usize) -> Result<(), Box<dyn Error>> {
    let n = network.biases_gradients_per_layer[layer].n_rows * network.biases_gradients_per_layer[layer].n_cols;
    let mut a_host = vec![0.0f32; n];
    network
        .stream
        .memcpy_dtoh(&network.biases_gradients_per_layer[layer].data, &mut a_host)?;
    if n < 4 {
        println!(
            "biases gradients (layer={}; n={}): [{}, ..., {}]",
            layer,
            n,
            a_host[0],
            a_host[a_host.len() - 1]
        );
    } else {
        println!(
            "biases gradients (layer={}; n={}): [{}, {}, {}, ..., {}]",
            layer,
            n,
            a_host[0],
            a_host[1],
            a_host[2],
            a_host[a_host.len() - 1]
        );
    }
    Ok(())
}

impl fmt::Display for Network {
    // fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), Box<dyn Error>> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
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
            weights_per_layer (len={}) = [
                {} (sum={}), 
                ..., 
                {} (sum={})
            ]
            biases_per_layer (len={}) = [
                {} (sum={}), 
                ..., 
                {} (sum={})
            ]
            weights_x_biases_per_layer (len={}) = [
                {} (sum={}), 
                ..., 
                {} (sum={})
            ]
            activations_per_layer (len={}) = [
                {} (sum={}), 
                ..., 
                {} (sum={})
            ]
            weights_gradients_per_layer (len={}) = [
                {} (sum={}), 
                ..., 
                {} (sum={})
            ]
            biases_gradients_per_layer (len={}) = [
                {} (sum={}), 
                ..., 
                {} (sum={})
            ]
            ",
            *self.stream,
            self.n_hidden_layers,
            self.n_hidden_nodes[0],
            self.n_hidden_nodes[self.n_hidden_nodes.len() - 1],
            self.dropout_rates[0],
            self.dropout_rates[self.dropout_rates.len() - 1],
            self.targets,
            self.predictions,
            self.activation,
            self.cost,
            self.weights_per_layer.len(),
            self.weights_per_layer[0],
            match self.weights_per_layer[0].summat(&self.stream) {
                Ok(x) => x,
                Err(_) => return Err(fmt::Error),
            },
            self.weights_per_layer[self.weights_per_layer.len() - 1],
            match self.weights_per_layer[self.weights_per_layer.len() - 1].summat(&self.stream) {
                Ok(x) => x,
                Err(_) => return Err(fmt::Error),
            },
            self.biases_per_layer.len(),
            self.biases_per_layer[0],
            match self.biases_per_layer[0].summat(&self.stream) {
                Ok(x) => x,
                Err(_) => return Err(fmt::Error),
            },
            self.biases_per_layer[self.biases_per_layer.len() - 1],
            match self.biases_per_layer[self.biases_per_layer.len() - 1].summat(&self.stream,) {
                Ok(x) => x,
                Err(_) => return Err(fmt::Error),
            },
            self.weights_x_biases_per_layer.len(),
            self.weights_x_biases_per_layer[0],
            match self.weights_x_biases_per_layer[0].summat(&self.stream) {
                Ok(x) => x,
                Err(_) => return Err(fmt::Error),
            },
            self.weights_x_biases_per_layer[self.weights_x_biases_per_layer.len() - 1],
            match self.weights_x_biases_per_layer[self.weights_x_biases_per_layer.len() - 1]
                .summat(&self.stream,)
            {
                Ok(x) => x,
                Err(_) => return Err(fmt::Error),
            },
            self.activations_per_layer.len(),
            self.activations_per_layer[0],
            match self.activations_per_layer[0].summat(&self.stream) {
                Ok(x) => x,
                Err(_) => return Err(fmt::Error),
            },
            self.activations_per_layer[self.activations_per_layer.len() - 1],
            match self.activations_per_layer[self.activations_per_layer.len() - 1]
                .summat(&self.stream)
            {
                Ok(x) => x,
                Err(_) => return Err(fmt::Error),
            },
            self.weights_gradients_per_layer.len(),
            self.weights_gradients_per_layer[0],
            match self.weights_gradients_per_layer[0].summat(&self.stream) {
                Ok(x) => x,
                Err(_) => return Err(fmt::Error),
            },
            self.weights_gradients_per_layer[self.weights_gradients_per_layer.len() - 1],
            match self.weights_gradients_per_layer[self.weights_gradients_per_layer.len() - 1]
                .summat(&self.stream)
            {
                Ok(x) => x,
                Err(_) => return Err(fmt::Error),
            },
            self.biases_gradients_per_layer.len(),
            self.biases_gradients_per_layer[0],
            match self.biases_gradients_per_layer[0].summat(&self.stream) {
                Ok(x) => x,
                Err(_) => return Err(fmt::Error),
            },
            self.biases_gradients_per_layer[self.biases_gradients_per_layer.len() - 1],
            match self.biases_gradients_per_layer[self.biases_gradients_per_layer.len() - 1]
                .summat(&self.stream)
            {
                Ok(x) => x,
                Err(_) => return Err(fmt::Error),
            },
        )
    }
}

impl Network {
    pub fn new(
        stream: Arc<CudaStream>,
        input_data: Matrix,
        output_data: Matrix,
        n_hidden_layers: usize,
        n_hidden_nodes: Vec<usize>,
        dropout_rates: Vec<f32>,
        seed: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let n_observations: usize = input_data.n_cols;
        let n_input_nodes: usize = input_data.n_rows;
        let n_output_nodes: usize = output_data.n_rows;
        if n_observations != output_data.n_cols {
            return Err(Box::new(MatrixError::DimensionMismatch(
                "We require the same number of observations for both input and output matrices."
                    .to_string(),
            )));
        }
        if n_observations < 1 {
            return Err(Box::new(MatrixError::DimensionMismatch(
                "We require observations.".to_string(),
            )));
        }
        if n_input_nodes < 2 {
            return Err(Box::new(MatrixError::DimensionMismatch(
                "We require at leat 2 input nodes.".to_string(),
            )));
        }
        if n_output_nodes < 1 {
            return Err(Box::new(MatrixError::DimensionMismatch(
                "We require at leat 1 output nodes.".to_string(),
            )));
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
        let predictions_host: Vec<f32> = {
            let mut tmp = vec![0f32; n_observations * n_output_nodes];
            rand::fill(&mut tmp[..]);
            tmp
        };
        let mut rng = ChaCha12Rng::seed_from_u64(seed as u64);
        let normal = Normal::new(0.0, 1.0)?;
        let predictions_dev: CudaSlice<f32> = stream.clone_htod(&predictions_host)?;
        let predictions: Matrix = Matrix::new(predictions_dev, n_output_nodes, n_observations)?;
        let mut weights_per_layer: Vec<Matrix> = vec![];
        let mut weights_gradients_per_layer: Vec<Matrix> = vec![];
        let mut biases_per_layer: Vec<Matrix> = vec![];
        let mut biases_gradients_per_layer: Vec<Matrix> = vec![];
        let mut weights_x_biases_per_layer: Vec<Matrix> = vec![];
        let mut activations_per_layer: Vec<Matrix> = vec![input_data];
        for i in 0..(n_nodes.len() - 1) {
            let n: usize = n_nodes[i + 1];
            let p: usize = n_nodes[i];
            // let normal = Normal::new(0.0, 2.0/(p as f32).sqrt())?;
            let weights_host: Vec<f32> = (&mut rng).sample_iter(normal).take(n * p).collect();
            let dweights_host: Vec<f32> = vec![0f32; n * p];
            let biases_host: Vec<f32> = vec![0f32; n * 1];
            let dbiases_host: Vec<f32> = vec![0f32; n * 1];
            let weights_x_biases_host: Vec<f32> = vec![0f32; n * n_observations];
            if i > 0 {
                let activations_host: Vec<f32> = vec![0f32; p * n_observations];
                let activations_dev: CudaSlice<f32> = stream.clone_htod(&activations_host)?;
                let activations_matrix: Matrix = Matrix::new(activations_dev, p, n_observations)?;
                activations_per_layer.push(activations_matrix);
            }
            let weights_dev: CudaSlice<f32> = stream.clone_htod(&weights_host)?;
            let dweights_dev: CudaSlice<f32> = stream.clone_htod(&dweights_host)?;
            let biases_dev: CudaSlice<f32> = stream.clone_htod(&biases_host)?;
            let dbiases_dev: CudaSlice<f32> = stream.clone_htod(&dbiases_host)?;
            let weights_x_biases_dev: CudaSlice<f32> = stream.clone_htod(&weights_x_biases_host)?;
            let weights_matrix: Matrix = Matrix::new(weights_dev, n, p)?;
            let dweights_matrix: Matrix = Matrix::new(dweights_dev, n, p)?;
            let biases_matrix: Matrix = Matrix::new(biases_dev, n, 1)?;
            let dbiases_matrix: Matrix = Matrix::new(dbiases_dev, n, 1)?;
            let weights_x_biases_matrix: Matrix =
                Matrix::new(weights_x_biases_dev, n, n_observations)?;
            weights_per_layer.push(weights_matrix);
            weights_gradients_per_layer.push(dweights_matrix);
            biases_per_layer.push(biases_matrix);
            biases_gradients_per_layer.push(dbiases_matrix);
            weights_x_biases_per_layer.push(weights_x_biases_matrix);
        }
        let out = Self {
            stream: stream,
            n_hidden_layers: n_hidden_layers,
            n_hidden_nodes: n_hidden_nodes,
            dropout_rates: dropout_rates,
            targets: output_data,
            predictions: predictions,
            weights_per_layer: weights_per_layer,
            weights_gradients_per_layer: weights_gradients_per_layer,
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
    fn test_network() -> Result<(), Box<dyn Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let n: usize = 10_000;
        let p: usize = 123;
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
        println!("network: {}", network);
        // Update activation and cost functions
        network.activation = activations::Activation::Sigmoid;
        network.cost = costs::Cost::MAE;
        println!("network: {}", network);

        assert_eq!(network.activation, activations::Activation::Sigmoid);
        assert_eq!(network.cost, costs::Cost::MAE);

        Ok(())
    }
}
