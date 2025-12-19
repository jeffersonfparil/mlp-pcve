use std::error::Error;
use std::fmt;
use crate::linalg::matrix::Matrix;
use crate::linalg::add::elemetwisematadd;
use crate::linalg::mult::scalarmatmul;
use crate::network::Network;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct OptimiserParameters {
    learning_rate: f32, // η = 0.001
    first_moment_decay: f32, // β₁ = 0.900
    second_moment_decay: f32, // β₁ = 0.999
    epsilon: f32, // ϵ = 1e-8 for numerical stability
    time_step: usize, // t = 0
    first_moments_of_weights_per_layer: Vec<Matrix>,
    second_moments_of_weights_per_layer: Vec<Matrix>,
    first_moments_of_biases_per_layer: Vec<Matrix>,
    second_moments_of_biases_per_layer: Vec<Matrix>,
}

impl fmt::Display for OptimiserParameters {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "
                base learning rate = {}
                first moment decay coefficient = {}
                second moment decay coefficient = {}
                epsilon for numerical stability = {}
                time step: {}
                first_moments_of_weights_per_layer: [{}, ..., {}]
                second_moments_of_weights_per_layer: [{}, ..., {}]
                first_moments_of_biases_per_layer: [{}, ..., {}]
                second_moments_of_biases_per_layer: [{}, ..., {}]
            ",
            self.learning_rate,
            self.first_moment_decay,
            self.second_moment_decay,
            self.epsilon,
            self.time_step,
            self.first_moments_of_weights_per_layer[0], self.first_moments_of_weights_per_layer[self.first_moments_of_weights_per_layer.len()-1],
            self.second_moments_of_weights_per_layer[0], self.second_moments_of_weights_per_layer[self.second_moments_of_weights_per_layer.len()-1],
            self.first_moments_of_biases_per_layer[0], self.first_moments_of_biases_per_layer[self.first_moments_of_biases_per_layer.len()-1],
            self.second_moments_of_biases_per_layer[0], self.second_moments_of_biases_per_layer[self.second_moments_of_biases_per_layer.len()-1],
        )
    }
}

impl OptimiserParameters {
    pub fn new(network: &Network) -> Result<Self, Box<dyn Error>> {
        let mut first_moments_of_weights_per_layer: Vec<Matrix> = vec![];
        let mut second_moments_of_weights_per_layer: Vec<Matrix> = vec![];
        let mut first_moments_of_biases_per_layer: Vec<Matrix> = vec![];
        let mut second_moments_of_biases_per_layer: Vec<Matrix> = vec![];
        for i in 0..(network.n_hidden_layers+1) {
            let n = network.weights_per_layer[i].n_rows;
            let p = network.weights_per_layer[i].n_cols;
            let weights_host = vec![0.0f32; n*p];
            let weights_dev = network.stream.clone_htod(&weights_host)?;
            let weights_matrix = Matrix::new(weights_dev, n, p)?;
            first_moments_of_weights_per_layer.push(weights_matrix.clone());
            second_moments_of_weights_per_layer.push(weights_matrix);

            let n = network.biases_per_layer[i].n_rows;
            let p = network.biases_per_layer[i].n_cols;
            assert!(p == 1);
            let biases_host = vec![0.0f32; n*p];
            let biases_dev = network.stream.clone_htod(&biases_host)?;
            let biases_matrix = Matrix::new(biases_dev, n, p)?;
            first_moments_of_biases_per_layer.push(biases_matrix.clone());
            second_moments_of_biases_per_layer.push(biases_matrix);
        }
        let out = Self {
            learning_rate: 0.001,
            first_moment_decay: 0.900,
            second_moment_decay: 0.999,
            epsilon: 1.0e-8,
            time_step: 0,
            first_moments_of_weights_per_layer: first_moments_of_weights_per_layer,
            second_moments_of_weights_per_layer: second_moments_of_weights_per_layer,
            first_moments_of_biases_per_layer: first_moments_of_biases_per_layer,
            second_moments_of_biases_per_layer: second_moments_of_biases_per_layer,
        };
        Ok(out)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Optimiser {
    GradientDescent,
    Adam,
    AdamMax,
}

pub fn gradientdescent(network: &mut Network, optimiser_parameters: &mut OptimiserParameters) -> Result<(), Box<dyn Error>> {
    for i in 0..(network.n_hidden_layers+1) {
        network.weights_per_layer[i] = scalarmatmul(optimiser_parameters.learning_rate, &network.weights_gradients_per_layer[i])?;
        network.biases_per_layer[i] = scalarmatmul(optimiser_parameters.learning_rate, &network.biases_gradients_per_layer[i])?;
    }
    Ok(())
}

pub fn adam(network: &mut Network, optimiser_parameters: &mut OptimiserParameters) -> Result<(), Box<dyn Error>> {
    optimiser_parameters.time_step += 1;
    for i in 0..(network.n_hidden_layers+1) {
        optimiser_parameters.first_moments_of_weights_per_layer[i] = elemetwisematadd(
            &scalarmatmul(
                optimiser_parameters.first_moment_decay, 
                &optimiser_parameters.first_moments_of_weights_per_layer[i],
            )?,
            &scalarmatmul(
                1.00f32 - optimiser_parameters.first_moment_decay, 
                &network.weights_gradients_per_layer[i],
            )?,
        )?;
    }

    // TODO
    Ok(())
}

pub fn adammax(network: &mut Network, optimiser_parameters: &mut OptimiserParameters) -> Result<(), Box<dyn Error>> {
    // TODO
    Ok(())
}


impl Optimiser {
    pub fn optimise(&self, network: &mut Network, optimiser_parameters: &mut OptimiserParameters) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::{CudaContext, CudaSlice};
    #[test]
    fn test_optimisers() -> Result<(), Box<dyn Error>> {
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
        println!("network (init): {}", network);
        let mut optimiser_parameters = OptimiserParameters::new(&network)?;
        println!("optimiser_parameters: {}", optimiser_parameters);

        network.forwardpass()?;
        println!("network (after forwardpass): {}", network);
        network.backpropagation()?;
        println!("network (after backpropagation): {}", network);

        gradientdescent(&mut network, &mut optimiser_parameters)?;
        println!("network (after gradientdescent): {}", network);
        println!("optimiser_parameters (after gradientdescent): {}", optimiser_parameters);

        adam(&mut network, &mut optimiser_parameters)?;
        println!("network (after adam): {}", network);
        println!("optimiser_parameters (after adam): {}", optimiser_parameters);



        Ok(())
    }
}