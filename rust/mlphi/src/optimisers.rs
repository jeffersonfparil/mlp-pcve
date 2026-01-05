use crate::linalg::add::{elementwisematadd, scalarmatadd};
use crate::linalg::matrix::Matrix;
use crate::linalg::mult::{elementwisematinverse, elementwisematmul, scalarmatmul};
use std::error::Error;
use std::fmt;
// use crate::linalg::sqrt::elementwisematsqrt;
use crate::network::Network;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct OptimiserParameters {
    learning_rate: f32,       // η = 0.001
    first_moment_decay: f32,  // β₁ = 0.900
    second_moment_decay: f32, // β₁ = 0.999
    epsilon: f32,             // ϵ = 1e-8 for numerical stability
    time_step: usize,         // t = 0
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
            self.first_moments_of_weights_per_layer[0],
            self.first_moments_of_weights_per_layer
                [self.first_moments_of_weights_per_layer.len() - 1],
            self.second_moments_of_weights_per_layer[0],
            self.second_moments_of_weights_per_layer
                [self.second_moments_of_weights_per_layer.len() - 1],
            self.first_moments_of_biases_per_layer[0],
            self.first_moments_of_biases_per_layer
                [self.first_moments_of_biases_per_layer.len() - 1],
            self.second_moments_of_biases_per_layer[0],
            self.second_moments_of_biases_per_layer
                [self.second_moments_of_biases_per_layer.len() - 1],
        )
    }
}

impl OptimiserParameters {
    pub fn new(network: &Network) -> Result<Self, Box<dyn Error>> {
        let mut first_moments_of_weights_per_layer: Vec<Matrix> = vec![];
        let mut second_moments_of_weights_per_layer: Vec<Matrix> = vec![];
        let mut first_moments_of_biases_per_layer: Vec<Matrix> = vec![];
        let mut second_moments_of_biases_per_layer: Vec<Matrix> = vec![];
        for i in 0..(network.n_hidden_layers + 1) {
            let n = network.weights_per_layer[i].n_rows;
            let p = network.weights_per_layer[i].n_cols;
            let weights_host = vec![0.0f32; n * p];
            let weights_dev = network.stream.clone_htod(&weights_host)?;
            let weights_matrix = Matrix::new(weights_dev, n, p)?;
            first_moments_of_weights_per_layer.push(weights_matrix.clone());
            second_moments_of_weights_per_layer.push(weights_matrix);

            let n = network.biases_per_layer[i].n_rows;
            let p = network.biases_per_layer[i].n_cols;
            assert!(p == 1);
            let biases_host = vec![0.0f32; n * p];
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

pub fn gradientdescent(
    network: &mut Network,
    optimiser_parameters: &mut OptimiserParameters,
) -> Result<(), Box<dyn Error>> {
    for i in 0..(network.n_hidden_layers + 1) {
        network.weights_per_layer[i] = scalarmatmul(
            optimiser_parameters.learning_rate,
            &network.weights_gradients_per_layer[i],
        )?;
        network.biases_per_layer[i] = scalarmatmul(
            optimiser_parameters.learning_rate,
            &network.biases_gradients_per_layer[i],
        )?;
    }
    Ok(())
}

pub fn adam(
    network: &mut Network,
    optimiser_parameters: &mut OptimiserParameters,
) -> Result<(), Box<dyn Error>> {
    optimiser_parameters.time_step += 1;
    for i in 0..(network.n_hidden_layers + 1) {
        // Weights update
        // optimiser_parameters.first_moments_of_weights_per_layer[i] = elementwisematadd(
        //     &scalarmatmul(
        //         optimiser_parameters.first_moment_decay,
        //         &optimiser_parameters.first_moments_of_weights_per_layer[i],
        //     )?,
        //     &scalarmatmul(
        //         1.00f32 - optimiser_parameters.first_moment_decay,
        //         &network.weights_gradients_per_layer[i],
        //     )?,
        // )?;
        optimiser_parameters.first_moments_of_weights_per_layer[i] = optimiser_parameters
            .first_moments_of_weights_per_layer[i]
            .scalarmatmul(optimiser_parameters.first_moment_decay)?
            .elementwisematadd(
                &network.weights_gradients_per_layer[i]
                    .scalarmatmul(1.00f32 - optimiser_parameters.first_moment_decay)?,
            )?;

        optimiser_parameters.second_moments_of_weights_per_layer[i] = elementwisematadd(
            &scalarmatmul(
                optimiser_parameters.second_moment_decay,
                &optimiser_parameters.second_moments_of_weights_per_layer[i],
            )?,
            &scalarmatmul(
                1.00f32 - optimiser_parameters.second_moment_decay,
                &network.weights_gradients_per_layer[i],
            )?,
        )?;
        let factor_moment_1 = 1.00
            / (1.00
                - optimiser_parameters
                    .first_moment_decay
                    .powi(optimiser_parameters.time_step as i32));
        let factor_moment_2 = 1.00
            / (1.00
                - optimiser_parameters
                    .second_moment_decay
                    .powi(optimiser_parameters.time_step as i32));
        let momentum = scalarmatmul(
            factor_moment_1,
            &optimiser_parameters.first_moments_of_weights_per_layer[i],
        )?;
        let velocity = scalarmatmul(
            factor_moment_2,
            &optimiser_parameters.second_moments_of_weights_per_layer[i],
        )?;
        // Ω.W[i] -= (η * momentum ./ (sqrt.(velocity) .+ state["ϵ"]))
        network.weights_per_layer[i] = elementwisematadd(
            &network.weights_per_layer[i],
            &scalarmatmul(
                -1.0,
                &elementwisematmul(
                    &scalarmatmul(optimiser_parameters.learning_rate, &momentum)?,
                    &elementwisematinverse(&scalarmatadd(
                        optimiser_parameters.epsilon,
                        &velocity.elementwisematsqrt()?,
                    )?)?,
                )?,
            )?,
        )?;

        let mut a_host = vec![
            0.0f32;
            optimiser_parameters.first_moments_of_weights_per_layer[i].n_rows
                * optimiser_parameters.first_moments_of_weights_per_layer[i]
                    .n_cols
        ];
        network.stream.memcpy_dtoh(
            &optimiser_parameters.first_moments_of_weights_per_layer[i].data,
            &mut a_host,
        )?;
        println!(
            "????????? layer {}: [{}, {}, {}, ..., {}]",
            i,
            a_host[0],
            a_host[1],
            a_host[2],
            a_host[a_host.len() - 1]
        );

        // Biases update
        optimiser_parameters.first_moments_of_biases_per_layer[i] = elementwisematadd(
            &scalarmatmul(
                optimiser_parameters.first_moment_decay,
                &optimiser_parameters.first_moments_of_biases_per_layer[i],
            )?,
            &scalarmatmul(
                1.00f32 - optimiser_parameters.first_moment_decay,
                &network.biases_gradients_per_layer[i],
            )?,
        )?;
        optimiser_parameters.second_moments_of_biases_per_layer[i] = elementwisematadd(
            &scalarmatmul(
                optimiser_parameters.second_moment_decay,
                &optimiser_parameters.second_moments_of_biases_per_layer[i],
            )?,
            &scalarmatmul(
                1.00f32 - optimiser_parameters.second_moment_decay,
                &network.biases_gradients_per_layer[i],
            )?,
        )?;
        let factor_moment_1 = 1.00
            / (1.00
                - optimiser_parameters
                    .first_moment_decay
                    .powi(optimiser_parameters.time_step as i32));
        let factor_moment_2 = 1.00
            / (1.00
                - optimiser_parameters
                    .second_moment_decay
                    .powi(optimiser_parameters.time_step as i32));
        let momentum = scalarmatmul(
            factor_moment_1,
            &optimiser_parameters.first_moments_of_biases_per_layer[i],
        )?;
        let velocity = scalarmatmul(
            factor_moment_2,
            &optimiser_parameters.second_moments_of_biases_per_layer[i],
        )?;
        // Ω.b[i] -= (η * momentum ./ (sqrt.(velocity) .+ state["ϵ"]))
        network.biases_per_layer[i] = elementwisematadd(
            &network.biases_per_layer[i],
            &scalarmatmul(
                -1.0,
                &elementwisematmul(
                    &scalarmatmul(optimiser_parameters.learning_rate, &momentum)?,
                    &elementwisematinverse(&scalarmatadd(
                        optimiser_parameters.epsilon,
                        &velocity.elementwisematsqrt()?,
                    )?)?,
                )?,
            )?,
        )?;
    }
    Ok(())
}

pub fn adammax(
    network: &mut Network,
    optimiser_parameters: &mut OptimiserParameters,
) -> Result<(), Box<dyn Error>> {
    // TODO
    Ok(())
}

impl Optimiser {
    pub fn optimise(
        &self,
        network: &mut Network,
        optimiser_parameters: &mut OptimiserParameters,
    ) -> Result<(), Box<dyn Error>> {
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

        let i: usize = 1;
        let mut a_host =
            vec![0.0f32; network.weights_per_layer[i].n_rows * network.weights_per_layer[i].n_cols];
        network
            .stream
            .memcpy_dtoh(&network.weights_per_layer[i].data, &mut a_host)?;
        println!(
            "weights of layer {} (before optimisations): [{}, {}, {}, ..., {}]",
            i,
            a_host[0],
            a_host[1],
            a_host[2],
            a_host[a_host.len() - 1]
        );

        network.forwardpass()?;
        network
            .stream
            .memcpy_dtoh(&network.weights_per_layer[i].data, &mut a_host)?;
        println!(
            "weights of layer {} (after forwardpass): [{}, {}, {}, ..., {}]",
            i,
            a_host[0],
            a_host[1],
            a_host[2],
            a_host[a_host.len() - 1]
        );

        network.backpropagation()?;
        network
            .stream
            .memcpy_dtoh(&network.weights_per_layer[i].data, &mut a_host)?;
        println!(
            "weights of layer {} (after backpropagation): [{}, {}, {}, ..., {}]",
            i,
            a_host[0],
            a_host[1],
            a_host[2],
            a_host[a_host.len() - 1]
        );
        network
            .stream
            .memcpy_dtoh(&network.weights_gradients_per_layer[i].data, &mut a_host)?;
        println!(
            "weights gradients of layer {} (after backpropagation): [{}, {}, {}, ..., {}]",
            i,
            a_host[0],
            a_host[1],
            a_host[2],
            a_host[a_host.len() - 1]
        );

        gradientdescent(&mut network, &mut optimiser_parameters)?;
        // println!("network (after gradientdescent): {}", network);
        // println!("optimiser_parameters (after gradientdescent): {}", optimiser_parameters);
        network
            .stream
            .memcpy_dtoh(&network.weights_per_layer[i].data, &mut a_host)?;
        println!(
            "weights of layer {} (after gradientdescent): [{}, {}, {}, ..., {}]",
            i,
            a_host[0],
            a_host[1],
            a_host[2],
            a_host[a_host.len() - 1]
        );

        network.forwardpass()?;
        // println!("network (after forwardpass): {}", network);
        network.backpropagation()?;
        // println!("network (after backpropagation): {}", network);

        adam(&mut network, &mut optimiser_parameters)?;
        // println!("network (after adam): {}", network);
        // println!("optimiser_parameters (after adam): {}", optimiser_parameters);
        network
            .stream
            .memcpy_dtoh(&network.weights_per_layer[i].data, &mut a_host)?;
        println!(
            "weights of layer {} (after adam): [{}, {}, {}, ..., {}]",
            i,
            a_host[0],
            a_host[1],
            a_host[2],
            a_host[a_host.len() - 1]
        );

        Ok(())
    }
}
