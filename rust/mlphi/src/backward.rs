use std::error::Error;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use cudarc::driver::{CudaContext, CudaSlice};
use crate::linalg::matrix::Matrix;
use crate::linalg::add::rowmatadd;
use crate::linalg::mult::{scalarmatmul, elementwisematmul, matmult0, matmul0t, matmul};
use crate::linalg::fold::rowsummat;
use crate::network::Network;

impl Network {
    pub fn backpropagation(
        &mut self
    ) -> Result<(), Box<dyn Error>> {
        // Cost gradients with respect to (w.r.t.) the weights: ∂C/∂Wˡ = (∂C/∂Aᴸ) * (∂Aᴸ/∂Sˡ) * (∂Sˡ/∂Wˡ)
        // Starting with the output layer down to the first hidden layer
        // Cost derivative with respect to (w.r.t.) to the activations at the output layer 
        let dC_over_dAL = self.cost.derivative(&self.predictions, &self.targets)?;
        println!("dC_over_dAL = {}", dC_over_dAL);
        // Activation derivative w.r.t. the sum of the weights (i.e. pre-activation values) at the output layer which is just 1.00 (linear activation) because this is a regression and not a classification network
        let dAl_over_dSl = 1.00f32;
        println!("dAl_over_dSl = {}", dAl_over_dSl);
        // Error for the output layer (cost derivative w.r.t. the sum of the weights via chain rule): element-wise product of the cost derivatives and activation derivatives
        let dC_over_dSL = scalarmatmul(dAl_over_dSl, &dC_over_dAL)?;
        println!("dC_over_dSL = {}", dC_over_dSL);
        let mut delta: Vec<Matrix> = vec![dC_over_dSL];
        println!("delta[0] = {}", delta[0]);
        // Now let us proceed from the last layer to the first hidden layer (i.e. just before the input layer)
        let n_total_layers = self.weights_per_layer.len();
        println!("n_total_layers = {}", n_total_layers);
        println!("self.n_hidden_layers = {}", self.n_hidden_layers);
        for i in 1..self.n_hidden_layers {
            println!("i={}", i);
            println!("self.weights_per_layer[n_total_layers-i] = {}", self.weights_per_layer[n_total_layers-i]);
            println!("delta[delta.len()-1] = {}", delta[delta.len()-1]);
            // Back-propagated (notice the transposed weights) cost derivative w.r.t. the activations at the current layer
            let dC_over_dAL = matmult0(&self.weights_per_layer[n_total_layers-i], &delta[delta.len()-1])?;
            println!("dC_over_dAL = {}", dC_over_dAL);
            // Activation derivative w.r.t. the sum of the weights (since Ω.S[end] == Ω.ŷ then the previous pre-activations are Ω.S[end-1])
            let dAl_over_dSl = self.activation.derivative(&self.weights_x_biases_per_layer[n_total_layers-(i+1)])?;
            println!("dAl_over_dSl = {}", dAl_over_dSl);
            // Chain rule-derived cost derivative w.r.t. the sum of the weights
            let dC_over_dSL = elementwisematmul(&dC_over_dAL, &dAl_over_dSl)?;
            println!("dC_over_dSL = {}", dC_over_dSL);
            // Add to Δ
            delta.push(dC_over_dSL);
        }
        // Calculate the gradients per layer
        // We want ∂C/∂Wˡ = (∂C/∂Sˡ) * (∂Sˡ/∂Wˡ)
        // where: ∂Sˡ/∂Wˡ = Aˡ⁻¹, since: Sˡ = Wˡ*Aˡ⁻¹ + bˡ
        // Then ∂C/∂Wˡ = (∂C/∂Sˡ) * (Aˡ⁻¹)'
        
        println!("n_total_layers = {}", n_total_layers);
        println!("self.n_hidden_layers = {}", self.n_hidden_layers);
        println!("delta.len() = {}", delta.len());
        
        // Δ = reverse(Δ)
        for i in 0..delta.len() {
            let j = delta.len() - (i+1);
            // Outer-product of the error in hidden layer 1 (l_1 x n) and the transpose of the activation at 1 layer below (n x l_0) to yield a gradient matrix corresponding to the weights matrix (l_1 x l_0)
            let x = matmul0t(&delta[j], &self.activations_per_layer[i])?;
            println!("x={}", x);
            self.weights_per_layer[i] = x;
            // Sum-up the errors across n samples in the current hidden layer to calculate the gradients for the bias
            let y = rowsummat(&self.stream, &delta[j])?;
            println!("y = {}", y);
            self.biases_per_layer[i] = y;
        }


        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_backward() -> Result<(), Box<dyn Error>> {
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
        
        network.backpropagation()?;

        Ok(())
    }
}