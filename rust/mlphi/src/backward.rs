use crate::linalg::matrix::Matrix;
use crate::network::Network;
use cudarc::driver::CudaSlice;
use std::error::Error;

// To prevent exploding gradients we clamp the gradients to a reasonable range:
const CLAMP_LOWER: f32 = -10_000.0;
const CLAMP_UPPER: f32 = 10_000.0;

impl Network {
    pub fn backpropagation(&mut self) -> Result<(), Box<dyn Error>> {
        // Cost gradients with respect to (w.r.t.) the weights: ∂C/∂Wˡ = (∂C/∂Aᴸ) * (∂Aᴸ/∂Sˡ) * (∂Sˡ/∂Wˡ)
        // Starting with the output layer down to the first hidden layer
        // Cost derivative with respect to (w.r.t.) to the activations at the output layer
        let dc_over_da = self.cost.derivative(&self.predictions, &self.targets)?;
        // Activation derivative w.r.t. the sum of the weights (i.e. pre-activation values) at the output layer which is just 1.00 (linear activation) because this is a regression and not a classification network
        let da_over_ds = 1.00f32;
        // Error for the output layer (cost derivative w.r.t. the sum of the weights via chain rule): element-wise product of the cost derivatives and activation derivatives
        let dc_over_ds = dc_over_da.scalarmatmul(da_over_ds)?;
        let mut delta: Vec<Matrix> = vec![dc_over_ds];
        // Now let us proceed from the last layer to the first hidden layer (i.e. just before the input layer)
        let n_total_layers = self.weights_per_layer.len();
        for i in 1..(self.n_hidden_layers + 1) {
            // Back-propagated (notice the transposed weights) cost derivative w.r.t. the activations at the current layer
            let dc_over_da =
                self.weights_per_layer[n_total_layers - i].matmult0(&delta[delta.len() - 1])?;
            // Activation derivative w.r.t. the sum of the weights (since Ω.S[end] == Ω.ŷ then the previous pre-activations are Ω.S[end-1])
            let da_over_ds = self
                .activation
                .derivative(&self.weights_x_biases_per_layer[n_total_layers - (i + 1)])?;
            // Chain rule-derived cost derivative w.r.t. the sum of the weights
            let dc_over_ds = dc_over_da.elementwisematmul(&da_over_ds)?;
            // Add to Δ
            delta.push(dc_over_ds);
        }
        // Calculate the gradients per layer starting from the first hidden layer
        // We want ∂C/∂Wˡ = (∂C/∂Sˡ) * (∂Sˡ/∂Wˡ)
        // where: ∂Sˡ/∂Wˡ = Aˡ⁻¹, since: Sˡ = Wˡ*Aˡ⁻¹ + bˡ
        // Then ∂C/∂Wˡ = (∂C/∂Sˡ) * (Aˡ⁻¹)' (similar applies to the biases)
        for i in 0..delta.len() {
            let j = delta.len() - (i + 1); // we start with the first hidden layer in Δ, i.e. we need to reverse Δ
            // Outer-product of the error in hidden layer 1 (l_1 x n) and the transpose of the activation at 1 layer below (n x l_0) to yield a gradient matrix corresponding to the weights matrix (l_1 x l_0)
            self.weights_gradients_per_layer[i] = delta[j]
                .matmul0t(&self.activations_per_layer[i])?
                .clamp(CLAMP_LOWER, CLAMP_UPPER)?; // Clamping to prevent exploding gradients
            // Sum-up the errors across n samples in the current hidden layer to calculate the gradients for the bias
            self.biases_gradients_per_layer[i] =
                delta[j].rowsummat()?.clamp(CLAMP_LOWER, CLAMP_UPPER)?; // Clamping to prevent exploding gradients
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::CudaContext;
    #[test]
    fn test_backward() -> Result<(), Box<dyn Error>> {
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
        // Assess the weights at the ith layer
        let i = 1;
        println!(
            "layer {} weights gradients (before backpropagation):\n{}",
            i, network.weights_gradients_per_layer[i],
        );
        network.backpropagation()?;
        println!(
            "layer {} weights gradients (after without forwardpass):\n{}",
            i, network.weights_gradients_per_layer[i],
        );
        // Without prior forward pass all weights become zero because the `weights_x_biases_per_layer` are initialised as all zeroes!
        let s = network.weights_gradients_per_layer[i].summat()?;
        println!("s (without forwardpass) = {}", s);
        assert!(s == 0.0);
        // Reset weights to random values then run with forward pass prior to backpropagation
        for j in 0..(network.n_hidden_layers + 1) {
            let mut a_host = vec![
                0.0f32;
                network.weights_gradients_per_layer[j].n_rows
                    * network.weights_gradients_per_layer[j].n_cols
            ];
            stream.memcpy_dtoh(&network.weights_gradients_per_layer[j].data, &mut a_host)?;
            rand::fill(&mut a_host[..]);
            network.weights_gradients_per_layer[j].data = stream.clone_htod(&a_host)?;
        }
        network.forwardpass()?;
        println!(
            "layer {} weights gradients (after forwardpass):\n{}",
            i, network.weights_gradients_per_layer[i],
        );
        network.backpropagation()?;
        println!(
            "layer {} weights gradients (after backpropagation WITH forwardpass):\n{}",
            i, network.weights_gradients_per_layer[i],
        );
        let s = network.weights_gradients_per_layer[i].summat()?;
        println!("s (with forwardpass) = {}", s);
        assert!(s != 0.0);
        Ok(())
    }
}
