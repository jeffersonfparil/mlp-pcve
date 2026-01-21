use crate::linalg::matrix::Matrix;
use crate::network::Network;
use crate::optimisers::{Optimiser, OptimiserParameters};
use std::error::Error;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;

impl Network {
    pub fn shufflesplit(self: &Self, n_batches: usize) -> Result<Vec<Vec<usize>>, Box<dyn Error>> {
        let n: usize = self.targets.n_cols;
        // let k: usize = self.targets.n_rows;
        let mut rng = ChaCha12Rng::seed_from_u64(self.seed as u64);
        let mut indexes: Vec<usize> = (0..n).collect();
        indexes.shuffle(&mut rng);
        let batch_size = (n + n_batches - 1) / n_batches;
        let mut batches: Vec<Vec<usize>> = Vec::new();
        for i in 0..n_batches {
            let start = i * batch_size;
            let end = match (i + 1) * batch_size {
                x if x > n => n,
                x => x,
            };
            if start >= end {
                break;
            }
            batches.push(indexes[start..end].to_vec());
        }
        Ok(batches)
    }

    pub fn predict(self: &mut Self) -> Result<(), Box<dyn Error>> {
        // Different from forwardpass in that it does not apply dropout.
        let n = self.n_hidden_layers;
        for i in 0..n {
            let weights_x_activations = self.weights_per_layer[i].matmul(&self.activations_per_layer[i])?;
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

        // TODO: prevent overflows in the output layer as we are getting infinities and NaNs!
        // Note: values blowing here likely due to the optimisers, i.e. GradientDescent and AdamMax
        // producing too large updates to weights and biases. Go check those out!!!

        Ok(())
    }


    pub fn train(self: &Self) -> Result<(), Box<dyn Error>> {
        Ok(())
    } 
    pub fn optim(self: &Self) -> Result<(), Box<dyn Error>> {
        Ok(())
    } 
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
    use crate::network::{printcost, printpredictions, printactivations, printweights, printbiases, printweightsgradients, printbiasesgradients};
    use ruviz::prelude::*;
    #[test]
    fn test_train() -> Result<(), Box<dyn Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let n: usize = 123; // number of observations
        let p: usize = 20; // number of input features
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
            stream,
            input_matrix,
            output_matrix,
            n_hidden_layers,
            vec![n_hidden_layer_nodes; n_hidden_layers],
            vec![0.0f32; n_hidden_layers],
            42,
        )?;
        println!("Network:\n{}\n\n", network);
        let mut optimiser_parameters = OptimiserParameters::new(&network)?;
        // optimiser_parameters.learning_rate = 0.00001f32;
        let optimiser = Optimiser::GradientDescent;
        // let optimiser = Optimiser::Adam;
        // let optimiser = Optimiser::AdamMax;



        // Tests


        let indexes: Vec<Vec<usize>> = network.shufflesplit(5)?;
        // println!("indexes: {:?}", indexes);
        println!("Number of batches: {:?}", indexes.len());
        let mut total_len: usize = 0;
        for i in 0..indexes.len() {
            println!("indexes[{}]: [{}, {}, ...{}] length: {:?}", i, indexes[i][0], indexes[i][1], indexes[i][indexes[i].len()-1], indexes[i].len());
            total_len += indexes[i].len();
        }
        println!("Total length: {:?}", total_len);
        assert!(total_len == network.targets.n_cols);


        let mut a_host =
            vec![0.0f32; network.targets.n_rows * network.targets.n_cols];

        network
            .stream
            .memcpy_dtoh(&network.targets.data, &mut a_host)?;
        println!(
            "targets: [{}, {}, {}, ..., {}]",
            a_host[0],
            a_host[1],
            a_host[2],
            a_host[a_host.len() - 1]
        );

        network
            .stream
            .memcpy_dtoh(&network.predictions.data, &mut a_host)?;
        println!(
            "predictions (before predict()): [{}, {}, {}, ..., {}]",
            a_host[0],
            a_host[1],
            a_host[2],
            a_host[a_host.len() - 1]
        );

        // fn printpreds(network: &Network) -> Result<(), Box<dyn Error>> {
        //     let mut a_host =
        //         vec![0.0f32; network.predictions.n_rows * network.predictions.n_cols];
        //     network
        //         .stream
        //         .memcpy_dtoh(&network.predictions.data, &mut a_host)?;
        //     println!(
        //         "predictions: [{}, {}, {}, ..., {}]",
        //         a_host[0],
        //         a_host[1],
        //         a_host[2],
        //         a_host[a_host.len() - 1]
        //     );
        //     Ok(())
        // }

        let mut epochs: Vec<f64> = Vec::new();
        let mut costs: Vec<f64> = Vec::new();
        for epoch in 0..10 {
            println!("\n=============================================");
            println!("Epoch {}", epoch + 1);
            network.forwardpass()?;
            network.backpropagation()?;
            optimiser.optimise(&mut network, &mut optimiser_parameters)?;
            network.predict()?;
            let layer: usize = 1;
            println!("Optimiser: {:?}", optimiser);
            printcost(&network)?;
            printpredictions(&network)?;
            printactivations(&network, layer)?;
            printweights(&network, layer)?;
            printbiases(&network, layer)?;
            printweightsgradients(&network, layer)?;
            printbiasesgradients(&network, layer)?;
            let e: f64 = epoch as f64;
            let c: f64 = (
                    network
                        .cost
                        .cost(&network.predictions, &network.targets)?
                        .summat(&network.stream)? as f64
                ) /
                (network.targets.n_cols as f64);
            epochs.push(e);
            costs.push(c);
            println!("=============================================\n");
        }
        // Plot::new()
        //     .line(&epochs, &costs)
        //     .title("Training Cost over Epochs")
        //     .save("test.png")?;

        
       
        
        Ok(())
    }
}

