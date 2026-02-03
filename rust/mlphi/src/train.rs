use crate::linalg::matrix::Matrix;
use crate::activations::Activation;
use crate::costs::Cost;
use crate::network::Network;
use crate::optimisers::{OptimisationParameters, Optimiser};
use chrono::Utc;
use cudarc::driver::CudaSlice;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use ruviz::core::{Plot, PlottingError};
use ruviz::prelude::LegendPosition;
use std::fmt;
use std::ops::Add;
use std::error::Error;
use std::path::PathBuf;
use std::env::current_dir;
use rayon::prelude::*;
use std::sync::Mutex;

#[derive(Debug, PartialEq)]
enum TrainingError {
    BatchingError(String),
    EpochError(String),
    OtherError(String),
}

/// Implement Error for TrainingError
impl Error for TrainingError {}

/// Implement std::fmt::Display for TrainingError
impl fmt::Display for TrainingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            TrainingError::BatchingError(msg) => write!(f, "Batching error during training: {}", msg),
            TrainingError::EpochError(msg) => write!(f, "Epoch error during training: {}", msg),
            TrainingError::OtherError(msg) => write!(f, "Other error during training: {}", msg),
        }
    }
}

impl From<PlottingError> for TrainingError {
    fn from(err: PlottingError) -> Self {
        TrainingError::OtherError(err.to_string())
    }
}

fn prep_each_hyperparam<T>(param_min_max_step: (T, T, T)) -> Result<Vec<T>, Box<dyn Error>>
where
    T: Copy + PartialOrd + Add<Output = T> + Default + PartialEq,
{
    let (min, max, step) = param_min_max_step;
    if step == T::default() {
        return Err(Box::new(TrainingError::OtherError(
            "Step must be non-zero.".to_string(),
        )));
    }
    // if step > 0 and max < min -> invalid; if step < 0 and max > min -> invalid
    if (step > T::default() && max < min) || (step < T::default() && max > min) {
        return Err(Box::new(TrainingError::OtherError(
            "Invalid range.".to_string(),
        )));
    }
    let mut selection: Vec<T> = Vec::new();
    let mut x = min;
    if step > T::default() {
        while x <= max {
            selection.push(x);
            x = x + step;
        }
    } else {
        while x >= max {
            selection.push(x);
            x = x + step; // step is negative here
        }
    }
    Ok(selection)
}

fn prep_all_hyperparams(
    range_hidden_layers: Option<(usize, usize, usize)>,
    range_hidden_layer_nodes: Option<(usize, usize, usize)>,
    range_dropout_rate: Option<(f32, f32, f32)>,
    range_learning_rate: Option<(f32, f32, f32)>,
    range_n_epochs: Option<(usize, usize, usize)>,
    range_f_patient_epochs: Option<(f32, f32, f32)>,
    range_n_batches: Option<(usize, usize, usize)>,
    selection_activations: Option<Vec<Activation>>,
    selection_costs: Option<Vec<Cost>>,
    selection_optimisers: Option<Vec<Optimiser>>,
) -> Result<Vec<(usize, usize, f32, f32, usize, f32, usize, Activation, Cost, Optimiser)>, Box<dyn Error>> {
    let selection_hidden_layers: Vec<usize> = match range_hidden_layers {
        Some(x) => prep_each_hyperparam(x)?,
        None => prep_each_hyperparam((1, 3, 1))?
    };
    let selection_hidden_layer_nodes: Vec<usize> = match range_hidden_layer_nodes {
        Some(x) => prep_each_hyperparam(x)?,
        None => prep_each_hyperparam((100, 500, 100))?
    };
    let selection_dropout_rates: Vec<f32> = match range_dropout_rate {
        Some(x) => prep_each_hyperparam(x)?,
        None => prep_each_hyperparam((0.0, 0.5, 0.01))?
    };
    let selection_learning_rates: Vec<f32> = match range_learning_rate {
        Some(x) => prep_each_hyperparam(x)?,
        None => prep_each_hyperparam((1e-5, 1e-2, 1e-4))?
    };
    let selection_n_epochs: Vec<usize> = match range_n_epochs {
        Some(x) => prep_each_hyperparam(x)?,
        None => prep_each_hyperparam((5, 10, 1))?
    };
    let selection_f_patient_epochs: Vec<f32> = match range_f_patient_epochs {
        Some(x) => prep_each_hyperparam(x)?,
        None => prep_each_hyperparam((0.5, 1.0, 0.5))?
    };
    let selection_n_batches: Vec<usize> = match range_n_batches {
        Some(x) => prep_each_hyperparam(x)?,
        None => prep_each_hyperparam((1, 3, 1))?
    };
    let selection_activations: Vec<Activation> = match selection_activations {
        Some(x) => x,
        None => vec![Activation::ReLU],
    };
    let selection_costs: Vec<Cost> = match selection_costs {
        Some(x) => x,
        None => vec![Cost::MSE],
    };
    let selection_optimisers: Vec<Optimiser> = match selection_optimisers {
        Some(x) => x,
        None => vec![Optimiser::Adam, Optimiser::AdamMax, Optimiser::GradientDescent],
    };
    let mut param_combinations: Vec<(usize, usize, f32, f32, usize, f32, usize, Activation, Cost, Optimiser)> = Vec::new();
    for n_hidden_layers in &selection_hidden_layers {
        for n_hidden_nodes in &selection_hidden_layer_nodes {
            for dropout_rate in &selection_dropout_rates {
                for learning_rate in &selection_learning_rates {
                    for n_epochs in &selection_n_epochs {
                        for f_patient_epochs in &selection_f_patient_epochs {
                            for n_batches in &selection_n_batches {
                                for activation in &selection_activations {
                                    for cost in &selection_costs {
                                        for optimiser in &selection_optimisers {
                                            param_combinations.push((
                                                *n_hidden_layers,
                                                *n_hidden_nodes,
                                                *dropout_rate,
                                                *learning_rate,
                                                *n_epochs,
                                                *f_patient_epochs,
                                                *n_batches,
                                                activation.clone(),
                                                cost.clone(),
                                                optimiser.clone(),
                                            ));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(param_combinations)
}

impl Network {
    pub fn shufflesplit(self: &Self, n_batches: usize) -> Result<Vec<Vec<usize>>, Box<dyn Error>> {
        let n: usize = self.targets.n_cols; // number of observations
        if n_batches == 0 {
            return Err(Box::new(TrainingError::BatchingError(
                "Number of batches must be greater than zero.".to_string(),
            )));
        }
        if n_batches > n {
            return Err(Box::new(TrainingError::BatchingError(
                "Number of batches cannot be greater than number of observations.".to_string(),
            )));
        }
        let mut rng = ChaCha12Rng::seed_from_u64(self.seed as u64);
        let mut indexes: Vec<usize> = (0..n).collect();
        indexes.shuffle(&mut rng);
        let batch_size = (n + n_batches - 1) / n_batches;
        let mut col_indexes_per_batch: Vec<Vec<usize>> = Vec::new();
        for i in 0..n_batches {
            let start = i * batch_size;
            let end = match (i + 1) * batch_size {
                x if x > n => n,
                x => x,
            };
            if start >= end {
                break;
            }
            col_indexes_per_batch.push(indexes[start..end].to_vec());
        }
        Ok(col_indexes_per_batch)
    }

    pub fn predict(self: &mut Self) -> Result<(), Box<dyn Error>> {
        // Different from forwardpass in that it does not apply dropout.
        let n = self.n_hidden_layers;
        for i in 0..n {
            let weights_x_activations =
                self.weights_per_layer[i].matmul(&self.activations_per_layer[i])?;
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
        Ok(())
    }

    pub fn train_per_batch(
        self: &mut Self,
        optimisation_parameters: &mut OptimisationParameters,
    ) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
        let mut epochs: Vec<f64> = Vec::new();
        let mut costs: Vec<f64> = Vec::new();
        let n_patient_epochs = (optimisation_parameters.f_patient_epochs * optimisation_parameters.n_epochs as f32).ceil() as usize;
        for epoch in 0..optimisation_parameters.n_epochs {
            self.forwardpass()?;
            self.backpropagation()?;
            self.optimise(optimisation_parameters)?;
            self.predict()?;
            let e: f64 = epoch as f64;
            let c: f64 = (self.cost.cost(&self.predictions, &self.targets)?.summat()? as f64)
                / (self.targets.n_cols as f64);
            epochs.push(e);
            costs.push(c);
            // Early stopping check, i.e. stop if no improvement in cost after n_patient_epochs
            if (epoch > n_patient_epochs) && (costs[epoch] >= costs[epoch - n_patient_epochs]) {
                // println!("Early stopping at epoch {}", epoch);
                break;
            }
        }
        Ok((epochs, costs))
    }

    pub fn train(
        self: &mut Self,
        optimisation_parameters: &OptimisationParameters,
        verbose: bool,
    ) -> Result<f32, Box<dyn Error>> {
        self.check_dimensions()?;
        if optimisation_parameters.n_epochs == 0 {
            return Err(Box::new(TrainingError::EpochError(
                "Number of epochs must be greater than zero.".to_string(),
            )));
        }
        if optimisation_parameters.n_batches == 0 {
            return Err(Box::new(TrainingError::BatchingError(
                "Number of batches must be greater than zero.".to_string(),
            )));
        }
        let stream = self.targets.data.context().default_stream();
        let (epochs, costs): (Vec<Vec<f64>>, Vec<Vec<f64>>) = if optimisation_parameters.n_batches
            == 1
        {
            // Only one batch, train on the whole dataset
            let mut params = optimisation_parameters.clone();
            let (epochs, costs) = self.train_per_batch(&mut params)?;
            (vec![epochs], vec![costs])
        } else {
            // Multiple batches, split the dataset then average the parameters after training on each batch
            let col_indexes_per_batch: Vec<Vec<usize>> =
                self.shufflesplit(optimisation_parameters.n_batches)?;
            let mut networks_per_batch: Vec<Network> =
                Vec::with_capacity(optimisation_parameters.n_batches);
            for col_indexes in col_indexes_per_batch {
                // indexes for each batch, i.e. for observations
                let network = self.slice(&col_indexes)?;
                networks_per_batch.push(network);
            }
            let epochs: Mutex<Vec<Vec<f64>>> = Mutex::new(Vec::new());
            let costs: Mutex<Vec<Vec<f64>>> = Mutex::new(Vec::new());
            networks_per_batch
                .par_iter_mut()
                .enumerate()
                .for_each(|(i, network)| {
                    if verbose {
                        println!(
                            "Training on batch {} with {} observations.",
                            i, network.targets.n_cols
                        );
                    }
                    let mut params = optimisation_parameters.clone();
                    let result = network.train_per_batch(&mut params);
                    match result {
                        Ok((epochs_batch, costs_batch)) => {
                            epochs.lock().unwrap().push(epochs_batch);
                            costs.lock().unwrap().push(costs_batch);
                        }
                        Err(e) => {
                            // Skip the batch
                            eprintln!("Error training on batch {}: {}", i, e);
                        }
                    }
                });
            // Merge the parameters from each batch network back into the original network via simple averaging with a better method
            for i in 0..self.n_hidden_layers + 1 {
                let zeros_weights_host: Vec<f32> =
                    vec![0.0; self.weights_per_layer[i].n_rows * self.weights_per_layer[i].n_cols];
                let zeros_biases_host: Vec<f32> =
                    vec![0.0; self.biases_per_layer[i].n_rows * self.biases_per_layer[i].n_cols];
                let zeros_weights_dev: CudaSlice<f32> = stream.clone_htod(&zeros_weights_host)?;
                let zeros_biases_dev: CudaSlice<f32> = stream.clone_htod(&zeros_biases_host)?;
                let mut summed_weights = Matrix::new(
                    zeros_weights_dev,
                    self.weights_per_layer[i].n_rows,
                    self.weights_per_layer[i].n_cols,
                )?;
                let mut summed_biases = Matrix::new(
                    zeros_biases_dev,
                    self.biases_per_layer[i].n_rows,
                    self.biases_per_layer[i].n_cols,
                )?;
                for network in &networks_per_batch {
                    summed_weights =
                        summed_weights.elementwisematadd(&network.weights_per_layer[i])?;
                    summed_biases =
                        summed_biases.elementwisematadd(&network.biases_per_layer[i])?;
                }
                self.weights_per_layer[i] =
                    summed_weights.scalarmatmul(1.00 / networks_per_batch.len() as f32)?;
                self.biases_per_layer[i] =
                    summed_biases.scalarmatmul(1.00 / networks_per_batch.len() as f32)?;
            }
            // Update predictions using the merged parameters
            self.predict()?;
            // Return epochs, costs
            (
                epochs.into_inner().unwrap(),
                costs.into_inner().unwrap(),
            )
        };
        // Assess cost after training
        let final_cost = self.cost.cost(&self.predictions, &self.targets)?;
        let final_cost_value = final_cost.summat()? as f32 / self.targets.n_cols as f32;
        if verbose {
            // Plot loss curve
            let dir: PathBuf = current_dir()?;
            let fname_svg = &format!(
                "{}/Loss_curve-{:?}-E{}-FPE{}-LR{}-{:?}-N{}-Time{}.svg",
                dir.display(),
                optimisation_parameters.optimiser,
                optimisation_parameters.n_epochs,
                optimisation_parameters.f_patient_epochs,
                optimisation_parameters.learning_rate,
                self.activation,
                self.n_hidden_layers,
                Utc::now().format("%Y%m%d%H%M%S")
            );
            let mut ylabel = String::from("Cost");
            ylabel.push_str(&format!(
                " ({:?}; {:?})",
                self.cost, optimisation_parameters.optimiser
            ));
            let mut plot_vec = vec![
                Plot::new()
                    .title("Training Cost over Epochs")
                    .legend_position(LegendPosition::Best)
                    .xlabel("Epochs")
                    .ylabel(&ylabel)
                    .line(&epochs[0], &costs[0])
                    .label("Batch 0")
                    .size(4.0, 3.0),
            ];
            for i in 1..optimisation_parameters.n_batches {
                plot_vec[0] = plot_vec[0]
                    .clone()
                    .line(&epochs[i], &costs[i])
                    .label(&format!("Batch {}", i));
            }
            // plot_vec[0].clone().save(fname_png)?;
            plot_vec[0].clone().export_svg(fname_svg)?;
            // Messages
            println!("Final cost after training: {}", final_cost_value);
            println!("Find the loss curve saved as: {}", fname_svg);
        }
        Ok(final_cost_value)
    }

    pub fn hyperoptimise(
        self: &mut Self,
        range_hidden_layers: Option<(usize, usize, usize)>,
        range_hidden_layer_nodes: Option<(usize, usize, usize)>,
        range_dropout_rate: Option<(f32, f32, f32)>,
        range_learning_rate: Option<(f32, f32, f32)>,
        range_n_epochs: Option<(usize, usize, usize)>,
        range_f_patient_epochs: Option<(f32, f32, f32)>,
        range_n_batches: Option<(usize, usize, usize)>,
        selection_activations: Option<Vec<Activation>>,
        selection_costs: Option<Vec<Cost>>,
        selection_optimisers: Option<Vec<Optimiser>>,
        verbose: bool,
    ) -> Result<(), Box<dyn Error>> {
        self.check_dimensions()?;
        let param_combinations = prep_all_hyperparams(
            range_hidden_layers,
            range_hidden_layer_nodes,
            range_dropout_rate,
            range_learning_rate,
            range_n_epochs,
            range_f_patient_epochs,
            range_n_batches,
            selection_activations,
            selection_costs,
            selection_optimisers,
        )?;
        // Hyper-parameter optimisations
        let mut results: Vec<(
            usize,
            usize,
            f32,
            f32,
            usize,
            f32,
            usize,
            Activation,
            Cost,
            Optimiser,
            Result<f32, Box<dyn Error>>,
        )> = Vec::new();
        for p in param_combinations {
            let (
                n_hidden_layers,
                n_hidden_nodes,
                dropout_rate,
                learning_rate,
                n_epochs,
                f_patient_epochs,
                n_batches,
                activation,
                cost,
                optimiser,
            ) = p;
            // Create a new instance of the network with the current hyper-parameters
            let mut network = Network::new(
                &self.activations_per_layer[0].data.context().default_stream(),
                self.activations_per_layer[0].clone(),
                self.targets.clone(),
                n_hidden_layers,
                vec![n_hidden_nodes; n_hidden_layers],
                vec![dropout_rate; n_hidden_layers],
                self.seed,
            )?;
            network.activation = activation.clone();
            network.cost = cost.clone();
            let mut optimisation_parameters = OptimisationParameters::new(&network)?;
            optimisation_parameters.learning_rate = learning_rate;
            optimisation_parameters.n_epochs = n_epochs;
            optimisation_parameters.f_patient_epochs = f_patient_epochs;
            optimisation_parameters.n_batches = n_batches;
            optimisation_parameters.optimiser = optimiser.clone();
            // Train the network with the current hyper-parameters
            let result = network.train(&optimisation_parameters, verbose);
            // Store the result of the training
            results.push((
                n_hidden_layers,
                n_hidden_nodes,
                dropout_rate,
                learning_rate,
                n_epochs,
                f_patient_epochs,
                n_batches,
                activation.clone(),
                cost.clone(),
                optimiser.clone(),
                result,
            ));
        }
        // for n_hidden_layers in selection_hidden_layers.clone() {
        //     for n_hidden_nodes in selection_hidden_layer_nodes.clone() {
        //         for dropout_rate in selection_dropout_rates.clone() {
        //             for learning_rate in selection_learning_rates.clone() {
        //                 for n_epochs in selection_n_epochs.clone() {
        //                     for f_patient_epochs in selection_f_patient_epochs.clone() {
        //                         for n_batches in selection_n_batches.clone() {
        //                             for activation in selection_activations.clone() {
        //                                 for cost in selection_costs.clone() {
        //                                     for optimiser in selection_optimisers.clone() {
        //                                         // Create a new instance of the network with the current hyper-parameters
        //                                         let mut network = Network::new(
        //                                             &self.activations_per_layer[0].data.context().default_stream(),
        //                                             self.activations_per_layer[0].clone(),
        //                                             self.targets.clone(),
        //                                             n_hidden_layers,
        //                                             vec![n_hidden_nodes; n_hidden_layers],
        //                                             vec![dropout_rate; n_hidden_layers],
        //                                             self.seed,
        //                                         )?;
        //                                         network.activation = activation.clone();
        //                                         network.cost = cost.clone();
        //                                         let mut optimisation_parameters = OptimisationParameters::new(&network)?;
        //                                         optimisation_parameters.learning_rate = learning_rate;
        //                                         optimisation_parameters.n_epochs = n_epochs;
        //                                         optimisation_parameters.f_patient_epochs = f_patient_epochs;
        //                                         optimisation_parameters.n_batches = n_batches;
        //                                         optimisation_parameters.optimiser = optimiser.clone();
        //                                         // Train the network with the current hyper-parameters
        //                                         let result = network.train(&optimisation_parameters, verbose);
        //                                         // Store the result of the training
        //                                         results.push((
        //                                             n_hidden_layers,
        //                                             n_hidden_nodes,
        //                                             dropout_rate,
        //                                             learning_rate,
        //                                             n_epochs,
        //                                             f_patient_epochs,
        //                                             n_batches,
        //                                             activation.clone(),
        //                                             cost.clone(),
        //                                             optimiser.clone(),
        //                                             result,
        //                                         ));
        //                                     }
        //                                 }
        //                             }
        //                         }
        //                     }
        //                 }
        //             }
        //         }
        //     }
        // }
        // Print the results
        println!("Hyper-parameter Optimisation Results:");
        println!("| Hidden_Layers | Hidden_Nodes | Dropout_Rate | Learning_Rate | Epochs | Patient_Epochs | Batches | Activation | Cost | Optimiser | Final_Cost |");
        for (n_hidden_layers, n_hidden_nodes, dropout_rate, learning_rate, n_epochs, f_patient_epochs, n_batches, activation, cost, optimiser, result) in results {
            match result {
                Ok(final_cost) => {
                    println!("| {:13} | {:12} | {:12.4} | {:13.6} | {:6} | {:14} | {:7} | {:?} | {:?} | {:?} | {:10.6} |", n_hidden_layers, n_hidden_nodes, dropout_rate, learning_rate, n_epochs, f_patient_epochs, n_batches, activation, cost, optimiser, final_cost);
                }
                Err(e) => {
                    println!("| {:13} | {:12} | {:12.4} | {:13.6} | {:6} | {:14} | {:7} | {:?} | {:?} | {:?} | Error: {} |", n_hidden_layers, n_hidden_nodes, dropout_rate, learning_rate, n_epochs, f_patient_epochs, n_batches, activation, cost, optimiser, e);
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::{CudaContext, CudaSlice};
    #[test]
    fn test_train() -> Result<(), Box<dyn Error>> {
        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let n: usize = 10_123; // number of observations
        let p: usize = 314; // number of input features
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
            &stream,
            input_matrix,
            output_matrix,
            n_hidden_layers,
            vec![n_hidden_layer_nodes; n_hidden_layers],
            vec![0.0f32; n_hidden_layers],
            42,
        )?;
        let mut optimisation_parameters = OptimisationParameters::new(&network)?;
        println!("Network:\n{}\n\n", network);
        println!("Optimisation Parameters:\n{}\n\n", optimisation_parameters);
        // optimisation_parameters.learning_rate = 0.00001f32;
        // optimisation_parameters.optimiser = Optimiser::GradientDescent;
        optimisation_parameters.optimiser = Optimiser::Adam;
        // optimisation_parameters.optimiser = Optimiser::AdamMax;

        // Tests

        let indexes: Vec<Vec<usize>> = network.shufflesplit(5)?;
        // println!("indexes: {:?}", indexes);
        println!("Number of batches: {:?}", indexes.len());
        let mut total_len: usize = 0;
        for i in 0..indexes.len() {
            println!(
                "indexes[{}]: [{}, {}, ...{}] length: {:?}",
                i,
                indexes[i][0],
                indexes[i][1],
                indexes[i][indexes[i].len() - 1],
                indexes[i].len()
            );
            total_len += indexes[i].len();
        }
        println!("Total length: {:?}", total_len);
        assert!(total_len == network.targets.n_cols);

        let stream = network.targets.data.context().default_stream();
        let mut a_host = vec![0.0f32; network.targets.n_rows * network.targets.n_cols];
        stream.memcpy_dtoh(&network.targets.data, &mut a_host)?;
        println!(
            "targets: [{}, {}, {}, ..., {}]",
            a_host[0],
            a_host[1],
            a_host[2],
            a_host[a_host.len() - 1]
        );

        stream.memcpy_dtoh(&network.predictions.data, &mut a_host)?;
        println!(
            "predictions (before predict()): [{}, {}, {}, ..., {}]",
            a_host[0],
            a_host[1],
            a_host[2],
            a_host[a_host.len() - 1]
        );
        network.train_per_batch(&mut optimisation_parameters)?;
        optimisation_parameters.n_epochs = 5;
        optimisation_parameters.n_batches = 2;
        network.train(&mut optimisation_parameters, true)?;

        // Hyper-parameter optimisations
        let range_hidden_layers = Some((1, 2, 1));
        let range_hidden_layer_nodes = Some((100, 100, 100));
        let range_dropout_rate = Some((0.0, 0.0, 0.1));
        let range_learning_rate = Some((0.0001, 0.0001, 0.0001));
        let range_n_epochs = Some((10, 10, 10));
        let range_f_patient_epochs = Some((0.5, 0.5, 0.5));
        let range_n_batches = Some((2, 2, 2));
        // let selection_activations = Some(vec![Activation::ReLU]);
        // let selection_costs = Some(vec![Cost::MSE]);
        // let selection_optimisers = Some(vec![Optimiser::Adam]);
        network.hyperoptimise(
            range_hidden_layers,
            range_hidden_layer_nodes,
            range_dropout_rate,
            range_learning_rate,
            range_n_epochs,
            range_f_patient_epochs,
            range_n_batches,
            None,
            None,
            None,
            true,
        )?;

        Ok(())
    }
}
