use std::error::Error;
use rand::prelude::*;
use rand_chacha::ChaCha12Rng;
use cudarc::driver::{CudaContext, CudaStream, CudaSlice};
use crate::linalg::matrix::{Matrix, MatrixError};
use crate::linalg::activations;
use crate::linalg::costs;
use crate::network::Network;

impl Network{
    pub fn forwardpass(
        &mut self,
    ) -> Result<(), Box<dyn Error>> {
        let mut rng = ChaCha12Rng::seed_from_u64(self.seed as u64);
        let x = (0..100).choose_multiple(&mut rng, 10);
        println!("x={:?}", x);

        // TODO //
        for i in 0..self.n_hidden_layers {
            ()
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward() -> Result<(), Box<dyn Error>> {
        let mut network: Network = Network::new(
            10_000,
            1_000,
            1,
            10,
            vec![256; 10],
            vec![0.0f32; 10],
            42,
        )?;
        network.forwardpass()?;
        println!("network: {}", network);


        Ok(())
    }
}
