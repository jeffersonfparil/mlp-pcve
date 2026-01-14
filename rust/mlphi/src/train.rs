use crate::linalg::matrix::Matrix;
use crate::network::Network;
use std::error::Error;

impl Network {
    pub fn split(self: &Self, ratio: f32) -> Result<(Self, Self), Box<dyn Error>> {
        Ok((self.clone(), self.clone()))
    }
    pub fn predict(self: &Self) -> Result<(), Box<dyn Error>> {
        Ok(())
    } 
    pub fn train(self: &Self) -> Result<(), Box<dyn Error>> {
        Ok(())
    } 
    pub fn optim(self: &Self) -> Result<(), Box<dyn Error>> {
        Ok(())
    } 
}