use cudarc::driver::DriverError;

mod activations;
mod backward;
mod costs;
mod forward;
mod linalg;
mod network;
mod optimisers;

fn main() -> Result<(), DriverError> {
    Ok(())
}
