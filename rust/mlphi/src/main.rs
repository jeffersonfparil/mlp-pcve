use cudarc::driver::DriverError;

mod linalg;
mod activations;
mod costs;
mod network;
mod forward;
mod backward;

fn main() -> Result<(), DriverError> {
    Ok(())
}
