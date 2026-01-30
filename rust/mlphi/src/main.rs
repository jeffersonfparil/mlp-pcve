use cudarc::driver::DriverError;

mod activations;
mod backward;
mod costs;
mod forward;
mod io;
mod linalg;
mod network;
mod optimisers;
mod train;

fn main() -> Result<(), DriverError> {
    Ok(())
}
