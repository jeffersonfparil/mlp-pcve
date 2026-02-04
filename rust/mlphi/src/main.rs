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
    // TODO: move to its own repo please...
    // TODO: UI... clap again or is there a better alternative?

    Ok(())
}
