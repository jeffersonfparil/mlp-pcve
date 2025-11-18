use cudarc::{
    driver::{CudaContext, DriverError, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};

// Notes:
// - needed to install nvrtc: `CONDA_NO_PLUGINS=true conda install --solver=classic cuda-nvrtc`
// - also append the conda library path to LD_LIBRARY_PATH: `export LD_LIBRARY_PATH=${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}`

fn main() {
    println!("Hello, world!");
    // Get a stream for GPU 0
    let ctx = cudarc::driver::CudaContext::new(0).unwrap();
    let stream = ctx.default_stream();

    // copy a rust slice to the device
    let inp = stream.clone_htod(&[1.0f32; 100]).unwrap();

    // or allocate directly
    let mut out = stream.alloc_zeros::<f32>(100).unwrap();
    
    let ptx = cudarc::nvrtc::compile_ptx("
    extern \"C\" __global__ void sin_kernel(float *out, const float *inp, const size_t numel) {
        unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < numel) {
            out[i] = sin(inp[i]);
        }
    }").unwrap();

    // Dynamically load it into the device
    let module = ctx.load_module(ptx).unwrap();
    let sin_kernel = module.load_function("sin_kernel").unwrap();

    let mut builder = stream.launch_builder(&sin_kernel);
    builder.arg(&mut out);
    builder.arg(&inp);
    builder.arg(&100usize);
    unsafe { builder.launch(LaunchConfig::for_num_elems(100)) }.unwrap();

    let out_host: Vec<f32> = stream.clone_dtoh(&out).unwrap();
    assert_eq!(out_host, [1.0; 100].map(|x| (f32::sin(x)*1000000.0).round()/1000000.0));

}
