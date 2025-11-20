# README

## Issues and notes:

- Make this more portable, i.e. allowing use of CPU or any type of GPU available in the system
- Current attempt at portability using PackageCompiler.jl and Julia version 1.11 (version 1.12 fails):
    + always compile CUDA with the graphics card you want to use the binary on
    + compile: `julia +1.11 -E 'using Pkg; Pkg.resolve(); using PackageCompiler; create_app("MLPPCVE.jl/", "mlppcve-bin")'`
    + do the following prior to running the binary: `export JULIA_CUDA_USE_COMPAT=false`
    + test run - no arguments show help docs: `./mlppcve-bin/bin/MLPPCVE -h`
    + to run a test example using simulated data: `./mlppcve-bin/bin/MLPPCVE -f example`

## Tests:

```shell
time julia +1.11 --threads=23,1 test/cli_tester.jl 
```

## Interactive development session:

```shell
julia +1.11 --threads=23,1 --load test/interactive_prelude.jl
```

## Compile and test:

```shell
julia +1.11 -E 'using Pkg; Pkg.resolve(); using PackageCompiler; create_app("MLPPCVE.jl/", "mlppcve-bin")'
export JULIA_CUDA_USE_COMPAT=false
./mlppcve-bin/bin/MLPPCVE -h
./mlppcve-bin/bin/MLPPCVE -f example
```