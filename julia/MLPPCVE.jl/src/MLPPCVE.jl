module MLPPCVE

using Random, LinearAlgebra, CUDA, JLD2, ArgParse

# TODO: 
# Make this more portable, i.e. allowing use of CPU or any type of GPU available in the system
# Current attempt at portability using PackageCompiler.jl and Julia version 1.11 (version 1.12 fails):
# - always compile CUDA with the graphics card you want to use the binary on
# - compile: `julia +1.11 -E 'using Pkg; Pkg.resolve(); using PackageCompiler; create_app("MLPPCVE.jl/", "mlppcve-bin")'`
# - do the following prior to running the binary: `export JULIA_CUDA_USE_COMPAT=false`
# - test run - no arguments show help docs: `./mlppcve-bin/bin/MLPPCVE`

# (0) Helper functions
include("helpers.jl")
# (1) Activation functions
include("activations.jl")
# (2) Cost functions and their derivatives
include("costs.jl")
# (3) Forward pass and backpropagation including the Network struct and initialisation
include("forwardbackward.jl")
# (4) Optimisers
include("optimisers.jl")
# (5) Training and validation
include("traintest.jl")
# (6) Input and output
include("io.jl")
# (7) Main function
include("main.jl")

export Network, init, mean, var, cov, std, sampleNormal, drawreplacenot, metrics, simulate
export Trial, squarest, levenshteindistance, isfuzzymatch, inittrial, simulatelineartrait!, simulatenonlineartrait!, simulatetrial
export writetrial, readtrial
export linear,
    linear_derivative,
    sigmoid,
    sigmoid_derivative,
    hyperbolictangent,
    hyperbolictangent_derivative,
    relu,
    relu_derivative,
    leakyrelu,
    leakyrelu_derivative
export MSE, MSE_derivative, MAE, MAE_derivative, HL, HL_derivative
export forwardpass!, backpropagation!
export gradientdescent!, Adam!, AdamMax!
export splitdata, predict, train, optim
export main

# Testing
# TODO: 
#   - improve argument parsing
#   - add trait selection
function args_parser()
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--fname", "-f"
            help = "Filename of the trial data saved as a delimited text file"
            arg_type = String
        "--delimiter", "-d"
            help = "Text delimiter used in the trial data file"
            arg_type = String
            default = ","
        "--n-batches", "-b"
            help = "Number of batches to divide the data to reduced memory footprint and for cross-validation during training. This must be ≥ 2."
            arg_type = Int
            default = 2
        "--expected-labels", "-L"
            help = ""
            arg_type = String
            default = "years,seasons,harvests,sites,replications,entries,populations,blocks,rows,cols"
        "--traits", "-T"
            help = ""
            arg_type = String
            default = ""
        "--opt-n-hidden-layers"
            help = ""
            arg_type = String
            default = "0,1,2,3"
        "--opt-n-nodes-per-hidden-layer"
            help = ""
            arg_type = String
            default = "128,256"
        "--opt-dropout-per-hidden-layer"
            help = ""
            arg_type = String
            default = "0.0"
        "--opt-n-epochs"
            help = ""
            arg_type = String
            default = "1000"
        "--opt-n-burnin-epochs"
            help = ""
            arg_type = String
            default = "10,100"
        "--opt-n-patient-epochs"
            help = ""
            arg_type = String
            default = "5"
        "--opt-activation-functions", "-F"
            help = "Activiation functions. Select from:, \"relu\", \"leakyrelu\", \"linear\", \"sigmoid\", and \"hyperbolictangent\"."
            arg_type = String
            default = "relu,leakyrelu"
        "--opt-cost-functions", "-C"
            help = "Cost functions. Select from:, \"MSE\", \"MAE\", and \"HL\"."
            arg_type = String
            default = "MSE"
        "--opt-optimisers"
            help = ""
            arg_type = String
            default = "Adam"
        "--learn-rate", "-n"
            help = "Learning rate (step size) used to scale the parameter update (η::Float64 = 0.001)"
            arg_type = Float64
            default = 0.001
        "--decay-rate-1", "-b"
            help = "Exponential decay rate for the first moment (β₁::Float64 = 0.9)"
            arg_type = Float64
            default = 0.9
        "--decay-rate-2", "-B"
            help = "Exponential decay rate for the second moment (β₂::Float64 = 0.999)"
            arg_type = Float64
            default = 0.999
        "--epsilon", "-e"
            help = "Numerical stability constant, i.e. a small number added to the denominator during optimisation to prevent singularities (ϵ::Float64 = 1e-8)"
            arg_type = Float64
            default = 1e-8
        "--n-threads", "-t"
            help = ""
            arg_type = Int64
            default = 2
        "--seed", "-s"
            help = ""
            arg_type = Int64
            default = 42
        "--output-prefix", "-o"
            help = ""
            arg_type = String
            default = "mlppcve"
        "--silent"
            help = ""
            action = :store_true
    end
    args = parse_args(s)
    @show args
    strip_these_chars = ['{', '}', '\n', ' ', '\t']
    # Instantiate output dictionary of parsed aruguments
    args_output = Dict()
    # Input file
    args_output["fname"] = if args["fname"] == "example"
        # Use simulated trial data as an example run
        writetrial(simulatetrial(verbose=false))
    else
        args["fname"]
    end
    args_output["delimiter"] = args["delimiter"]
    args_output["n-batches"] = args["n-batches"]
    args_output["expected-labels-A"] = String.(split(strip(args["expected-labels"], strip_these_chars), ","))
    args_output["traits"] = String.(split(strip(args["traits"], strip_these_chars), ","))
    # Vectors of Int64
    for k in [
        "opt-n-hidden-layers",
        "opt-n-nodes-per-hidden-layer",
        "opt-n-epochs",
        "opt-n-burnin-epochs",
        "opt-n-patient-epochs",
    ]
        args_output[k] = begin
            a = []
            for b in split(strip(args[k], strip_these_chars), ",")
                try
                    push!(a, parse(Int64, b))
                catch
                    error("Cannot parse `--$k` as Int64: $(args[k])")
                end
            end
            a
        end
    end
    # Vectors of Float64
    for k in [
        "opt-dropout-per-hidden-layer",
    ]
        args_output[k] = begin
            a = []
            for b in split(strip(args[k], strip_these_chars), ",")
                try
                    push!(a, parse(Float64, b))
                catch
                    error("Cannot parse `--$k` as Float64: $(args[k])")
                end
            end
            a
        end
    end
    # Vectors of Functions, i.e. activation and cost functions
    for k in [
        "opt-activation-functions",
        "opt-cost-functions",
    ]
        args_output[k] = begin
            a = []
            for b in split(strip(args[k], strip_these_chars), ",")
                f = if b == "relu"
                    push!(a, Dict(:F => relu, :∂F => relu_derivative))
                elseif b == "leakyrelu"
                    push!(a, Dict(:F => leakyrelu, :∂F => leakyrelu_derivative))
                elseif b == "linear"
                    push!(a, Dict(:F => linear, :∂F => linear_derivative))
                elseif b == "sigmoid"
                    push!(a, Dict(:F => sigmoid, :∂F => sigmoid_derivative))
                elseif b == "hyperbolictangent"
                    push!(a, Dict(:F => hyperbolictangent, :∂F => hyperbolictangent_derivative))
                elseif b == "MSE"
                    push!(a, Dict(:C => MSE, :∂C => MSE_derivative))
                elseif b == "MAE"
                    push!(a, Dict(:C => MAE, :∂C => MAE_derivative))
                elseif b == "HL"
                    push!(a, Dict(:C => HL, :∂C => HL_derivative))
                else
                    error("Unrecgonised functions: $(args[k])")
                end
            end
            a
        end
    end
    args_output["opt-optimisers"] = begin
        optimisers = String.(split(strip(args["opt-optimisers"], strip_these_chars), ","))
        for optimiser in optimisers
            if optimiser ∉ ["GD", "Adam", "AdamMax"]
                error("Unrecognised optimiser: $optimiser. Please choose from \"GD\", \"Adam\", and \"AdamMax\".")
            end
        end
        optimisers
    end
    args_output["learn-rate"] = args["learn-rate"]
    args_output["decay-rate-1"] = args["decay-rate-1"]
    args_output["decay-rate-2"] = args["decay-rate-2"]
    args_output["epsilon"] = args["epsilon"]
    args_output["n-threads"] = args["n-threads"]
    args_output["seed"] = args["seed"]
    args_output["output-prefix"] = args["output-prefix"]
    args_output["verbose"] = !args["silent"]
    # Output
    return args_output
end

function julia_main()::Cint
    args = args_parser()
    main(args)
    return 0
end

end
