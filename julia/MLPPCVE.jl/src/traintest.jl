function train()
    X, y = simulate() # It it recommended that X be standardised (μ=0, and σ=1)
    n_layers::Int64 = 2
    n_hidden_nodes::Vector{Int64} = vcat([x * 100 for x in 1:n_layers], 1)
    F::Function = relu
    δF::Function = relu_derivative
    C::Function = MSE
    δC::Function = MSE_derivative
    n_epochs::Int64 = 1_000
    optimiser = ["GD", "Adam", "AdamMax"][2]
    η = 0.001

    W, b = init(X)
    ∇W = initgradient(W)
    ∇b = initgradient(b)

    




end