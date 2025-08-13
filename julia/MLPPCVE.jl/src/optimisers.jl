module Optimisers

using CUDA

# using Random, UnicodePlots; seed = 42; Random.seed!(seed); CUDA.seed!(seed); T = Float32; n = 1_000; p = 3; l = 2; hs = vcat(repeat([p], l), 1); X = CuArray{T}(rand(Bool, p, n)); W::Vector{CuArray{T}} = [CUDA.randn(T, hs[1], p)]; B::Vector{CuArray{T}} = [CUDA.randn(T, hs[1])]; [(push!(W, CUDA.randn(T, hs[i+1], hs[i])), push!(B, CUDA.randn(T, hs[i+1]))) for i in 1:l]; y = (CUDA.randn(T, 1, size(X, 1)) * X) .+ (T(0.01) .* CUDA.randn(T, 1, size(X, 2)));
# epochs = []
# loss = []
# state::Dict{String, Any} = Dict(
#     "β₁" => T(0.900),
#     "β₂" => T(0.999),
#     "ϵ" => T(1e-7),
#     "t" => T(0.0),
#     "m_W" => [CuArray{T}(zeros(size(x))) for x in W],
#     "v_W" => [CuArray{T}(zeros(size(x))) for x in W],
#     "m_B" => [CuArray{T}(zeros(size(x))) for x in B],
#     "v_B" => [CuArray{T}(zeros(size(x))) for x in B],
# );
# for i in 1:500
#     FP = ForwardBackward.forwardpass(X=X, W=W, B=B, F=Activations.relu)
#     BP = ForwardBackward.backpropagation(y=y, ŷ=CuArray(FP["ŷ"]), W=W, S=Vector(FP["S"]), A=Vector(FP["A"]), δC=Costs.MSE_derivative, δF=Activations.relu_derivative)
#     # Optimisers.gradientdescent!(W, B, BP["∇W"], BP["∇B"], r=T(0.0001))
#     Optimisers.Adam!(W, B, BP["∇W"], BP["∇B"], r=T(0.1), state=state)
#     # Optimisers.AdamMax!(W, B, BP["∇W"], BP["∇B"], r=T(0.1), state=state)
#     push!(epochs, i)
#     push!(loss, sum(Costs.MSE(ForwardBackward.forwardpass(X=X, W=W, B=B, F=Activations.relu)["ŷ"], y)))
# end
# UnicodePlots.scatterplot(epochs, loss)
# UnicodePlots.scatterplot(Matrix(ForwardBackward.forwardpass(X=X, W=W, B=B, F=Activations.relu)["ŷ"])[1, :], Matrix(y)[1, :])


function gradientdescent!(
    W::Vector{CuArray{T}},
    B::Vector{CuArray{T}},
    ∇W::Vector{CuArray{T}},
    ∇B::Vector{CuArray{T}};
    r::T = 0.001,
)::Nothing where T <: AbstractFloat
    # T = Float32; n = 1_000; p = 3; l = 2; hs = vcat(repeat([p], l), 1); X = CuArray{T}(rand(Bool, p, n)); W::Vector{CuArray{T}} = [CUDA.randn(T, hs[1], p)]; B::Vector{CuArray{T}} = [CUDA.randn(T, hs[1])]; [(push!(W, CUDA.randn(T, hs[i+1], hs[i])), push!(B, CUDA.randn(T, hs[i+1]))) for i in 1:l]; y = CUDA.randn(1, size(X, 2)); F = Activations.relu; FP = ForwardBackward.forwardpass(X=X, W=W, B=B, F=F); δC = Costs.MSE_derivative; δF = Activations.relu_derivative; BP = ForwardBackward.backpropagation(y=y, ŷ=FP["ŷ"], S=FP["S"], A=FP["A"], W=W, δC=δC, δF=δF); ∇W = BP["∇W"]; ∇B = BP["∇B"]; r = T(0.001); 
    for i in eachindex(W)
        # i = 1
        W[i] -= (r .* ∇W[i])
        B[i] -= (r .* ∇B[i])
    end
    nothing
end

function Adam!(
    W::Vector{CuArray{T}},
    B::Vector{CuArray{T}},
    ∇W::Vector{CuArray{T}},
    ∇B::Vector{CuArray{T}};
    r::T = 0.001,
    state::Dict{String, Any} = Dict(
        "β₁" => T(0.900),
        "β₂" => T(0.999),
        "ϵ" => T(1e-7),
        "t" => T(0.0),
        "m_W" => [CuArray{T}(zeros(size(x))) for x in W],
        "v_W" => [CuArray{T}(zeros(size(x))) for x in W],
        "m_B" => [CuArray{T}(zeros(size(x))) for x in B],
        "v_B" => [CuArray{T}(zeros(size(x))) for x in B],
    ),
)::Nothing where T <: AbstractFloat
    # T = Float32; n = 1_000; p = 3; l = 2; hs = vcat(repeat([p], l), 1); X = CuArray{T}(rand(Bool, p, n)); W::Vector{CuArray{T}} = [CUDA.randn(T, hs[1], p)]; B::Vector{CuArray{T}} = [CUDA.randn(T, hs[1])]; [(push!(W, CUDA.randn(T, hs[i+1], hs[i])), push!(B, CUDA.randn(T, hs[i+1]))) for i in 1:l]; y = CUDA.randn(1, size(X, 2)); F = Activations.relu; FP = ForwardBackward.forwardpass(X=X, W=W, B=B, F=F); δC = Costs.MSE_derivative; δF = Activations.relu_derivative; BP = ForwardBackward.backpropagation(y=y, ŷ=FP["ŷ"], S=FP["S"], A=FP["A"], W=W, δC=δC, δF=δF); ∇W = BP["∇W"]; ∇B = BP["∇B"]; r = T(0.001); 
    # state::Dict{String, Any} = Dict(
    #     "β₁" => T(0.900),
    #     "β₂" => T(0.999),
    #     "ϵ" => T(1e-7),
    #     "t" => T(0.0),
    #     "m_W" => [CuArray{T}(zeros(size(x))) for x in W],
    #     "v_W" => [CuArray{T}(zeros(size(x))) for x in W],
    #     "m_B" => [CuArray{T}(zeros(size(x))) for x in B],
    #     "v_B" => [CuArray{T}(zeros(size(x))) for x in B],
    # );
    state["t"] += T(1.00)
    for i in eachindex(W)
        # i = 1
        # Update the gradient of the weights
        state["m_W"][i] = (state["β₁"] .* state["m_W"][i]) .+ ((1.00 - state["β₁"]) .* ∇W[i])
        state["v_W"][i] = (state["β₂"] .* state["v_W"][i]) .+ ((1.00 - state["β₂"]) .* (∇W[i].^2))
        momentum = state["m_W"][i] ./ (1.00 - (state["β₁"]^state["t"]))
        velocity = state["v_W"][i] ./ (1.00 - (state["β₂"]^state["t"]))
        W[i] -= (r * momentum ./ (sqrt.(velocity) .+ state["ϵ"]))
        # Update the gradient of the biases
        state["m_B"][i] = (state["β₁"] .* state["m_B"][i]) .+ ((1.00 - state["β₁"]) .* ∇B[i])
        state["v_B"][i] = (state["β₂"] .* state["v_B"][i]) .+ ((1.00 - state["β₂"]) .* (∇B[i].^2))
        momentum = state["m_B"][i] ./ (1.00 - (state["β₁"]^state["t"]))
        velocity = state["v_B"][i] ./ (1.00 - (state["β₂"]^state["t"]))
        B[i] -= (r * momentum ./ (sqrt.(velocity) .+ state["ϵ"]))
    end
    nothing
end

function AdamMax!(
    W::Vector{CuArray{T}},
    B::Vector{CuArray{T}},
    ∇W::Vector{CuArray{T}},
    ∇B::Vector{CuArray{T}};
    r::T = 0.001,
    state::Dict{String, Any} = Dict(
        "β₁" => T(0.900),
        "β₂" => T(0.999),
        "ϵ" => T(1e-7),
        "t" => T(0.0),
        "m_W" => [CuArray{T}(zeros(size(x))) for x in W],
        "v_W" => [CuArray{T}(zeros(size(x))) for x in W],
        "m_B" => [CuArray{T}(zeros(size(x))) for x in B],
        "v_B" => [CuArray{T}(zeros(size(x))) for x in B],
    ),
)::Nothing where T <: AbstractFloat
    # T = Float32; n = 1_000; p = 3; l = 2; hs = vcat(repeat([p], l), 1); X = CuArray{T}(rand(Bool, p, n)); W::Vector{CuArray{T}} = [CUDA.randn(T, hs[1], p)]; B::Vector{CuArray{T}} = [CUDA.randn(T, hs[1])]; [(push!(W, CUDA.randn(T, hs[i+1], hs[i])), push!(B, CUDA.randn(T, hs[i+1]))) for i in 1:l]; y = CUDA.randn(1, size(X, 2)); F = Activations.relu; FP = ForwardBackward.forwardpass(X=X, W=W, B=B, F=F); δC = Costs.MSE_derivative; δF = Activations.relu_derivative; BP = ForwardBackward.backpropagation(y=y, ŷ=FP["ŷ"], S=FP["S"], A=FP["A"], W=W, δC=δC, δF=δF); ∇W = BP["∇W"]; ∇B = BP["∇B"]; r = T(0.001); 
    # state::Dict{String, Any} = Dict(
    #     "β₁" => T(0.900),
    #     "β₂" => T(0.999),
    #     "ϵ" => T(1e-7),
    #     "t" => T(0.0),
    #     "m_W" => [CuArray{T}(zeros(size(x))) for x in W],
    #     "v_W" => [CuArray{T}(zeros(size(x))) for x in W],
    #     "m_B" => [CuArray{T}(zeros(size(x))) for x in B],
    #     "v_B" => [CuArray{T}(zeros(size(x))) for x in B],
    # );
    state["t"] += T(1.00)
    for i in eachindex(W)
        # i = 1
        # Update the gradient of the weights
        state["m_W"][i] = (state["β₁"] .* state["m_W"][i]) .+ ((1.00 - state["β₁"]) .* ∇W[i])
        G1 = state["β₂"] .* state["v_W"][i]
        G2 = abs.(∇W[i])
        state["v_W"][i] = [G1, G2][argmax([sum(G1), sum(G2)])]
        r_adj = r / (1.00 - state["β₁"]^state["t"])
        W[i] -= (r_adj * state["m_W"][i] ./ (state["v_W"][i] .+ state["ϵ"]))
        # Update the gradient of the biases
        state["m_B"][i] = (state["β₁"] .* state["m_B"][i]) .+ ((1.00 - state["β₁"]) .* ∇B[i])
        G1 = state["β₂"] .* state["v_B"][i]
        G2 = abs.(∇B[i])
        state["v_B"][i] = [G1, G2][argmax([sum(G1), sum(G2)])]
        r_adj = r / (1.00 - state["β₁"]^state["t"])
        B[i] -= (r_adj * state["m_B"][i] ./ (state["v_B"][i] .+ state["ϵ"]))
    end
    nothing
end


end