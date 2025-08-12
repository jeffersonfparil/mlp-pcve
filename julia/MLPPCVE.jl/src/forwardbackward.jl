module ForwardBackward

using CUDA

# T = Float32; n = 1_000; p = 50; l = 2; hs = vcat([x * 100 for x in 1:l], 1); X = CuArray{T}(rand(Bool, p, n)); W::Vector{CuArray{T}} = [CUDA.randn(T, hs[1], p)]; B::Vector{CuArray{T}} = [CUDA.randn(T, hs[1])]; [(push!(W, CUDA.randn(T, hs[i+1], hs[i])), push!(B, CUDA.randn(T, hs[i+1]))) for i in 1:l]; F = Activations.leakyrelu
# FP = ForwardBackward.forwardpass(X=X, W=W, B=B, F=F)
function forwardpass(;
    X::CuArray{T},
    W::Vector{CuArray{T}},
    B::Vector{CuArray{T}},
    F::Function,
)::Dict{String, Union{CuArray{T}, Vector{CuArray{T}}}} where T <:AbstractFloat
    # T = Float32; n = 1_000; p = 50; l = 2; hs = vcat([x * 100 for x in 1:l], 1); X = CuArray{T}(rand(Bool, p, n)); W::Vector{CuArray{T}} = [CUDA.randn(T, hs[1], p)]; B::Vector{CuArray{T}} = [CUDA.randn(T, hs[1])]; [(push!(W, CUDA.randn(T, hs[i+1], hs[i])), push!(B, CUDA.randn(T, hs[i+1]))) for i in 1:l]; F = Activations.leakyrelu
    for i in eachindex(W)
        if i == 1
            @assert size(W[i]) == (length(B[i]), size(X,1))
        else
            @assert size(W[i]) == (length(B[i]), size(W[i-1], 1))
        end
    end
    S::Vector{CuArray{T}} = []
    A::Vector{CuArray{T}} = [X]
    for i in eachindex(W)
        s = (W[i] * A[i]) .+ B[i]
        push!(S, s)
        push!(A, F.(s))
    end
    Dict(
        "ŷ" => A[end], 
        "weighted_sums" => S, 
        "activations" => A[1:(end-1)], 
    )
end

# T = Float32; n = 1_000; p = 50; l = 2; hs = vcat([x * 100 for x in 1:l], 1); X = CuArray{T}(rand(Bool, p, n)); W::Vector{CuArray{T}} = [CUDA.randn(T, hs[1], p)]; B::Vector{CuArray{T}} = [CUDA.randn(T, hs[1])]; [(push!(W, CUDA.randn(T, hs[i+1], hs[i])), push!(B, CUDA.randn(T, hs[i+1]))) for i in 1:l]; F = Activations.relu; FP = ForwardBackward.forwardpass(X=X, W=W, B=B, F=F); C = Costs.MSE; ŷ = FP["ŷ"]; y = CUDA.randn(1, length(ŷ)); δF = Activations.relu_derivative; S = FP["weighted_sums"]; A = FP["activations"]
# BP = ForwardBackward.backpropagation(y=y, ŷ=ŷ, S=S, W=W, A=A, C=C, δF=δF)
function backpropagation(;
    y::CuArray{T}, 
    ŷ::CuArray{T},
    S::Vector{CuArray{T}},
    W::Vector{CuArray{T}},
    A::Vector{CuArray{T}},
    δC::Function,
    δF::Function,
)::Dict{String, Vector{CuArray{T}}} where T <: AbstractFloat
    # T = Float32; n = 1_000; p = 50; l = 2; hs = vcat([x * 100 for x in 1:l], 1); X = CuArray{T}(rand(Bool, p, n)); W::Vector{CuArray{T}} = [CUDA.randn(T, hs[1], p)]; B::Vector{CuArray{T}} = [CUDA.randn(T, hs[1])]; [(push!(W, CUDA.randn(T, hs[i+1], hs[i])), push!(B, CUDA.randn(T, hs[i+1]))) for i in 1:l]; F = Activations.relu; FP = ForwardBackward.forwardpass(X=X, W=W, B=B, F=F); δC = Costs.MSE_derivative; ŷ = FP["ŷ"]; y = CUDA.randn(1, length(ŷ)); δF = Activations.relu_derivative; S = FP["weighted_sums"]; A = FP["activations"]
    # Extract the errors
    L = δC(y, ŷ)
    δ = δF.(S[end])
    Δ::Vector{CuArray{T}} = [L .* δ]
    for i in 1:(length(S)-1)
        # i = 1
        L = W[end-(i-1)]' * Δ[end]
        δ = δF.(S[end-i])
        push!(Δ, L .* δ)
    end
    Δ = reverse(Δ)
    ∇W::Vector{CuArray{T}} = []
    ∇B::Vector{CuArray{T}} = []
    for i in eachindex(Δ)
        # i = 1
        push!(∇W, Δ[i] * A[i]')
        push!(∇B, sum(Δ[i], dims=2)/size(Δ[i], 2))
    end
    # [sum(x .!= 0.0) for x in ∇W]
    # [sum(x .!= 0.0) for x in W]
    # [sum(x .!= 0.0) for x in ∇B]
    # [sum(x .!= 0.0) for x in B]
    Dict(
        "∇W" => ∇W, 
        "∇B" => ∇B, 
    )
end



end