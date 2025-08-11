module Forward

using CUDA

# T = Float32; n = 1_000; p = 50; l = 2; hs = vcat([x * 100 for x in 1:l], 1); X = CuArray{T}(rand(Bool, p, n)); W::Vector{CuArray{T}} = [CUDA.randn(T, hs[1], p)]; B::Vector{CuArray{T}} = [CUDA.randn(T, hs[1])]; [(push!(W, CUDA.randn(T, hs[i+1], hs[i])), push!(B, CUDA.randn(T, hs[i+1]))) for i in 1:l]; F = Activations.leaky_relu
# ŷ, A = forwardpass(X, W, B, F)
function forwardpass(
    X::CuArray{T},
    W::Vector{CuArray{T}},
    B::Vector{CuArray{T}},
    F::Function,
)::Tuple{CuArray{T}, Vector{CuArray{T}}} where T <:AbstractFloat
    # T = Float32; n = 1_000; p = 50; l = 2; hs = vcat([x * 100 for x in 1:l], 1); X = CuArray{T}(rand(Bool, p, n)); W::Vector{CuArray{T}} = [CUDA.randn(T, hs[1], p)]; B::Vector{CuArray{T}} = [CUDA.randn(T, hs[1])]; [(push!(W, CUDA.randn(T, hs[i+1], hs[i])), push!(B, CUDA.randn(T, hs[i+1]))) for i in 1:l]; F = Activations.leaky_relu
    for i in eachindex(W)
        if i == 1
            @assert size(W[i]) == (length(B[i]), size(X,1))
        else
            @assert size(W[i]) == (length(B[i]), size(W[i-1], 1))
        end
    end
    A::Vector{CuArray{T}} = [X]
    for i in eachindex(W)
        push!(A, F.((W[i] * A[i]) .+ B[i]))
    end
    (
        A[end],
        A[2:(end-1)]
    )
end




end