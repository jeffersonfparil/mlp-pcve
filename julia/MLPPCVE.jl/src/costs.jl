module Costs

using CUDA

function MSE(y::CuArray{T}, ŷ::CuArray{T})::CuArray{T} where T <: AbstractFloat
    # T = Float32; n = 1_000; y = CUDA.randn(T, (1, n)); ŷ = CUDA.randn(T, (1, n))
    map(x -> 0.5*(x^2) , y - ŷ)
end

function MSE_derivative(y::CuArray{T}, ŷ::CuArray{T})::CuArray{T} where T <: AbstractFloat
    # T = Float32; n = 1_000; y = CUDA.randn(T, (1, n)); ŷ = CUDA.randn(T, (1, n))
    map(x -> 2.0*x , y - ŷ)
end

function MAE(y::CuArray{T}, ŷ::CuArray{T})::CuArray{T} where T <: AbstractFloat
    # T = Float32; n = 1_000; y = CUDA.randn(T, (1, n)); ŷ = CUDA.randn(T, (1, n))
    map(x -> 0.5*abs(x) , y - ŷ)
end

function MAE_derivative(y::CuArray{T}, ŷ::CuArray{T})::CuArray{T} where T <: AbstractFloat
    # T = Float32; n = 1_000; y = CUDA.randn(T, (1, n)); ŷ = CUDA.randn(T, (1, n))
    map(x -> x < 0.0 ? 1.0 : -1.0, y - ŷ)
end

# TODO
function HL(y::CuArray{T}, ŷ::CuArray{T}, δ::T=1e-7)::CuArray{T} where T <: AbstractFloat
    map(x -> abs(x) <= δ ? 0.5*(x^2) : δ*(abs(x) - (0.5*δ)), y - ŷ)
end

end