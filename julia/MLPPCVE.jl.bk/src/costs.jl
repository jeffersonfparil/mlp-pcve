function MSE(ŷ::CuArray{T,2}, y::CuArray{T,2})::CuArray{T,2} where {T<:AbstractFloat}
    # T = Float32; n = 1_000; y = CUDA.randn(T, (1, n)); ŷ = CUDA.randn(T, (1, n))
    map(x -> (x^2) / 2, ŷ - y) # division by 2 is cosmetic-ish --> just so that the derivative is simple the difference - see below
end

function MSE_derivative(
    ŷ::CuArray{T,2},
    y::CuArray{T,2},
)::CuArray{T,2} where {T<:AbstractFloat}
    # T = Float32; n = 1_000; y = CUDA.randn(T, (1, n)); ŷ = CUDA.randn(T, (1, n))
    map(x -> x, ŷ - y)
end

function MAE(ŷ::CuArray{T,2}, y::CuArray{T,2})::CuArray{T,2} where {T<:AbstractFloat}
    # T = Float32; n = 1_000; y = CUDA.randn(T, (1, n)); ŷ = CUDA.randn(T, (1, n))
    map(x -> abs(x) / 2, ŷ - y)
end

function MAE_derivative(
    ŷ::CuArray{T,2},
    y::CuArray{T,2},
)::CuArray{T,2} where {T<:AbstractFloat}
    # T = Float32; n = 1_000; y = CUDA.randn(T, (1, n)); ŷ = CUDA.randn(T, (1, n))
    map(x -> x < 0.0 ? 1.0 : -1.0, ŷ - y)
end

function HL(
    ŷ::CuArray{T,2},
    y::CuArray{T,2};
    δ::T = 1.0,
)::CuArray{T,2} where {T<:AbstractFloat}
    map(x -> abs(x) <= δ ? 0.5 * (x^2) : δ * (abs(x) - (0.5 * δ)), ŷ - y)
end

function HL_derivative(
    ŷ::CuArray{T,2},
    y::CuArray{T,2};
    δ::T = 1.0,
)::CuArray{T,2} where {T<:AbstractFloat}
    map(x -> abs(x) <= δ ? -x : -δ * sign(x), ŷ - y)
end
