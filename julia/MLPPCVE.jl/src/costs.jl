"""
    MSE(ŷ::CuArray{T,2}, y::CuArray{T,2}) where {T<:AbstractFloat}

Calculate the Mean Squared Error (MSE) loss between predicted values `ŷ` and target values `y`.

Returns a CuArray containing element-wise MSE values: (ŷ - y)²/2

# Arguments
- `ŷ::CuArray{T,2}`: Predicted values
- `y::CuArray{T,2}`: Target values
"""
function MSE(ŷ::CuArray{T,2}, y::CuArray{T,2})::CuArray{T,2} where {T<:AbstractFloat}
    # T = Float32; n = 1_000; y = CUDA.randn(T, (1, n)); ŷ = CUDA.randn(T, (1, n))
    map(x -> (x^2) / 2, ŷ - y) # division by 2 is cosmetic-ish --> just so that the derivative is simple the difference - see below
end

"""
    MSE_derivative(ŷ::CuArray{T,2}, y::CuArray{T,2}) where {T<:AbstractFloat}

Calculate the derivative of Mean Squared Error (MSE) loss with respect to predicted values.

Returns a CuArray containing element-wise derivatives: ŷ - y

# Arguments
- `ŷ::CuArray{T,2}`: Predicted values
- `y::CuArray{T,2}`: Target values
"""
function MSE_derivative(
    ŷ::CuArray{T,2},
    y::CuArray{T,2},
)::CuArray{T,2} where {T<:AbstractFloat}
    # T = Float32; n = 1_000; y = CUDA.randn(T, (1, n)); ŷ = CUDA.randn(T, (1, n))
    map(x -> x, ŷ - y)
end

"""
    MAE(ŷ::CuArray{T,2}, y::CuArray{T,2}) where {T<:AbstractFloat}

Calculate the Mean Absolute Error (MAE) loss between predicted values `ŷ` and target values `y`.

Returns a CuArray containing element-wise MAE values: |ŷ - y|/2

# Arguments
- `ŷ::CuArray{T,2}`: Predicted values
- `y::CuArray{T,2}`: Target values
"""
function MAE(ŷ::CuArray{T,2}, y::CuArray{T,2})::CuArray{T,2} where {T<:AbstractFloat}
    # T = Float32; n = 1_000; y = CUDA.randn(T, (1, n)); ŷ = CUDA.randn(T, (1, n))
    map(x -> abs(x) / 2, ŷ - y)
end

"""
    MAE_derivative(ŷ::CuArray{T,2}, y::CuArray{T,2}) where {T<:AbstractFloat}

Calculate the derivative of Mean Absolute Error (MAE) loss with respect to predicted values.

Returns a CuArray containing element-wise derivatives: sign(ŷ - y)

# Arguments
- `ŷ::CuArray{T,2}`: Predicted values
- `y::CuArray{T,2}`: Target values
"""
function MAE_derivative(
    ŷ::CuArray{T,2},
    y::CuArray{T,2},
)::CuArray{T,2} where {T<:AbstractFloat}
    # T = Float32; n = 1_000; y = CUDA.randn(T, (1, n)); ŷ = CUDA.randn(T, (1, n))
    map(x -> x < 0.0 ? 1.0 : -1.0, ŷ - y)
end

"""
    HL(ŷ::CuArray{T,2}, y::CuArray{T,2}; δ::T=1.0) where {T<:AbstractFloat}

Calculate the Huber Loss between predicted values `ŷ` and target values `y`.
Combines the best properties of MSE and MAE, using MSE for small errors and MAE for large errors.

Returns a CuArray containing element-wise Huber loss values:
- If |x| ≤ δ: 0.5x²
- If |x| > δ: δ(|x| - 0.5δ)
where x = ŷ - y

# Arguments
- `ŷ::CuArray{T,2}`: Predicted values
- `y::CuArray{T,2}`: Target values
- `δ::T=1.0`: Threshold parameter that determines the transition point between MSE and MAE
"""
function HL(
    ŷ::CuArray{T,2},
    y::CuArray{T,2};
    δ::T = 1.0,
)::CuArray{T,2} where {T<:AbstractFloat}
    map(x -> abs(x) <= δ ? 0.5 * (x^2) : δ * (abs(x) - (0.5 * δ)), ŷ - y)
end

"""
    HL_derivative(ŷ::CuArray{T,2}, y::CuArray{T,2}; δ::T=1.0) where {T<:AbstractFloat}

Calculate the derivative of Huber Loss with respect to predicted values.

Returns a CuArray containing element-wise derivatives:
- If |x| ≤ δ: -x
- If |x| > δ: -δ * sign(x)
where x = ŷ - y

# Arguments
- `ŷ::CuArray{T,2}`: Predicted values
- `y::CuArray{T,2}`: Target values
- `δ::T=1.0`: Threshold parameter that determines the transition point between MSE and MAE
"""
function HL_derivative(
    ŷ::CuArray{T,2},
    y::CuArray{T,2};
    δ::T = 1.0,
)::CuArray{T,2} where {T<:AbstractFloat}
    map(x -> abs(x) <= δ ? -x : -δ * sign(x), ŷ - y)
end
