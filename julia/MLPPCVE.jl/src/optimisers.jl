"""
    gradientdescent!(Ω::Network{T}; η::T = 0.001) where {T<:AbstractFloat}

Perform an in-place vanilla gradient descent update on the parameters of `Ω`.

# Arguments
- `Ω::Network{T}`: A network instance whose parameter arrays `W` and `b`
  and corresponding gradient arrays `∇W` and `∇b` are expected to exist and
  have matching indexing.
- `η::T = 0.001`: Learning rate (step size) used to scale the parameter update.

# Behaviour
- For each index `i` in `eachindex(Ω.W)`, updates
  `Ω.W[i] -= η .* Ω.∇W[i]` and `Ω.b[i] -= η .* Ω.∇b[i]`.
- Updates are done in-place; the function returns `nothing`.
- The `!` suffix indicates mutation of `Ω`.

# Notes
- Assumes gradients have already been computed and stored in `Ω.∇W` and `Ω.∇b`.
- No gradient clipping, weight decay, or momentum is applied — add these
  externally if required.
- Ensure `η` is chosen appropriately for the floating-point type `T` to
  avoid numerical instability.

# TODO: Examples
```jldoctest; setup = :(using MLPPCVE)
julia> 0
0
```
"""
function gradientdescent!(Ω::Network{T}; η::T = 0.001)::Nothing where {T<:AbstractFloat}
    for i in eachindex(Ω.W)
        # i = 1
        Ω.W[i] -= (η .* Ω.∇W[i])
        Ω.b[i] -= (η .* Ω.∇b[i])
    end
    nothing
end

"""
    Adam!(Ω::Network{T}; η::T = 0.001, state::Dict{String,Any} = default_state) where {T<:AbstractFloat} -> Nothing

Perform an in-place Adam optimization step on `Ω` (mutates `Ω.W` and `Ω.b`) using CUDA arrays.

# Arguments
- `Ω::Network{T}`: Network instance parameterized by element type `T<:AbstractFloat`. Required fields used by this routine:
  - `Ω.W`: Vector/collection of weight arrays (CuArray{T,2}).
  - `Ω.b`: Vector/collection of bias arrays (CuArray{T,1}).
  - `Ω.∇W`: Gradients for weights with the same shapes as `Ω.W`.
  - `Ω.∇b`: Gradients for biases with the same shapes as `Ω.b`.
- `η::T = 0.001`: Learning rate (scalar).
- `state::Dict{String,Any}`: Mutable dictionary containing optimizer state. Expected keys and types:
  + `"β₁"::T`: exponential decay rate for the first moment (default 0.9).
  + `"β₂"::T`: exponential decay rate for the second moment (default 0.999).
  + `"ϵ"::T`: numerical stability constant (default 1e-8).
  + `"t"::T`: time step counter (scalar).
  + `"m_W"::Vector{CuArray{T,2}}`: first moment estimates for weights, same shapes as `Ω.W`.
  + `"v_W"::Vector{CuArray{T,2}}`: second moment estimates for weights, same shapes as `Ω.W`.
  + `"m_b"::Vector{CuArray{T,1}}`: first moment estimates for biases, same shapes as `Ω.b`.
  + `"v_b"::Vector{CuArray{T,1}}`: second moment estimates for biases, same shapes as `Ω.b`.

# Notes
- Increments the internal time-step `state["t"]` by 1 and uses it to compute bias-corrected estimates:
    m_hat = m / (1 - β₁^t), v_hat = v / (1 - β₂^t)
  then updates parameters with:
    param -= η * m_hat ./ (sqrt.(v_hat) .+ ϵ)
- All updates are performed in-place on GPU arrays to avoid unnecessary host-device transfers.
- The function returns `nothing` and mutates the provided `Ω` and `state`.
- CUDA and CuArrays must be available and the network parameters and state arrays must already reside on the GPU (CuArray).
- Gradients (`Ω.∇W`, `Ω.∇b`) must be computed and populated prior to calling this function.
- The default `state` allocates zero-initialized CuArrays matching the shapes of `Ω.W` and `Ω.b` when invoked with the default argument. If you provide a custom `state`, ensure its arrays match parameter shapes and types.

# TODO: Examples
```jldoctest; setup = :(using MLPPCVE)
julia> 0
0
```
"""
function Adam!(
    Ω::Network{T};
    η::T = 0.001,
    state::Dict{String,Any} = Dict(
        "β₁" => T(0.900),
        "β₂" => T(0.999),
        "ϵ" => T(1e-8),
        "t" => T(0.0),
        "m_W" => [CuArray{T,2}(zeros(size(x))) for x in Ω.W],
        "v_W" => [CuArray{T,2}(zeros(size(x))) for x in Ω.W],
        "m_b" => [CuArray{T,1}(zeros(size(x))) for x in Ω.b],
        "v_b" => [CuArray{T,1}(zeros(size(x))) for x in Ω.b],
    ),
)::Nothing where {T<:AbstractFloat}
    state["t"] += T(1.00)
    for i in eachindex(Ω.W)
        # i = 1
        # Update the gradient of the weights
        state["m_W"][i] =
            (state["β₁"] .* state["m_W"][i]) .+ ((1.00 - state["β₁"]) .* Ω.∇W[i])
        state["v_W"][i] =
            (state["β₂"] .* state["v_W"][i]) .+ ((1.00 - state["β₂"]) .* (Ω.∇W[i] .^ 2))
        momentum = state["m_W"][i] ./ (1.00 - (state["β₁"]^state["t"]))
        velocity = state["v_W"][i] ./ (1.00 - (state["β₂"]^state["t"]))
        Ω.W[i] -= (η * momentum ./ (sqrt.(velocity) .+ state["ϵ"]))
        # Update the gradient of the biases
        state["m_b"][i] =
            (state["β₁"] .* state["m_b"][i]) .+ ((1.00 - state["β₁"]) .* Ω.∇b[i])
        state["v_b"][i] =
            (state["β₂"] .* state["v_b"][i]) .+ ((1.00 - state["β₂"]) .* (Ω.∇b[i] .^ 2))
        momentum = state["m_b"][i] ./ (1.00 - (state["β₁"]^state["t"]))
        velocity = state["v_b"][i] ./ (1.00 - (state["β₂"]^state["t"]))
        Ω.b[i] -= (η * momentum ./ (sqrt.(velocity) .+ state["ϵ"]))
    end
    nothing
end

"""
    AdamMax!(Ω::Network{T}; η::T=0.001, state::Dict{String,Any}=...)

Perform an in-place AdamMax optimization step on the given network `Ω`.

# Arguments
- `Ω::Network{T}`: The neural network to update. Must provide
  - `Ω.W`: collection of weight arrays (e.g., Vector of matrices),
  - `Ω.b`: collection of bias arrays (e.g., Vector of vectors),
  - `Ω.∇W`, `Ω.∇b`: corresponding gradients with the same shapes as `W` and `b`.
- `η::T=0.001`: (keyword) Base learning rate.
- `state::Dict{String,Any}`: (keyword) Mutable dictionary holding optimizer
  state across calls. Expected keys (and their typical meanings) are:
  - `"β₁"`: first-moment decay coefficient (e.g., 0.9)
  - `"β₂"`: second-moment decay coefficient (e.g., 0.999)
  - `"ϵ"`: small constant for numerical stability (e.g., 1e-8)
  - `"t"`: timestep counter (scalar)
  - `"m_W"`, `"v_W"`: vectors of arrays storing first and second moments for weights
  - `"m_b"`, `"v_b"`: vectors of arrays storing first and second moments for biases

# Notes
For each parameter array (weights and biases) the function:
- Increments the timestep `t`.
- Updates the exponential moving average of the gradient (`m`).
- Updates the second moment (`v`) by taking the elementwise maximum of the decayed previous `v` and the absolute current gradient.
- Applies a bias-corrected step for the first moment: `step = η / (1 - β₁^t) * m ./ (v .+ ϵ)` and subtracts `step` from the parameters.

# TODO: Examples
```jldoctest; setup = :(using MLPPCVE)
julia> 0
0
```
"""
function AdamMax!(
    Ω::Network{T};
    η::T = 0.001,
    state::Dict{String,Any} = Dict(
        "β₁" => T(0.900),
        "β₂" => T(0.999),
        "ϵ" => T(1e-8),
        "t" => T(0.0),
        "m_W" => [CuArray{T,2}(zeros(size(x))) for x in Ω.W],
        "v_W" => [CuArray{T,2}(zeros(size(x))) for x in Ω.W],
        "m_b" => [CuArray{T,1}(zeros(size(x))) for x in Ω.b],
        "v_b" => [CuArray{T,1}(zeros(size(x))) for x in Ω.b],
    ),
)::Nothing where {T<:AbstractFloat}
    state["t"] += T(1.00)
    for i in eachindex(Ω.W)
        # i = 1
        # Update the gradient of the weights
        state["m_W"][i] =
            (state["β₁"] .* state["m_W"][i]) .+ ((1.00 - state["β₁"]) .* Ω.∇W[i])
        G1 = state["β₂"] .* state["v_W"][i]
        G2 = abs.(Ω.∇W[i])
        state["v_W"][i] = [G1, G2][argmax([sum(G1), sum(G2)])]
        r_adj = η / (1.00 - state["β₁"]^state["t"])
        Ω.W[i] -= (r_adj * state["m_W"][i] ./ (state["v_W"][i] .+ state["ϵ"]))
        # Update the gradient of the biases
        state["m_b"][i] =
            (state["β₁"] .* state["m_b"][i]) .+ ((1.00 - state["β₁"]) .* Ω.∇b[i])
        G1 = state["β₂"] .* state["v_b"][i]
        G2 = abs.(Ω.∇b[i])
        state["v_b"][i] = [G1, G2][argmax([sum(G1), sum(G2)])]
        r_adj = η / (1.00 - state["β₁"]^state["t"])
        Ω.b[i] -= (r_adj * state["m_b"][i] ./ (state["v_b"][i] .+ state["ϵ"]))
    end
    nothing
end
