function gradientdescent!(
    W::Vector{CuArray{T, 2}},
    b::Vector{CuArray{T, 1}},
    ∇W::Vector{CuArray{T, 2}},
    ∇b::Vector{CuArray{T, 1}};
    η::T = 0.001,
)::Nothing where T <: AbstractFloat
    for i in eachindex(W)
        # i = 1
        W[i] -= (η .* ∇W[i])
        b[i] -= (η .* ∇b[i])
    end
    nothing
end

function Adam!(
    W::Vector{CuArray{T, 2}},
    b::Vector{CuArray{T, 1}},
    ∇W::Vector{CuArray{T, 2}},
    ∇b::Vector{CuArray{T, 1}};
    η::T = 0.001,
    state::Dict{String, Any} = Dict(
        "β₁" => T(0.900),
        "β₂" => T(0.999),
        "ϵ" => T(1e-8),
        "t" => T(0.0),
        "m_W" => [CuArray{T,2}(zeros(size(x))) for x in W],
        "v_W" => [CuArray{T,2}(zeros(size(x))) for x in W],
        "m_b" => [CuArray{T,1}(zeros(size(x))) for x in b],
        "v_b" => [CuArray{T,1}(zeros(size(x))) for x in b],
    ),
)::Nothing where T <: AbstractFloat
    state["t"] += T(1.00)
    for i in eachindex(W)
        # i = 1
        # Update the gradient of the weights
        state["m_W"][i] = (state["β₁"] .* state["m_W"][i]) .+ ((1.00 - state["β₁"]) .* ∇W[i])
        state["v_W"][i] = (state["β₂"] .* state["v_W"][i]) .+ ((1.00 - state["β₂"]) .* (∇W[i].^2))
        momentum = state["m_W"][i] ./ (1.00 - (state["β₁"]^state["t"]))
        velocity = state["v_W"][i] ./ (1.00 - (state["β₂"]^state["t"]))
        W[i] -= (η * momentum ./ (sqrt.(velocity) .+ state["ϵ"]))
        # Update the gradient of the biases
        state["m_b"][i] = (state["β₁"] .* state["m_b"][i]) .+ ((1.00 - state["β₁"]) .* ∇b[i])
        state["v_b"][i] = (state["β₂"] .* state["v_b"][i]) .+ ((1.00 - state["β₂"]) .* (∇b[i].^2))
        momentum = state["m_b"][i] ./ (1.00 - (state["β₁"]^state["t"]))
        velocity = state["v_b"][i] ./ (1.00 - (state["β₂"]^state["t"]))
        b[i] -= (η * momentum ./ (sqrt.(velocity) .+ state["ϵ"]))
    end
    nothing
end

function AdamMax!(
    W::Vector{CuArray{T, 2}},
    b::Vector{CuArray{T, 1}},
    ∇W::Vector{CuArray{T, 2}},
    ∇b::Vector{CuArray{T, 1}};
    η::T = 0.001,
    state::Dict{String, Any} = Dict(
        "β₁" => T(0.900),
        "β₂" => T(0.999),
        "ϵ" => T(1e-7),
        "t" => T(0.0),
        "m_W" => [CuArray{T,2}(zeros(size(x))) for x in W],
        "v_W" => [CuArray{T,2}(zeros(size(x))) for x in W],
        "m_b" => [CuArray{T,1}(zeros(size(x))) for x in b],
        "v_b" => [CuArray{T,1}(zeros(size(x))) for x in b],
    ),
)::Nothing where T <: AbstractFloat
    state["t"] += T(1.00)
    for i in eachindex(W)
        # i = 1
        # Update the gradient of the weights
        state["m_W"][i] = (state["β₁"] .* state["m_W"][i]) .+ ((1.00 - state["β₁"]) .* ∇W[i])
        G1 = state["β₂"] .* state["v_W"][i]
        G2 = abs.(∇W[i])
        state["v_W"][i] = [G1, G2][argmax([sum(G1), sum(G2)])]
        r_adj = η / (1.00 - state["β₁"]^state["t"])
        W[i] -= (r_adj * state["m_W"][i] ./ (state["v_W"][i] .+ state["ϵ"]))
        # Update the gradient of the biases
        state["m_b"][i] = (state["β₁"] .* state["m_b"][i]) .+ ((1.00 - state["β₁"]) .* ∇b[i])
        G1 = state["β₂"] .* state["v_b"][i]
        G2 = abs.(∇b[i])
        state["v_b"][i] = [G1, G2][argmax([sum(G1), sum(G2)])]
        r_adj = η / (1.00 - state["β₁"]^state["t"])
        b[i] -= (r_adj * state["m_b"][i] ./ (state["v_b"][i] .+ state["ϵ"]))
    end
    nothing
end
