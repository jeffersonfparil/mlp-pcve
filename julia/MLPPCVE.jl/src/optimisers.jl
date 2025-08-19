function gradientdescent!(Ω::Network{T}; η::T = 0.001)::Nothing where T <: AbstractFloat
    for i in eachindex(Ω.W)
        # i = 1
        Ω.W[i] -= (η .* Ω.∇W[i])
        Ω.b[i] -= (η .* Ω.∇b[i])
    end
    nothing
end

function Adam!(
    Ω::Network{T};
    η::T = 0.001,
    state::Dict{String, Any} = Dict(
        "β₁" => T(0.900),
        "β₂" => T(0.999),
        "ϵ" => T(1e-8),
        "t" => T(0.0),
        "m_W" => [CuArray{T,2}(zeros(size(x))) for x in Ω.W],
        "v_W" => [CuArray{T,2}(zeros(size(x))) for x in Ω.W],
        "m_b" => [CuArray{T,1}(zeros(size(x))) for x in Ω.b],
        "v_b" => [CuArray{T,1}(zeros(size(x))) for x in Ω.b],
    ),
)::Nothing where T <: AbstractFloat
    state["t"] += T(1.00)
    for i in eachindex(Ω.W)
        # i = 1
        # Update the gradient of the weights
        state["m_W"][i] = (state["β₁"] .* state["m_W"][i]) .+ ((1.00 - state["β₁"]) .* Ω.∇W[i])
        state["v_W"][i] = (state["β₂"] .* state["v_W"][i]) .+ ((1.00 - state["β₂"]) .* (Ω.∇W[i].^2))
        momentum = state["m_W"][i] ./ (1.00 - (state["β₁"]^state["t"]))
        velocity = state["v_W"][i] ./ (1.00 - (state["β₂"]^state["t"]))
        Ω.W[i] -= (η * momentum ./ (sqrt.(velocity) .+ state["ϵ"]))
        # Update the gradient of the biases
        state["m_b"][i] = (state["β₁"] .* state["m_b"][i]) .+ ((1.00 - state["β₁"]) .* Ω.∇b[i])
        state["v_b"][i] = (state["β₂"] .* state["v_b"][i]) .+ ((1.00 - state["β₂"]) .* (Ω.∇b[i].^2))
        momentum = state["m_b"][i] ./ (1.00 - (state["β₁"]^state["t"]))
        velocity = state["v_b"][i] ./ (1.00 - (state["β₂"]^state["t"]))
        Ω.b[i] -= (η * momentum ./ (sqrt.(velocity) .+ state["ϵ"]))
    end
    nothing
end

function AdamMax!(
    Ω::Network{T};
    η::T = 0.001,
    state::Dict{String, Any} = Dict(
        "β₁" => T(0.900),
        "β₂" => T(0.999),
        "ϵ" => T(1e-8),
        "t" => T(0.0),
        "m_W" => [CuArray{T,2}(zeros(size(x))) for x in Ω.W],
        "v_W" => [CuArray{T,2}(zeros(size(x))) for x in Ω.W],
        "m_b" => [CuArray{T,1}(zeros(size(x))) for x in Ω.b],
        "v_b" => [CuArray{T,1}(zeros(size(x))) for x in Ω.b],
    ),
)::Nothing where T <: AbstractFloat
    state["t"] += T(1.00)
    for i in eachindex(Ω.W)
        # i = 1
        # Update the gradient of the weights
        state["m_W"][i] = (state["β₁"] .* state["m_W"][i]) .+ ((1.00 - state["β₁"]) .* Ω.∇W[i])
        G1 = state["β₂"] .* state["v_W"][i]
        G2 = abs.(Ω.∇W[i])
        state["v_W"][i] = [G1, G2][argmax([sum(G1), sum(G2)])]
        r_adj = η / (1.00 - state["β₁"]^state["t"])
        Ω.W[i] -= (r_adj * state["m_W"][i] ./ (state["v_W"][i] .+ state["ϵ"]))
        # Update the gradient of the biases
        state["m_b"][i] = (state["β₁"] .* state["m_b"][i]) .+ ((1.00 - state["β₁"]) .* Ω.∇b[i])
        G1 = state["β₂"] .* state["v_b"][i]
        G2 = abs.(Ω.∇b[i])
        state["v_b"][i] = [G1, G2][argmax([sum(G1), sum(G2)])]
        r_adj = η / (1.00 - state["β₁"]^state["t"])
        Ω.b[i] -= (r_adj * state["m_b"][i] ./ (state["v_b"][i] .+ state["ϵ"]))
    end
    nothing
end
