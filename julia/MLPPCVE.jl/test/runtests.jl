using MLPPCVE
using Test
using Documenter

Documenter.doctest(MLPPCVE)

# using Random, UnicodePlots, Distributions, ProgressMeter
# function testing(;
#     T = Float32,
#     seed = 42,
#     n = 1_000,
#     p = 10,
#     l = 5,
#     F = Activations.relu,
#     F_derivative = Activations.relu_derivative,
#     phen_type = ["random", "linear", "non-linear"][3],
#     normalise_X = true,
#     n_epochs = 1_000,
#     optimiser = ["GD", "Adam", "AdamMax"][2],
#     η = 0.001,
#     h² = 0.75,
#     p_unobserved = 1,
# )
#     out = Dict("ρ_lm" => NaN, "ρ_dl" => NaN)

#     Random.seed!(seed); CUDA.seed!(seed); 
#     hs = vcat(repeat([p], l), 1); 
#     W::Vector{CuArray{T, 2}} = [CUDA.randn(T, hs[1], p)]; 
#     b::Vector{CuArray{T, 1}} = [CUDA.randn(T, hs[1])]; 
#     [(push!(W, CUDA.randn(T, hs[i+1], hs[i])), push!(b, CUDA.randn(T, hs[i+1]))) for i in 1:l]; 

#     X = rand(Bool, n, p)
#     ϕ = if phen_type == "random"
#         rand(n)
#     elseif phen_type == "linear"
#         β = rand(p)
#         σ²g = var(X*β)
#         σ²e = σ²g*((1/h²) - 1.00)
#         N = Distributions.Normal(0.0, σ²e)
#         e = rand(N, n)
#         ϕ = (X*β) .+ e
#         # # ϕ = (ϕ .- mean(ϕ)) ./ std(ϕ) # messes up with predictions
#         # b̂ = inv(X'*X) * (X'*ϕ)
#         # ŷ = X * b̂
#         # cor(ŷ, ϕ)
#         ϕ
#     elseif phen_type == "non-linear"
#         p_true = p + p_unobserved
#         X_true = rand(Bool, n, p_true)
#         W_true::Vector{CuArray{T, 2}} = [CUDA.randn(T, hs[1], p_true)]
#         b_true::Vector{CuArray{T, 1}} = [CUDA.randn(T, hs[1])]
#         for i in 1:l
#             push!(W_true, CUDA.randn(T, hs[i+1], hs[i]))
#             push!(b_true, CUDA.randn(T, hs[i+1]))
#         end
#         FP = ForwardBackward.forwardpass(
#             X=CuArray{T, 2}(Matrix(X_true')), 
#             W=W_true, 
#             b=b_true, 
#             F=F
#         )
#         ϕ = Matrix(FP["ŷ"])[1, :]
#         ϕ
#     else
#         throw("Please use `phen_type=\"random\"` or `phen_type=\"linear\"` or `phen_type=\"non-linear\"`")
#     end
#     X = normalise_X ? (X .- mean(X, dims=1)) ./ std(X, dims=1) : X
#     display(UnicodePlots.histogram(ϕ, title="Target phenotype"))
#     X = CuArray{T, 2}(Matrix(X'))
#     y = CuArray{T, 2}(reshape(ϕ, (1, n)))
#     begin
#         b̂ = inv(X * X') * (X * y')
#         ŷ = b̂' * X
#         out["ρ_lm"] = cor(Matrix(ŷ)[1, :], Matrix(y)[1, :])
#         display(UnicodePlots.scatterplot(Matrix(ŷ)[1, :], Matrix(y)[1, :]))
#         @show out["ρ_lm"]
#     end

#     epochs = []
#     loss = []
#     state::Dict{String, Any} = Dict(
#         "β₁" => T(0.900),
#         "β₂" => T(0.999),
#         "ϵ" => T(1e-7),
#         "t" => T(0.0),
#         "m_W" => [CuArray{T,2}(zeros(size(x))) for x in W],
#         "v_W" => [CuArray{T,2}(zeros(size(x))) for x in W],
#         "m_b" => [CuArray{T,1}(zeros(size(x))) for x in b],
#         "v_b" => [CuArray{T,1}(zeros(size(x))) for x in b],
#     );
#     ∇W = ForwardBackward.initgradient(W)
#     ∇b = ForwardBackward.initgradient(b)
#     pb = ProgressMeter.Progress(n_epochs, desc="Training")
#     for i in 1:n_epochs
#         # i = 1
#         FP = ForwardBackward.forwardpass(X=X, W=W, b=b, F=F)
#         ForwardBackward.backpropagation!(∇W, ∇b; y=y, ŷ=CuArray(FP["ŷ"]), W=W, S=Vector(FP["S"]), A=Vector(FP["A"]), δC=Costs.MSE_derivative, δF=F_derivative)
#         if optimiser == "GD"
#             Optimisers.gradientdescent!(W, b, ∇W, ∇b, η=T(η))
#         elseif optimiser == "Adam"
#             Optimisers.Adam!(W, b, ∇W, ∇b, η=T(η), state=state)
#         elseif optimiser == "AdamMax"
#             Optimisers.AdamMax!(W, b, ∇W, ∇b, η=T(η), state=state)
#         else
#             throw("Please use `optimiser=\"GD\"` or `optimiser=\"Adam\"` or `optimiser=\"AdamMax\"`")
#         end
#         push!(epochs, i)
#         push!(loss, sum(Costs.MSE(ForwardBackward.forwardpass(X=X, W=W, b=b, F=F)["ŷ"], y)))
#         ProgressMeter.next!(pb)
#     end
#     ProgressMeter.finish!(pb)
#     out["ρ_dl"] = cor(Matrix(ForwardBackward.forwardpass(X=X, W=W, b=b, F=F)["ŷ"])[1, :], Matrix(y)[1, :])
#     display(UnicodePlots.scatterplot(epochs, loss))
#     display(UnicodePlots.scatterplot(Matrix(ForwardBackward.forwardpass(X=X, W=W, b=b, F=F)["ŷ"])[1, :], Matrix(y)[1, :]))
#     @show out["ρ_dl"]

#     out
# end

@testset "MLPPCVE" begin
    # testing()
    # testing(p_unobserved=0)
    # testing(p_unobserved=0, n_epochs=10_000)
    # testing(phen_type="random")
    # testing(phen_type="random", n_epochs=10_000)
    # testing(phen_type="linear")
    # testing(phen_type="linear", l=1)
    # testing(phen_type="linear", l=1, n_epochs=20_000)

    # vec_outs_not_norm_X = []
    # vec_outs_norm_X = []
    # for i in 1:10
    #     push!(vec_outs_not_norm_X, testing(seed=i, normalise_X=false, n_epochs=2_000))
    #     push!(vec_outs_norm_X, testing(seed=i, normalise_X=true, n_epochs=2_000))
    # end
    # if mean([x["ρ_dl"] for x in vec_outs_not_norm_X]) < mean([x["ρ_dl"] for x in vec_outs_norm_X])
    #     println("Better normalise X!")
    # else
    #     println("Better NOT normalise X!")
    # end
    # hcat(([x["ρ_dl"] for x in vec_outs_not_norm_X]), ([x["ρ_dl"] for x in vec_outs_norm_X]))

    @assert 0 == 0

end
