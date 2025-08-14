function sampleNormal(n::Int64; μ::T=0.0, σ::T=1.00)::Vector{T} where T <: AbstractFloat
    # n = 100; μ = 0.0; σ = 1.0;
    U = rand(n)
    V = rand(n)
    z = sqrt.(-2.00 .* log.(U)) .* cos.(2.00*π .* V)
    T.((z .* σ) .+ μ)
end

function meanX(X::Matrix{T}; dims::Int64=1)::Matrix{T} where T <: AbstractFloat
    # X = rand(10, 4)
    dims > 2 ? throw("Maximum `dims` is 2 because `X` is a matrix, i.e. 2-dimensional") : nothing
    n = size(X, dims)
    sum(X, dims=dims) ./ n
end

function varX(X::Matrix{T}; dims::Int64=1)::Matrix{T} where T <: AbstractFloat
    # X = rand(10, 4)
    dims > 2 ? throw("Maximum `dims` is 2 because `X` is a matrix, i.e. 2-dimensional") : nothing
    n = size(X, dims)
    μ = sum(X, dims=dims) ./ n
    sum((X .- μ).^2, dims=dims) ./ (n-1)
end

function stdX(X::Matrix{T}; dims::Int64=1)::Matrix{T} where T <: AbstractFloat
    sqrt.(varX(X, dims=dims))
end

function simulate(;
    T::Type = Float32,
    seed::Int64 = 42,
    n::Int64 = 1_000,
    p::Int64 = 10,
    l::Int64 = 5,
    F::Function = relu,
    phen_type::String = ["random", "linear", "non-linear"][3],
    normalise_X::Bool = true,
    h²::Float64 = 0.75,
    p_unobserved::Int64 = 1,
)::Tuple{CuArray{T, 2}, CuArray{T, 2}}
    Random.seed!(seed); CUDA.seed!(seed); 
    X = rand(Bool, n, p)
    ϕ = if phen_type == "random"
        rand(n)
    elseif phen_type == "linear"
        β = rand(p, 1)
        σ²g = varX(X*β, dims=1)[1, 1]
        σ²e = σ²g*((1/h²) - 1.00)
        e = sampleNormal(n, μ=T(0.00), σ=T(sqrt(σ²e)))
        (X*β) .+ e
    elseif phen_type == "non-linear"
        p_true = p + p_unobserved
        X_true = rand(Bool, n, p_true)
        hs = vcat(repeat([p], l), 1); 
        W_true::Vector{CuArray{T, 2}} = [CUDA.randn(T, hs[1], p_true)]
        b_true::Vector{CuArray{T, 1}} = [CUDA.randn(T, hs[1])]
        for i in 1:l
            push!(W_true, CUDA.randn(T, hs[i+1], hs[i]))
            push!(b_true, CUDA.randn(T, hs[i+1]))
        end
        FP = forwardpass(
            X=CuArray{T, 2}(Matrix(X_true')), 
            W=W_true, 
            b=b_true, 
            F=F
        )
        Matrix(FP["ŷ"])[1, :]
    else
        throw("Please use `phen_type=\"random\"` or `phen_type=\"linear\"` or `phen_type=\"non-linear\"`")
    end
    X = normalise_X ? (X .- meanX(T.(X), dims=1)) ./ stdX(T.(X), dims=1) : X
    X = CuArray{T, 2}(Matrix(X'))
    y = CuArray{T, 2}(reshape(ϕ, (1, n)))
    (X, y)
end

# simulate()
# simulate(phen_type="random")
# simulate(phen_type="linear")
