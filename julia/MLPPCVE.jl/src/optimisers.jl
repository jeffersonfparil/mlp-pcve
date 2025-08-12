module Optimisers

using CUDA

T = Float32; n = 1_000; p = 3; l = 1; hs = vcat(repeat([p], l), 1); 
X = CuArray{T}(rand(Bool, p, n)); 
W_true::Vector{CuArray{T}} = [CUDA.randn(T, hs[1], p)]; 
B_true::Vector{CuArray{T}} = [CUDA.randn(T, hs[1], 1)];
[(push!(W_true, CUDA.randn(T, hs[i+1], hs[i])), push!(B_true, CUDA.randn(T, hs[i+1], 1))) for i in 1:l]; 
F = Activations.relu; 
FP = ForwardBackward.forwardpass(X=X, W=W_true, B=B_true, F=F)
y = FP["ŷ"]; 
δF = Activations.relu_derivative; 
δC = Costs.MSE_derivative; 
C = Costs.MSE; 
begin
    @assert sum(C(y, y)) == T(0.0)

    BP = ForwardBackward.backpropagation(y=y, ŷ=FP["ŷ"], S=FP["weighted_sums"], A=FP["activations"], W=W_true, δC=δC, δF=δF); 
    @assert sum([sum(x .!= 0.0) for x in BP["∇W"]]) == 0
    @assert sum([sum(x .!= 0.0) for x in BP["∇B"]]) == 0
    W_duplicate = deepcopy(W_true)
    B_duplicate = deepcopy(B_true)
    Optimisers.gradientdescent!(W_duplicate, B_duplicate, BP["∇W"], BP["∇B"], r=T(0.0001))
    @assert W_true == W_duplicate
    @assert B_true == B_duplicate

    CUDA.@allowscalar W_duplicate[1][1,1] = 0.0
    FP = ForwardBackward.forwardpass(X=X, W=W_duplicate, B=B_duplicate, F=F)
    ŷ = FP["ŷ"]; 
    sum(C(y, ŷ))

    BP = ForwardBackward.backpropagation(y=y, ŷ=FP["ŷ"], S=FP["weighted_sums"], A=FP["activations"], W=W_duplicate, δC=δC, δF=δF); 
    [sum(x .!= 0.0) for x in BP["∇W"]]
    [sum(x .!= 0.0) for x in BP["∇B"]]
    Optimisers.gradientdescent!(W_duplicate, B_duplicate, BP["∇W"], BP["∇B"], r=T(0.01))
    FP = ForwardBackward.forwardpass(X=X, W=W_duplicate, B=B_duplicate, F=F)
    ŷ = FP["ŷ"]; 
    sum(C(y, ŷ))
end
# #############
W::Vector{CuArray{T}} = [CUDA.randn(T, hs[1], p)]; 
B::Vector{CuArray{T}} = [CUDA.randn(T, hs[1])];
[(push!(W, CUDA.randn(T, hs[i+1], hs[i])), push!(B, CUDA.randn(T, hs[i+1]))) for i in 1:l]; 
begin
    FP = ForwardBackward.forwardpass(X=X, W=W, B=B, F=F)
    ŷ = FP["ŷ"]; 
    C = Costs.MSE; 
    sum(C(y, ŷ))
end
for i in 1:100
    # i = 1
    FP = ForwardBackward.forwardpass(X=X, W=W, B=B, F=F)
    println("i=$i; loss=$(sum(C(y, FP["ŷ"]))/length(y))")
    BP = ForwardBackward.backpropagation(y=y, ŷ=FP["ŷ"], S=FP["weighted_sums"], A=FP["activations"], W=W, δC=δC, δF=δF); 
    Optimisers.gradientdescent!(W, B, BP["∇W"], BP["∇B"], r=T(0.001))
end

function gradientdescent!(
    W::Vector{CuArray{T}},
    B::Vector{CuArray{T}},
    ∇W::Vector{CuArray{T}},
    ∇B::Vector{CuArray{T}};
    r::T = 0.001,
) where T <: AbstractFloat
    # r = 0.01
    for i in eachindex(W)
        # i = 1
        W[i] = W[i] - (r .* ∇W[i])
        B[i] = B[i] - (r .* ∇B[i])
    end
end






function Adam()
end

end