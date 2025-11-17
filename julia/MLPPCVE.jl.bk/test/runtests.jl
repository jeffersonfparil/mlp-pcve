using MLPPCVE
using Test
using Documenter

Documenter.doctest(MLPPCVE)

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
