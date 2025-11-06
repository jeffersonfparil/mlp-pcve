using MLPPCVE
using Documenter

DocMeta.setdocmeta!(MLPPCVE, :DocTestSetup, :(using MLPPCVE); recursive = true)

makedocs(;
    modules = [MLPPCVE],
    authors = "jeffersonparil@gmail.com",
    sitename = "MLPPCVE.jl",
    format = Documenter.HTML(;
        canonical = "https://jeffersonfparil.github.io/MLPPCVE.jl",
        edit_link = "main",
        assets = String[],
        size_threshold = 1000000,
    ),
    pages = ["Home" => "index.md"],
)

deploydocs(;
    repo = "github.com/jeffersonfparil/mlppcve/julia/MLPPCVE.jl",
    devbranch = "main",
)
