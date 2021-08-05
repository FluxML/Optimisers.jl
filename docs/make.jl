using Documenter, Optimisers

DocMeta.setdocmeta!(Optimisers, :DocTestSetup, :(using Optimisers); recursive = true)

makedocs(modules = [Optimisers],
         doctest = VERSION == v"1.6",
         sitename = "Optimisers.jl",
         pages = ["Home" => "optimisers.md",
                  "API" => "api.md"],
         format = Documenter.HTML(
             analytics = "UA-36890222-9",
             assets = ["assets/flux.css"],
             prettyurls = get(ENV, "CI", nothing) == "true"),
         )

deploydocs(repo = "github.com/FluxML/Optimisers.jl.git",
           target = "build",
           push_preview = true)
