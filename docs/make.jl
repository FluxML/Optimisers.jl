using Documenter, Optimisers, Zygote, StaticArrays, Functors

DocMeta.setdocmeta!(Optimisers, :DocTestSetup, :(using Optimisers, Functors); recursive = true)
DocMeta.setdocmeta!(Functors, :DocTestSetup, :(using Functors); recursive = true)

makedocs(modules = [Optimisers, Functors],
         doctest = false,
         sitename = "Optimisers.jl",
         pages = ["Home" => "index.md",
                  "API" => "api.md"],
         format = Documenter.HTML(
             canonical = "https://fluxml.ai/Optimisers.jl/stable/",
             analytics = "UA-36890222-9",
             assets = ["assets/flux.css"],
             prettyurls = get(ENV, "CI", nothing) == "true"
         ),
         checkdocs = :none, # don't check that Functors' docstrings are all reported here
)

deploydocs(
   repo = "github.com/FluxML/Optimisers.jl.git",
   target = "build",
   push_preview = true,
)
