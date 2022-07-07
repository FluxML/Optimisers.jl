using Documenter, Optimisers, Zygote, StaticArrays, Functors

DocMeta.setdocmeta!(Optimisers, :DocTestSetup, :(using Optimisers); recursive = true)

makedocs(modules = [Optimisers, Zygote, StaticArrays, Functors],
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
)

deploydocs(
   repo = "github.com/FluxML/Optimisers.jl.git",
   target = "build",
   push_preview = true,
)
