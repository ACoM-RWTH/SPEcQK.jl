using Documenter, SPEcQK

makedocs(sitename="SPEcQK", remotes = nothing,
         pages = [
        "Home" => "index.md",
        "Scaling" => "scaling.md",
        "API Reference" => "api.md",
        ])