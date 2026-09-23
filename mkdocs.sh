julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs -e 'using Pkg; Pkg.add("Documenter")'
julia --project=docs docs/make.jl