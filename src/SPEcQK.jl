module SPEcQK


const EXP_CLAMP = 200.0  # Clamp input to avoid overflow/underflow in exp(...)
include("dual_solver.jl")
include("solver_wrappers.jl")

end # module SPEcQK
