module SPEcQK


const EXP_CLAMP = 200.0  # Clamp input to avoid overflow/underflow in exp(...)
include("velocity_grid.jl")
include("moments.jl")
include("dual_solver.jl")
include("initial_solution.jl")
include("solver_wrappers.jl")
include("distributions.jl")

end # module SPEcQK
