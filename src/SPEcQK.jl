module SPEcQK

using MuladdMacro

const EXP_CLAMP = 200.0  # Clamp input to avoid overflow/underflow in exp(...)
include("velocity_grid.jl")
include("moments.jl")
include("dual_solver.jl")
include("distributions.jl")
include("initial_solution.jl")
include("solver_wrappers.jl")
include("io.jl")

end # module SPEcQK
