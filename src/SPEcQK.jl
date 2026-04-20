module SPEcQK

using MuladdMacro

const EXP_CLAMP = 200.0  # Clamp input to avoid overflow/underflow in exp(...)
include("utils.jl")
include("velocity_grid.jl")
include("moments.jl")
include("dual_solver.jl")
include("distributions.jl")
include("initial_solution.jl")
include("solver_wrappers.jl")
include("io.jl")

export Grid3D, VDF3D
export all_powers_up_to_M_3D
export construct_moment_measurement_matrix_3D
export maxwell_boltzmann!, druyvesteyn!
export unroll, grid_weights
export find_index
export find_MB_solution!
export solve_iterate_over_L1_values
export full_solve_with_init!
export write_grid_and_vdf_and_solution_iterated_over_L1_values
export build_next_moment_index_direction

end # module SPEcQK
