module SPEcQK

using StaticArrays
using MuladdMacro
using TimerOutputs

const EXP_CLAMP = 200.0  # Clamp input to avoid overflow/underflow in exp(...)
include("utils.jl")
include("velocity_grid.jl")
include("maxwellian_tp.jl")
include("moments.jl")
include("dual_solver.jl")
include("distributions.jl")
include("initial_solution.jl")
include("solver_wrappers.jl")
include("io.jl")
include("bgk.jl")
include("1d_lf.jl")
include("1d_dvm.jl")

export Grid3D, VDF3D
export all_powers_up_to_M_3D
export construct_moment_measurement_matrix_3D
export maxwell_boltzmann!, druyvesteyn!, mott_smith!, read_in_vdf!
export MaxwellianTP, maxwell_boltzmann_axes!, densify!
export axis_moment_table!, axis_sq_moment_table!, separable_moments!, separable_row_norms!
export unroll, grid_weights
export find_index
export find_MB_solution!
export find_MB_solution_T_and_v!, find_T_and_v!
export solve_iterate_over_L1_values
export full_solve_with_init!
export write_grid_and_vdf_and_solution_iterated_over_L1_values
export build_next_moment_index_direction
export tau_BGK
export run_1d, convect_1D_LF!, maxwellian_constraint_moments
export lax_friedrichs_wall_bc!, lax_wendroff_wall_bc!, shared_limiter!
export kinetic_upwind_wall_bc!, upwind_lax_wendroff_wall_bc!
export run_1d_dvm, convect_1D_DVM!
export init_moment_io, write_moment_snapshot!, close_moment_io
export Knudsen_number

end # module SPEcQK
