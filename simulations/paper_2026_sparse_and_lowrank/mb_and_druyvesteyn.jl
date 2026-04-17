using SPEcQK

function run(target_vdf, output_prefix, target_vdf_name, n_v, extent, lambda_values_unscaled,
             max_moment_constraint; output=true)


    moment_powers_constraint = all_powers_up_to_M_3D(max_moment_constraint)
    moment_powers_test = all_powers_up_to_M_3D(max_moment_constraint+1)
    moment_powers_test = [x for x in moment_powers_test if !(x in moment_powers_constraint)]

    grid = Grid3D([-extent, extent], [-extent, extent], [-extent, extent], n_v, n_v, n_v)
    gw = grid_weights(grid)
    Δv = unroll(gw)
    lambda_values_scaled = lambda_values_unscaled .* sum(Δv) / length(Δv)
    
    # compute target VDF
    vdf = VDF3D(grid)
    target_vdf(vdf, grid)
    vdf_unrolled = unroll(vdf.w)

    # used to store weighting M-B distribution
    vdf_mb = VDF3D(grid)

    mmm_constraint = construct_moment_measurement_matrix_3D(grid, n_v, moment_powers_constraint)
    mmm_test = construct_moment_measurement_matrix_3D(grid, n_v, moment_powers_test)

    ref_moms_constraint = mmm_constraint * vdf_unrolled
    ref_moms_test = mmm_test * vdf_unrolled

    A = zeros((length(moment_powers_constraint), n_v^3))

    println("Conserving all moments up to $max_moment_constraint")
    println("$(length(moment_powers_constraint)) constraints")
    println("Conserved moment indices: $moment_powers_constraint")
    println("Will test on $moment_powers_test")
    println("lambda values unscaled=$(lambda_values_unscaled)")
    println("lambda values scaled=$(lambda_values_scaled)")
    println("Reference moments: $ref_moms_constraint")
    println("$n_v x $n_v x $n_v grid with extent [-$extent, $extent]")

    vx_index = find_index(moment_powers_constraint, (1,0,0))
    vy_index = find_index(moment_powers_constraint, (0,1,0))
    vz_index = find_index(moment_powers_constraint, (0,0,1))

    Ex_index = find_index(moment_powers_constraint, (2,0,0))
    Ey_index = find_index(moment_powers_constraint, (0,2,0))
    Ez_index = find_index(moment_powers_constraint, (0,0,2))
    println("Indices for 1st and 2nd moments: $vx_index $vy_index $vz_index; $Ex_index $Ey_index $Ez_index")
    println("Sanity checks: vx: $(moment_powers_constraint[vx_index])")
    println("vy: $(moment_powers_constraint[vy_index])")
    println("vz: $(moment_powers_constraint[vz_index])")
    println("Sanity checks: Ex: $(moment_powers_constraint[Ex_index])")
    println("Ey: $(moment_powers_constraint[Ey_index])")
    println("Ez: $(moment_powers_constraint[Ez_index])")

    T_MB = find_MB_solution!(vdf_mb, grid, ref_moms_constraint, vx_index, vy_index, vz_index,
                             Ex_index, Ey_index, Ez_index; tol=1e-11)
    println("T(M-B approximation) = $(T_MB)")

    w = unroll(vdf_mb.w)
    n = length(w)

    println("Total DOFs: $n")
    target_KL = ones(n)

    solutions, constraint_moments_predicted, next_moments_predicted, sparsity = solve_iterate_over_L1_values(lambda_values_unscaled,
                                 w, target_KL, Δv,
                                 mmm_constraint, mmm_test,
                                 ref_moms_constraint, ref_moms_test, n_v; threshold=1e-6,
                                 tol=1e-9, maxiter=100,
                                 mu=1e-12, backtrack_rho=0.5, backtrack_c=1e-4, find_y0=true,
                                 verbose=2, warmup=true)
                                 
    io_path = "$(output_prefix)/$(target_vdf_name)_constraint_M_upto$(max_moment_constraint)_$(n_v).h5"
end

const write_output = false
const mb_vdf(a,b) = maxwell_boltzmann!(a, b, 1.0)

const lambda_values_unscaled = [0.0, 0.5]
run(mb_vdf, "output", "Maxwell_Boltzmann", 20, 4.0, lambda_values_unscaled, 5; output=write_output)