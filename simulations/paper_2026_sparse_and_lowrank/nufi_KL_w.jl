using SPEcQK
using HDF5

function run(target_vdf, output_prefix, target_vdf_name, n_v, extent, lambda_values_unscaled,
             max_moment_constraint; output=true, threshold=1e-6)


    moment_powers_constraint = all_powers_up_to_M_3D(max_moment_constraint)
    moment_powers_test = all_powers_up_to_M_3D(max_moment_constraint+1)
    moment_powers_test = [x for x in moment_powers_test if !(x in moment_powers_constraint)]

    grid = Grid3D([-extent, extent], [-extent, extent], [-extent, extent], n_v, n_v, n_v)
    gw = grid_weights(grid)
    Δv = unroll(gw)
    lambda_values_scaled = lambda_values_unscaled .* sum(Δv) / length(Δv)
    
    # compute target VDF
    vdf_hidden_truth = VDF3D(grid)
    target_vdf(vdf_hidden_truth, grid)
    vdf_hidden_truth_unrolled = unroll(vdf_hidden_truth.w)

    # used to store weighting distribution
    vdf_w = VDF3D(grid)

    mmm_constraint = construct_moment_measurement_matrix_3D(grid, n_v, moment_powers_constraint)
    mmm_test = construct_moment_measurement_matrix_3D(grid, n_v, moment_powers_test)

    ref_moms_constraint = mmm_constraint * vdf_hidden_truth_unrolled
    ref_moms_test = mmm_test * vdf_hidden_truth_unrolled

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

    vdf_w = VDF3D(grid)
    target_vdf(vdf_w, grid)

    vdf_w.w[vdf_w.w .<= 1e-14] .= 1e-14
    w = unroll(vdf_w.w)
    n = length(w)

    println("Total DOFs: $n")
    println("Min(w) = $(minimum(w)), Max(w) = $(maximum(w))")
    target_KL = copy(w)

    solutions, cconstraint_predictions, test_predictions, sparsity = solve_iterate_over_L1_values(lambda_values_scaled,
                                 w, target_KL, Δv,
                                 mmm_constraint, mmm_test,
                                 ref_moms_constraint, ref_moms_test, n_v; threshold=threshold,
                                 tol=1e-7, maxiter=100,
                                 mu=1e-12, backtrack_rho=0.5, backtrack_c=1e-4, find_y0=true,
                                 verbose=2, warmup=true)

    if output
        io_path = "$(output_prefix)/$(target_vdf_name)_constraint_M_upto$(max_moment_constraint)_$(n_v).h5"

        write_grid_and_vdf_and_solution_iterated_over_L1_values(io_path, target_vdf_name,
                                                                vdf_hidden_truth, grid,
                                                                moment_powers_constraint, moment_powers_test,
                                                                mmm_constraint, mmm_test,
                                                                ref_moms_constraint, ref_moms_test,
                                                                cconstraint_predictions, test_predictions,
                                                                solutions, lambda_values_unscaled, vdf_w.w,
                                                                threshold,
                                                                sparsity)
    end
end

const write_output = true
const step = 2

arr = zeros((256, 256, 256))

const grid_size_per_dir = 256 ÷ step


# 0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 
const lambda_values_unscaled = [0.0, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0]
const sparse_threshold = 1e-7

for t in [300, 200, 100]  # [100, 200, 300]

    h5open("$(ARGS[1])/hdf5_$(t).h5", "r") do f
        arr[:,:,:] .= read(f["f"])
    end
    arr_lower = arr[1:step:end, 1:step:end, 1:step:end]
    nufi_vdf(a, b) = (a, b) => read_in_vdf!(a, b, arr_lower)

    run(nufi_vdf, "output", "KL_VDFw_NuFI_$(t)_$(step)", grid_size_per_dir, 1.0, lambda_values_unscaled, 4; output=write_output, threshold=sparse_threshold)
    run(nufi_vdf, "output", "KL_VDFw_NuFI_$(t)_$(step)", grid_size_per_dir, 1.0, lambda_values_unscaled, 6; output=write_output, threshold=sparse_threshold)
end