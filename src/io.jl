using HDF5

"""
    write_grid_and_vdf_and_solution_iterated_over_L1_values(path, vdf_name, vdf_hidden_truth,
                                                            grid, mp_t, mp_c, mmms_t, mmms_c,
                                                            rmv_t, rmv_c,
                                                            pmv_t, pmv_c,
                                                            solutions, lambdas, weighting_function,
                                                            threshold_value,
                                                            sparsity_values)

Write solutions produced by `solve_iterate_over_L1_values` to a HDF5 file.
The full VDF can be then reconstructed by taking a `solutions[:,:,:,i]` and multiplying it with
the `weighting_function`.

# Positional arguments:
* `path`: path to output file
* `vdf_name`: name of VDF used for testing
* `vdf_hidden_truth`: a VDF3D instance containing the underlying "hidden truth" distribution
* `grid`: the velocity grid used
* `mp_c`: vector of 3-tuples of moment powers for the moments acting as constraints
* `mp_t`: vector of 3-tuples of moment powers for the moments used for testing
* `mmms_c`: moment measurement matrix for the moments acting as constraints
* `mmms_t`: moment measurement matrix for the moments used for testing
* `rmv_c`: vector of reference moments for the moments acting as constraints
* `rmv_t`: vector of reference moments for the moments used for testing
* `pmv_c`: matrix of predicted moments for the moments acting as constraints (`m x length(lambdas)`)
* `pmv_t`: matrix of predicted moments for the moments used for testing (`m x length(lambdas)`)
* `solutions`: array of size `(n_vx, n_vy, n_vz, n_lambdas)` containing the computed solutions for each L1 regularization value
* `lambdas`: vector of L1 regularization values
* `weighting_function`: array of size `(n_vx, n_vy, n_vz)` containing the weighting function used
* `threshold_value`: threshold value used for the solution to set values below this threshold to zero
* `sparsity_values`: vector of length `n_lambdas` containing the amount of zero values for each solution
"""
function write_grid_and_vdf_and_solution_iterated_over_L1_values(path, vdf_name, vdf_hidden_truth,
                                                                 grid, mp_c, mp_t, mmms_c, mmms_t,
                                                                 rmv_c, rmv_t,
                                                                 pmv_c, pmv_t,
                                                                 solutions, lambdas, weighting_function,
                                                                 threshold_value,
                                                                 sparsity_values)
    
    vdf_hidden_truth_unrolled = unroll(vdf_hidden_truth.w)
    
    tensor_grid = zeros((grid.n_vx, grid.n_vy, grid.n_vz, 3))

    for k in 1:grid.n_vz
        for j in 1:grid.n_vy
            for i in 1:grid.n_vx
                tensor_grid[i,j,k,1] = grid.vx[i]
                tensor_grid[i,j,k,2] = grid.vy[j]
                tensor_grid[i,j,k,3] = grid.vz[k]
            end
        end
    end

    @assert size(mmms_c)[2] == size(mmms_t)[2] == grid.n_vx*grid.n_vy*grid.n_vz
    @assert length(mp_t) == size(mmms_t)[1] == length(rmv_t)
    @assert length(mp_c) == size(mmms_c)[1] == length(rmv_c)

    # for k in 1:length(lambdas)
    #     @assert length(predicted_moment_values[k]) == length(mp_all)
    # end

    h5open(path, "w") do file
        write(file, "vdf_hidden_truth", vdf_hidden_truth.w)
        write(file, "vdf_hidden_truth_unrolled", vdf_hidden_truth_unrolled)
        write(file, "grid", tensor_grid)
        write(file, "weighting_function", weighting_function)
        write(file, "lambdas", lambdas)
        write(file, "solutions", solutions)
        write(file, "sparsity_degree", sparsity_values)

        attributes(file["vdf_hidden_truth"])["description"] = "$(vdf_name) vdf"
        attributes(file["vdf_hidden_truth_unrolled"])["description"] = "$(vdf_name) vdf unrolled "
        attributes(file["grid"])["description"] = "$(grid.n_vx) x $(grid.n_vy) x $(grid.n_vz) grid"
        attributes(file["weighting_function"])["description"] = "weighting function, f_ijk=w_ijk * g_ijk"
        attributes(file["solutions"])["description"] = "solutions, solution[i,j,k,l] = g_ijk with lambda=lambdas[l]"
        attributes(file["sparsity_degree"])["description"] = "# of elements smaller than threshold; threshold=$(threshold_value)"

        for (mom, rmv) in zip(mp_c, rmv_c)
            write(file, "ref_M_constraint_$(mom[1]),$(mom[2]),$(mom[3])", rmv)
        end
        for (mom, rmv) in zip(mp_t, rmv_t)
            write(file, "ref_M_test_$(mom[1]),$(mom[2]),$(mom[3])", rmv)
        end

        write(file, "moment_measurement_matrix_constraints", mmms_c) 
        write(file, "moment_measurement_matrix_testing", mmms_t) 

        for (i,mom) in enumerate(mp_c)
            write(file, "predicted_M_$(mom[1]),$(mom[2]),$(mom[3])", pmv_c[i,:])
        end
        for (i,mom) in enumerate(mp_t)
            write(file, "predicted_M_$(mom[1]),$(mom[2]),$(mom[3])", pmv_t[i,:])
        end
    end
end