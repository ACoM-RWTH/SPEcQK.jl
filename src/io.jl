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

"""
    init_moment_io(path, x, moment_powers_constraint, m_constraint, n_cells; attrs...)

Open an HDF5 file for streaming time-evolution output from the 1D solver and
create extendible datasets `/moments` (m_constraint × n_cells × n_snap),
`/T_MB` (n_cells × n_snap), `/phi` (n_cells × n_snap), `/times`, `/steps`.
Static fields `/x` and `/moment_powers` are written once. Run metadata passed as
keyword arguments is stored as attributes on `/moments`.

`/T_MB` holds the temperature of the discretely matched Maxwellian (the
grid-cut-off-consistent temperature); snapshots written without it carry NaN.

`/phi` holds the Lax–Wendroff blending factor, `phi[j]` being the interface
*to the right of* interior cell `j` (at `x_min + j*dx`); the last entry is the
right wall interface, which is not a limited interface and is always 0.
Snapshots written without it carry NaN — so it is all-NaN for every first-order
and every DVM run, exactly as `/T_MB` already is for `run_1d`.

Returns an io handle (NamedTuple) to pass to `write_moment_snapshot!`; close it
with `close_moment_io`.
"""
function init_moment_io(path, x, moment_powers_constraint, m_constraint, n_cells; attrs...)
    fid = h5open(path, "w")
    write(fid, "x", collect(x))
    write(fid, "moment_powers", reduce(hcat, [collect(p) for p in moment_powers_constraint]))

    dM = create_dataset(fid, "moments", Float64,
                        ((m_constraint, n_cells, 1), (m_constraint, n_cells, -1));
                        chunk=(m_constraint, n_cells, 1))
    dT = create_dataset(fid, "times", Float64, ((1,), (-1,)); chunk=(1,))
    dS = create_dataset(fid, "steps", Int,     ((1,), (-1,)); chunk=(1,))
    dTMB = create_dataset(fid, "T_MB", Float64,
                          ((n_cells, 1), (n_cells, -1));
                          chunk=(n_cells, 1))
    dPHI = create_dataset(fid, "phi", Float64,
                          ((n_cells, 1), (n_cells, -1));
                          chunk=(n_cells, 1))

    # self-describing run metadata as attributes on /moments
    for (k, v) in pairs(attrs)
        attributes(dM)[string(k)] = v
    end
    attributes(dM)["description"] = "moments[m, cell, snapshot]; constraint moment m at interior cell"
    attributes(dTMB)["description"] = "T_MB[cell, snapshot]; temperature of the discretely matched Maxwellian (NaN if not computed)"
    attributes(dPHI)["description"] = "phi[cell, snapshot]; Lax-Wendroff blending factor at the interface right of the cell (NaN if not applicable)"

    return (fid=fid, M=dM, T=dT, S=dS, TMB=dTMB, PHI=dPHI, n=Ref(0))
end

"""
    write_moment_snapshot!(io, step, time, moments_interior, T_mb_interior=nothing;
                           phi_interior=nothing)

Append one snapshot (`moments_interior` is m_constraint × n_cells) to the
extendible datasets of an io handle from `init_moment_io`. `T_mb_interior`
(length n_cells) is the matched-Maxwellian temperature profile; when omitted
the `/T_MB` slice is filled with NaN. `phi_interior` (length n_cells) is the
Lax–Wendroff blending factor per interface, likewise NaN-filled when omitted —
it is a keyword so that the DVM and first-order callers need no change.
"""
function write_moment_snapshot!(io, step, time, moments_interior, T_mb_interior=nothing;
                                phi_interior=nothing)
    k = (io.n[] += 1)
    HDF5.set_extent_dims(io.M, (size(io.M, 1), size(io.M, 2), k))
    io.M[:, :, k] = moments_interior
    HDF5.set_extent_dims(io.T, (k,)); io.T[k] = time
    HDF5.set_extent_dims(io.S, (k,)); io.S[k] = step
    HDF5.set_extent_dims(io.TMB, (size(io.TMB, 1), k))
    io.TMB[:, k] = isnothing(T_mb_interior) ? fill(NaN, size(io.TMB, 1)) : T_mb_interior
    HDF5.set_extent_dims(io.PHI, (size(io.PHI, 1), k))
    io.PHI[:, k] = isnothing(phi_interior) ? fill(NaN, size(io.PHI, 1)) : phi_interior
    return nothing
end

"""
    close_moment_io(io)

Close the HDF5 file held by an io handle from `init_moment_io`.
"""
close_moment_io(io) = close(io.fid)