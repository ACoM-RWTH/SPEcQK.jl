@muladd begin

"""
    compute_E(vdf::VDF3D{N_vx, N_vy, N_vz}, grid, w) where {N_vx, N_vy, N_vz}

Compute the specific energy of the VDF `vdf` given the velocity grid `grid` and the density `w`.
"""
function compute_E(vdf::VDF3D{N_vx, N_vy, N_vz}, grid, w) where {N_vx, N_vy, N_vz}
    # compute current value of the energy
    # we assume vx=vy=vz and density w is known!

    E = 0.0
    @inbounds for k in 1:N_vz
        for j in 1:N_vy
            for i in 1:N_vx
                
                vsq = grid.vxsq[i]+grid.vysq[j]+grid.vzsq[k]
                E += vdf.w[i,j,k] * vsq * grid.Δvx[i] * grid.Δvy[j] * grid.Δvz[k]
            end
        end
    end

    return E/w
end

"""
    de_dT(grid::Grid3D{N_vx,N_vy,N_vz}, T) where {N_vx, N_vy, N_vz}

Derivative of energy w.r.t T (since density is also affected due to grid cut-off).
"""
function de_dT(grid::Grid3D{N_vx,N_vy,N_vz}, T) where {N_vx, N_vy, N_vz}
    # d exp(-a/x) / dx = a exp(-a/x) / x^2
    # e = C^2 exp(-C^2/T) => de/dT = C^4 exp(-C^2 / T) / T^2
    # e = E/w, de/dT = (e'w - w'e)/w^2

    dE = 0.0
    E = 0.0
    dw = 0.0

    w = 0.0
    inv_T = 1.0/T
    @inbounds for k in 1:grid.n_vz
        for j in 1:grid.n_vy
            for i in 1:grid.n_vx
                vsq = grid.vxsq[i]+grid.vysq[j]+grid.vzsq[k]
                # we don't need the grid spacing because we're dividing by the density in the loop anyway
                # so it's just a constant multiplier in the numerator and denominator
                exp_val = exp(-vsq*inv_T) # * grid.Δvx[i] * delta_vy_vz
                w += exp_val
                dE += vsq^2 * exp_val  # E = 
                E += vsq * exp_val
                dw += vsq * exp_val
            end
        end
    end

    return (dE * w - E * dw) / (T^2 * w^2)
end

"""
    find_T!(T0, E_target, vdf, grid, w_tot, tol)

Newton-Raphson iteration to find the temperature T that gives the target energy E_target.
"""
function find_T!(T0, E_target, vdf, grid, w_tot, tol)
    iter = 0

    T = T0

    maxwell_boltzmann!(vdf, grid, T0)
    E = compute_E(vdf, grid, w_tot)

    r = E - E_target

    while abs(r) > tol
        T -= r / de_dT(grid, T)

        maxwell_boltzmann!(vdf, grid, T)
        E = compute_E(vdf, grid, w_tot)
        r = E - E_target

        iter += 1
    end
    return T
end

"""
    find_MB_solution!(vdf_mb, w0, ref_moms_constraint, vx_index, vy_index, vz_index,
                      Ex_index, Ey_index, Ez_index)

Approximate the target VDF with a Maxwell-Boltzmann distribution with target velocity
and energy.
This finds the temperature by solving a root finding problem, but then shifts
the Maxwell-Boltzmann distribution by the streaming velocity, so the final
distribution might not have the same velocity and/or energy due to the grid cut-off.
**Important**: this assumes the density of distribution is 1. If this is not the case,
the `ref_moms_constraint` should be re-scaled accordingly, and the resulting `vdf_mb`
should be multiplied by the density afterwards.

# Positional arguments
* `vdf_mb`: a VDF3D instance where computed solution will be written
* `ref_moms_constraint`: a vector of reference moment values
* `vx_index`: index of the (1,0,0) moment in `ref_moms_constraint`
* `vy_index`: index of the (0,1,0) moment in `ref_moms_constraint`
* `vz_index`: index of the (0,0,1) moment in `ref_moms_constraint`
* `Ex_index`: index of the (2,0,0) moment in `ref_moms_constraint`
* `Ey_index`: index of the (0,2,0) moment in `ref_moms_constraint`
* `E`_index`: index of the (0,0,2) moment in `ref_moms_constraint`

# Keyword arguments
* `tol`: tolerance for the Newton-Raphson method
"""
function find_MB_solution!(vdf_mb, ref_moms_constraint, vx_index, vy_index, vz_index,
                           Ex_index, Ey_index, Ez_index; tol=1e-11)

    vx0 = ref_moms_constraint[vx_index]
    vy0 = ref_moms_constraint[vy_index]
    vz0 = ref_moms_constraint[vz_index]
    
    E_target = ref_moms_constraint[Ex_index] - vx0^2
    E_target += ref_moms_constraint[Ey_index] - vy0^2
    E_target += ref_moms_constraint[Ez_index] - vz0^2
    
    T0 = (2.0/3.0) * E_target
    
    T_sol = find_T!(T0, E_target, vdf_mb, grid, 1.0, tol)

    maxwell_boltzmann!(vdf_mb, grid, vx0, vy0, vz0, T_sol)
end
end