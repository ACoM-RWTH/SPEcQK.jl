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
    compute_E(mb::MaxwellianTP{N_vx, N_vy, N_vz}, grid, w) where {N_vx, N_vy, N_vz}

Tensor-product version for a separable Maxwellian: the energy factorizes into
per-axis sums, `E = C (Ex Ny Nz + Nx Ey Nz + Nx Ny Ez)` with `N = S[0]`,
`E = S[2]` per axis. Costs `O(n_v)` instead of `O(n_v³)`.
"""
function compute_E(mb::MaxwellianTP{N_vx, N_vy, N_vz}, grid, w) where {N_vx, N_vy, N_vz}
    Nx = 0.0; Ex = 0.0
    @inbounds for i in 1:N_vx
        g = mb.gx[i] * grid.Δvx[i]
        Nx += g
        Ex += grid.vxsq[i] * g
    end
    Ny = 0.0; Ey = 0.0
    @inbounds for j in 1:N_vy
        g = mb.gy[j] * grid.Δvy[j]
        Ny += g
        Ey += grid.vysq[j] * g
    end
    Nz = 0.0; Ez = 0.0
    @inbounds for k in 1:N_vz
        g = mb.gz[k] * grid.Δvz[k]
        Nz += g
        Ez += grid.vzsq[k] * g
    end

    E = mb.C * (Ex * Ny * Nz + Nx * Ey * Nz + Nx * Ny * Ez)
    return E / w
end

"""
    de_dT(grid::Grid3D{N_vx,N_vy,N_vz}, T) where {N_vx, N_vy, N_vz}

Derivative of energy w.r.t T (since density is also affected due to grid cut-off).
"""
function de_dT(grid::Grid3D{N_vx,N_vy,N_vz}, T) where {N_vx, N_vy, N_vz}
    # d exp(-a/x) / dx = a exp(-a/x) / x^2
    # e = C^2 exp(-C^2/T) => de/dT = C^4 exp(-C^2 / T) / T^2
    # e = E/w, de/dT = (e'w - w'e)/w^2
    #
    # The zero-centered Gaussian is separable, so all four sums expand into
    # per-axis raw sums of order 0/2/4: 3 n_v exps instead of n_v³.
    # As before, the grid spacing is a constant multiplier that cancels in the
    # ratio and is omitted.

    inv_T = 1.0/T

    ax0 = 0.0; ax2 = 0.0; ax4 = 0.0
    @inbounds for i in 1:grid.n_vx
        v2 = grid.vxsq[i]
        e = exp(-v2*inv_T)
        ax0 += e; ax2 += v2 * e; ax4 += v2 * v2 * e
    end
    ay0 = 0.0; ay2 = 0.0; ay4 = 0.0
    @inbounds for j in 1:grid.n_vy
        v2 = grid.vysq[j]
        e = exp(-v2*inv_T)
        ay0 += e; ay2 += v2 * e; ay4 += v2 * v2 * e
    end
    az0 = 0.0; az2 = 0.0; az4 = 0.0
    @inbounds for k in 1:grid.n_vz
        v2 = grid.vzsq[k]
        e = exp(-v2*inv_T)
        az0 += e; az2 += v2 * e; az4 += v2 * v2 * e
    end

    w = ax0 * ay0 * az0
    # Σ vsq exp = Σ (vx²+vy²+vz²) exp — also equals dw
    E = ax2 * ay0 * az0 + ax0 * ay2 * az0 + ax0 * ay0 * az2
    dw = E
    # Σ vsq² exp: (vx²+vy²+vz²)² = Σ v_α⁴ + 2 Σ_{α<β} v_α² v_β²
    dE = ax4 * ay0 * az0 + ax0 * ay4 * az0 + ax0 * ay0 * az4 +
         2.0 * (ax2 * ay2 * az0 + ax2 * ay0 * az2 + ax0 * ay2 * az2)

    return (dE * w - E * dw) / (T^2 * w^2)
end

"""
    find_T!(T0, E_target, vdf, grid, w_tot, tol)

Newton-Raphson iteration to find the temperature T that gives the target energy E_target.
The iteration runs entirely in per-axis (tensor-product) space — `vdf` is kept
for API compatibility but is neither read nor written; callers that need the
dense distribution must fill it afterwards (e.g. via `maxwell_boltzmann!`).
"""
function find_T!(T0, E_target, vdf, grid, w_tot, tol)
    # <v²> of the zero-centered density-1 discrete Maxwellian, per-axis:
    # E = Ex/Nx + Ey/Ny + Ez/Nz
    energy = function (T)
        inv_T = 1.0 / T
        nx, _, ex, _, _ = axis_shifted_raw_sums(grid.vx, grid.Δvx, 0.0, inv_T)
        ny, _, ey, _, _ = axis_shifted_raw_sums(grid.vy, grid.Δvy, 0.0, inv_T)
        nz, _, ez, _, _ = axis_shifted_raw_sums(grid.vz, grid.Δvz, 0.0, inv_T)
        return (ex / nx + ey / ny + ez / nz) / w_tot
    end

    iter = 0

    T = T0

    E = energy(T0)

    r = E - E_target

    while abs(r) > tol && iter < 1000
        T -= r / de_dT(grid, T)

        E = energy(T)
        r = E - E_target

        iter += 1
    end
    return T
end

"""
    find_MB_solution!(vdf_mb, grid, ref_moms_constraint, vx_index, vy_index, vz_index,
                      Ex_index, Ey_index, Ez_index; tol=1e-11)

Approximate the target VDF with a Maxwell-Boltzmann distribution with target velocity
and energy.
This finds the temperature by solving a root finding problem, but then shifts
the Maxwell-Boltzmann distribution by the streaming velocity, so the final
distribution might not have the same velocity and/or energy due to the grid cut-off.
**Important**: this assumes the moments passed to the function are **NOT** divided by the number density.

# Positional arguments
* `vdf_mb`: a VDF3D instance where computed solution will be written
* `grid`: the velocity grid
* `ref_moms_constraint`: a vector of reference moment values
* `ndens_index`: index of the (0,0,0) moment in `ref_moms_constraint`
* `vx_index`: index of the (1,0,0) moment in `ref_moms_constraint`
* `vy_index`: index of the (0,1,0) moment in `ref_moms_constraint`
* `vz_index`: index of the (0,0,1) moment in `ref_moms_constraint`
* `Ex_index`: index of the (2,0,0) moment in `ref_moms_constraint`
* `Ey_index`: index of the (0,2,0) moment in `ref_moms_constraint`
* `E`_index`: index of the (0,0,2) moment in `ref_moms_constraint`

# Keyword arguments
* `tol`: tolerance for the Newton-Raphson method

# Returns
* `T_sol`: computed temperature
"""
function find_MB_solution!(vdf_mb, grid, ref_moms_constraint, ndens_index, vx_index, vy_index, vz_index,
                           Ex_index, Ey_index, Ez_index; tol=1e-11)
    n = ref_moms_constraint[ndens_index]
    vx0 = ref_moms_constraint[vx_index] / n
    vy0 = ref_moms_constraint[vy_index] / n
    vz0 = ref_moms_constraint[vz_index] / n
    
    E_target = ref_moms_constraint[Ex_index] / n - vx0^2
    E_target += ref_moms_constraint[Ey_index] / n - vy0^2
    E_target += ref_moms_constraint[Ez_index] / n - vz0^2
    
    T0 = (2.0/3.0) * E_target
    
    T_sol = find_T!(T0, E_target, vdf_mb, grid, 1.0, tol)

    maxwell_boltzmann!(vdf_mb, grid, n, vx0, vy0, vz0, T_sol)

    return T_sol
end

"""
    find_T_and_v!(vdf, grid, ux_t, uy_t, uz_t, E2_t, ux0, uy0, uz0, T0; tol=1e-11, maxiter=50)

Newton solve for the streaming velocity `(ux,uy,uz)` and temperature `T` of a
discrete (grid-cut-off) Maxwell–Boltzmann distribution so that its mean velocity
matches `(ux_t,uy_t,uz_t)` and its full second moment `<vx²+vy²+vz²>` matches
`E2_t`. Unlike `find_T!`, this accounts for the fact that shifting a cut-off
Maxwellian by a velocity changes both its mean velocity and its energy, so it
solves all four unknowns jointly.

The Jacobian of the density-1 moment `<φ_a>` w.r.t. parameter `θ_b` is the
covariance `<φ_a ψ_b> - <φ_a><ψ_b>`, with `φ = (vx, vy, vz, vx²+vy²+vz²)` and
`ψ = (2(vx-ux)/T, 2(vy-uy)/T, 2(vz-uz)/T, r²/T²)`, `r² = |v-u|²`.

All moments are assembled from per-axis raw sums of order ≤ 4 of the
separable Maxwellian (`O(n_v)` per iteration, no dense fill). `vdf` is kept
for API compatibility but is neither read nor written; callers that need the
dense distribution must fill it afterwards (e.g. via `maxwell_boltzmann!`).

# Keyword arguments
* `tol`: convergence tolerance on the residual norm
* `maxiter`: maximum number of Newton iterations

# Returns
A NamedTuple `(T, ux, uy, uz, converged, iters)`.
"""
function find_T_and_v!(vdf, grid, ux_t, uy_t, uz_t, E2_t,
                       ux0, uy0, uz0, T0; tol=1e-11, maxiter=50)
    ux = ux0; uy = uy0; uz = uz0; T = T0

    converged = false
    iters = 0

    for it in 1:maxiter
        iters = it
        inv_T  = 1.0 / T
        inv_T2 = inv_T * inv_T

        # per-axis raw sums S_a = Σ v^a g Δv of the (unnormalized) separable
        # Maxwellian; every entry of mphi/mpsi/mpp is a per-axis moment of
        # degree ≤ 4 and factorizes into products of the normalized s_a
        ax0, ax1, ax2, ax3, ax4 = axis_shifted_raw_sums(grid.vx, grid.Δvx, ux, inv_T)
        ay0, ay1, ay2, ay3, ay4 = axis_shifted_raw_sums(grid.vy, grid.Δvy, uy, inv_T)
        az0, az1, az2, az3, az4 = axis_shifted_raw_sums(grid.vz, grid.Δvz, uz, inv_T)

        # normalized per-axis moments s_a = <v^a> along each axis (density 1)
        sx1 = ax1/ax0; sx2 = ax2/ax0; sx3 = ax3/ax0; sx4 = ax4/ax0
        sy1 = ay1/ay0; sy2 = ay2/ay0; sy3 = ay3/ay0; sy4 = ay4/ay0
        sz1 = az1/az0; sz2 = az2/az0; sz3 = az3/az0; sz4 = az4/az0

        # central per-axis moments <(v-u)²>, <v (v-u)²>, <v² (v-u)>, <v² (v-u)²>
        cx2 = sx2 - 2.0*ux*sx1 + ux*ux
        cy2 = sy2 - 2.0*uy*sy1 + uy*uy
        cz2 = sz2 - 2.0*uz*sz1 + uz*uz
        tx3 = sx3 - 2.0*ux*sx2 + ux*ux*sx1
        ty3 = sy3 - 2.0*uy*sy2 + uy*uy*sy1
        tz3 = sz3 - 2.0*uz*sz2 + uz*uz*sz1
        wx3 = sx3 - ux*sx2
        wy3 = sy3 - uy*sy2
        wz3 = sz3 - uz*sz2
        qx4 = sx4 - 2.0*ux*sx3 + ux*ux*sx2
        qy4 = sy4 - 2.0*uy*sy3 + uy*uy*sy2
        qz4 = sz4 - 2.0*uz*sz3 + uz*uz*sz2

        mphi = SVector(sx1, sy1, sz1, sx2 + sy2 + sz2)                    # <φ_a>
        mpsi = SVector(2.0*(sx1 - ux)*inv_T, 2.0*(sy1 - uy)*inv_T,        # <ψ_b>
                       2.0*(sz1 - uz)*inv_T, (cx2 + cy2 + cz2)*inv_T2)

        # <φ_a ψ_b>
        mpp = @SMatrix [2.0*(sx2 - ux*sx1)*inv_T   2.0*sx1*(sy1 - uy)*inv_T   2.0*sx1*(sz1 - uz)*inv_T   (tx3 + sx1*(cy2 + cz2))*inv_T2;
                        2.0*sy1*(sx1 - ux)*inv_T   2.0*(sy2 - uy*sy1)*inv_T   2.0*sy1*(sz1 - uz)*inv_T   (ty3 + sy1*(cx2 + cz2))*inv_T2;
                        2.0*sz1*(sx1 - ux)*inv_T   2.0*sz1*(sy1 - uy)*inv_T   2.0*(sz2 - uz*sz1)*inv_T   (tz3 + sz1*(cx2 + cy2))*inv_T2;
                        2.0*(wx3 + (sy2 + sz2)*(sx1 - ux))*inv_T   2.0*(wy3 + (sx2 + sz2)*(sy1 - uy))*inv_T   2.0*(wz3 + (sx2 + sy2)*(sz1 - uz))*inv_T   (qx4 + qy4 + qz4 + sx2*(cy2 + cz2) + sy2*(cx2 + cz2) + sz2*(cx2 + cy2))*inv_T2]

        R = mphi - SVector(ux_t, uy_t, uz_t, E2_t)

        if norm(R) < tol
            converged = true
            break
        end

        J = mpp - mphi * mpsi'

        Δ = J \ R
        ux -= Δ[1]; uy -= Δ[2]; uz -= Δ[3]
        newT = T - Δ[4]
        T = newT > 0.0 ? newT : 0.5 * T   # keep the temperature positive
    end

    return (T=T, ux=ux, uy=uy, uz=uz, converged=converged, iters=iters)
end

"""
    find_MB_solution_T_and_v!(vdf_mb, grid, ref_moms_constraint, ndens_index,
                              vx_index, vy_index, vz_index,
                              Ex_index, Ey_index, Ez_index; tol=1e-11, maxiter=50)

Like `find_MB_solution!`, but the resulting Maxwell–Boltzmann distribution matches
BOTH the target streaming velocity AND the target energy on the cut-off grid, by
solving for `(ux,uy,uz,T)` jointly with `find_T_and_v!`. On return `vdf_mb` holds
the solution scaled to the target density.

**Important**: moments in `ref_moms_constraint` are **NOT** divided by the number density.

# Returns
A NamedTuple `(T, ux, uy, uz, converged, iters)` from `find_T_and_v!`.
"""
function find_MB_solution_T_and_v!(vdf_mb, grid, ref_moms_constraint, ndens_index,
                                   vx_index, vy_index, vz_index,
                                   Ex_index, Ey_index, Ez_index; tol=1e-11, maxiter=50)
    n = ref_moms_constraint[ndens_index]
    ux_t = ref_moms_constraint[vx_index] / n
    uy_t = ref_moms_constraint[vy_index] / n
    uz_t = ref_moms_constraint[vz_index] / n
    E2_t = (ref_moms_constraint[Ex_index] +
            ref_moms_constraint[Ey_index] +
            ref_moms_constraint[Ez_index]) / n

    # initial guess from the ideal-gas relations (exact for an un-cut Maxwellian)
    T0 = (2.0 / 3.0) * (E2_t - (ux_t^2 + uy_t^2 + uz_t^2))

    # if E2_t < 0
    #     println("n=$n")
    #     println("ux_t=$ux_t uy_t=$uy_t uz_t=$uz_t")
    #     println(ref_moms_constraint[Ex_index], " ", ref_moms_constraint[Ey_index], " ", ref_moms_constraint[Ez_index])
    # end
    sol = find_T_and_v!(vdf_mb, grid, ux_t, uy_t, uz_t, E2_t,
                        ux_t, uy_t, uz_t, T0; tol=tol, maxiter=maxiter)

    # write the final distribution scaled to the target density
    maxwell_boltzmann!(vdf_mb, grid, n, sol.ux, sol.uy, sol.uz, sol.T)

    return sol
end
end