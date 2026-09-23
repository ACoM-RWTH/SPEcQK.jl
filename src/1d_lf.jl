"""
    lax_friedrichs_wall_bc!(moments_next, moments_current, fluxes_current,
                            wall_flux_left, wall_flux_right,
                            dt, dx, max_eigenvalue, n_cells, n_moments)

First-order transport of the `n_moments` state moments. Interior interfaces use
the Lax–Friedrichs flux `0.5(f_L+f_R) - 0.5*max_eigenvalue*(u_R-u_L)` (f = the
next-order x-moment in `fluxes_current`). The two wall interfaces — left of cell
2 and right of cell `n_cells+1` — use the externally supplied kinetic fluxes
`wall_flux_left`/`wall_flux_right` (see `kinetic_wall_flux!`). State arrays are
sized `(_, n_cells+2)`; interior cells are columns `2..n_cells+1`. Ghost columns
are no longer read.
"""
function lax_friedrichs_wall_bc!(moments_next, moments_current, fluxes_current,
                                 wall_flux_left, wall_flux_right,
                                 dt, dx, max_eigenvalue, n_cells, n_moments)
    inv_dx = 1.0 / dx

    @inbounds for m in 1:n_moments
        for i in 2:(n_cells+1)
            u_C = moments_current[m, i]

            # left interface flux (i - 1/2)
            if i == 2
                flux_left = wall_flux_left[m]                       # kinetic wall flux
            else
                f_L = fluxes_current[m, i-1]; f_C = fluxes_current[m, i]
                u_L = moments_current[m, i-1]
                flux_left = 0.5 * (f_L + f_C) - 0.5 * max_eigenvalue * (u_C - u_L)
            end

            # right interface flux (i + 1/2)
            if i == n_cells + 1
                flux_right = wall_flux_right[m]                     # kinetic wall flux
            else
                f_C = fluxes_current[m, i]; f_R = fluxes_current[m, i+1]
                u_R = moments_current[m, i+1]
                flux_right = 0.5 * (f_C + f_R) - 0.5 * max_eigenvalue * (u_R - u_C)
            end

            moments_next[m, i] = u_C - dt * inv_dx * (flux_right - flux_left)
        end
    end

    return nothing
end

"""
    kinetic_upwind_wall_bc!(moments_next, moments_current, fplus, fminus,
                            wall_flux_left, wall_flux_right,
                            dt, dx, n_cells, n_moments)

First-order transport of the `n_moments` state moments with the **per-node
kinetic upwind** flux. Same layout and wall treatment as
`lax_friedrichs_wall_bc!`, but the interior interface flux is

    F_{i+1/2} = A⁺[f_i] + A⁻[f_{i+1}] = fplus[:,i] + fminus[:,i+1]

where `A±` are the half-space moment operators
`A±_j[f] = Σ_{±v_x>0} m_j(v) v_x f Δv` (see `mmm_plus`/`mmm_minus` in `run_1d`).
Both halves are moments of a single cell's own reconstructed VDF, so — exactly
as for the Lax–Wendroff correction — no interface reconstruction and no extra
dual solves are needed.

This is the identical formula to `kinetic_wall_flux!` with the neighbouring
cell's VDF replacing `σ_w M_wall`, so interior and wall interfaces become the
same operator. Note there is no `max_eigenvalue`: the dissipation is
`a_k = |v_k|` per node rather than the global `a = extent`, which is where the
~5x reduction in numerical viscosity comes from (cf. `run_1d_dvm`'s
`dissipation=:upwind`). The time step is unchanged — stability is still set by
`max |v_x| = extent`.
"""
function kinetic_upwind_wall_bc!(moments_next, moments_current, fplus, fminus,
                                 wall_flux_left, wall_flux_right,
                                 dt, dx, n_cells, n_moments)
    inv_dx = 1.0 / dx

    @inbounds for m in 1:n_moments
        for i in 2:(n_cells+1)
            flux_left  = i == 2           ? wall_flux_left[m]  : fplus[m, i-1] + fminus[m, i]
            flux_right = i == n_cells + 1 ? wall_flux_right[m] : fplus[m, i]   + fminus[m, i+1]
            moments_next[m, i] = moments_current[m, i] - dt * inv_dx * (flux_right - flux_left)
        end
    end

    return nothing
end

"""
    limiter_vanleer(d_up, d_loc, scale)

van Leer limiter `ψ(r) = (r+|r|)/(1+|r|)` with `r = d_up/d_loc`, clamped to
`[0, 1]`. `d_up` is the upwind-side difference of the indicator, `d_loc` the
difference across the interface being limited, `scale` a magnitude used to
decide when `d_loc` counts as zero.

The clamp to 1 is deliberate: the Sweby TVD region admits `ψ` up to 2
but capping at 1 means the correction never exceeds
full Lax–Wendroff. `ψ(1) = 1` so smooth regions still get the full second-order
flux.

A locally flat indicator (`|d_loc|` below `scale`) is smooth, not an extremum, so
it returns 1 rather than 0, otherwise a uniform density would veto the
correction for every other moment sharing this interface. See `shared_limiter!`
for how `scale` must be calibrated.
"""
@inline function limiter_vanleer(d_up, d_loc, scale)
    abs(d_loc) <= scale && return 1.0        # locally flat => smooth
    d_up * d_loc <= 0.0 && return 0.0        # extremum / sign change => full LF
    r = d_up / d_loc
    return min((r + abs(r)) / (1.0 + abs(r)), 1.0)
end

"""
    shared_limiter!(phi, ind_n, ind_T, moments_current, density_index,
                    vx_index, vy_index, vz_index, Ex_index, Ey_index, Ez_index,
                    n_cells; rel_flat=1e-3)

Fills `phi[i]` — the Lax–Wendroff blending factor for the interior interface
between cells `i` and `i+1`, for `i in 2:n_cells`, with a single scalar per
interface, shared by every moment component.

Limiting each of the `m_constraint` components independently would produce moment
vectors that are not moments of any positive VDF, which the entropic dual solver
then fails to invert. Driving one scalar from a few physical indicators and
applying it uniformly keeps the correction parallel to a realizable direction.

The indicators are density and temperature, both read from the evolved constraint
moments (`ind_n`/`ind_T` are scratch of length `n_cells+2`).

Each indicator contributes a van Leer factor from a symmetric pair of ratios
(backward- and forward-biased), because a kinetic system carries information in
both directions regardless of the sign of the bulk velocity; `phi` is the minimum
over both directions and both indicators. Interfaces whose 4-cell stencil runs
off the interior fall back to whichever direction is available, and to `phi = 0`
(pure Lax–Friedrichs) if neither is.

`rel_flat` sets when an indicator counts as locally flat, as a fraction of that
indicator's **variation across the domain** (`max - min` over the interior), not
of its magnitude.
"""
function shared_limiter!(phi, ind_n, ind_T, moments_current, density_index,
                         vx_index, vy_index, vz_index, Ex_index, Ey_index, Ez_index,
                         n_cells; rel_flat=1e-3)
    # --- per-cell indicators from the constraint moments ---
    @inbounds for i in 2:(n_cells+1)
        nl = moments_current[density_index, i]
        ind_n[i] = nl
        if nl > 0.0
            ux = moments_current[vx_index, i] / nl
            uy = moments_current[vy_index, i] / nl
            uz = moments_current[vz_index, i] / nl
            E2 = (moments_current[Ex_index, i] + moments_current[Ey_index, i] +
                  moments_current[Ez_index, i]) / nl
            ind_T[i] = (E2 - (ux^2 + uy^2 + uz^2)) / 3.0
        else
            ind_T[i] = 0.0      # non-positive density: let the density veto decide
        end
    end

    # --- flatness scale per indicator, from its variation across the interior ---
    lo_n = hi_n = ind_n[2]
    lo_T = hi_T = ind_T[2]
    @inbounds for i in 3:(n_cells+1)
        lo_n = min(lo_n, ind_n[i]); hi_n = max(hi_n, ind_n[i])
        lo_T = min(lo_T, ind_T[i]); hi_T = max(hi_T, ind_T[i])
    end
    scale_n = rel_flat * (hi_n - lo_n)
    scale_T = rel_flat * (hi_T - lo_T)

    fill!(phi, 0.0)

    @inbounds for i in 2:n_cells                     # interface between cells i, i+1
        have_back = (i - 1) >= 2                     # cell i-1 is interior
        have_fwd  = (i + 2) <= (n_cells + 1)         # cell i+2 is interior
        if !have_back && !have_fwd
            continue                                 # phi stays 0 => Lax-Friedrichs
        end

        p = 1.0
        for (s, scale) in ((ind_n, scale_n), (ind_T, scale_T))
            d_loc = s[i+1] - s[i]
            if have_back
                p = min(p, limiter_vanleer(s[i] - s[i-1], d_loc, scale))
            end
            if have_fwd
                p = min(p, limiter_vanleer(s[i+2] - s[i+1], d_loc, scale))
            end
        end
        phi[i] = p
    end

    return nothing
end

"""
    lax_wendroff_wall_bc!(moments_next, moments_current, fluxes_current,
                          fluxes2_current, wall_flux_left, wall_flux_right, phi,
                          dt, dx, max_eigenvalue, n_cells, n_moments)

Flux-limited Lax–Wendroff transport of the `n_moments` state moments. Same
layout and wall treatment as `lax_friedrichs_wall_bc!`; interior interfaces use

    F = F_LF + phi * (F_LW - F_LF)

with `phi` from `shared_limiter!`. `phi = 0` recovers `lax_friedrichs_wall_bc!`
exactly and `phi = 1` gives pure Lax–Wendroff.

The high-order flux needs no flux Jacobian. Transport is linear advection at
speed `v_k` for every velocity node, so node-wise Lax–Wendroff projected onto the
constraint moment `(a,b,c)` is

    F_LW = 0.5*(u1_L + u1_R) - (dt/2dx)*(u2_R - u2_L)

where `u1` is the `(a+1,b,c)` moment (`fluxes_current`, as in the LF flux) and
`u2` the `(a+2,b,c)` moment (`fluxes2_current`). Both are moments of each cell's
*own* reconstructed VDF, read from `moments_work` via `next_dir_index_x` and
`next2_dir_index_x` — no interface reconstruction, so no extra dual solves.

Limiting is per *interface*, not per cell: both cells sharing an interface see
the same blended flux, so the fluxes still telescope and mass conservation stays
exact.
"""
function lax_wendroff_wall_bc!(moments_next, moments_current, fluxes_current,
                               fluxes2_current, wall_flux_left, wall_flux_right, phi,
                               dt, dx, max_eigenvalue, n_cells, n_moments)
    inv_dx = 1.0 / dx
    lw_coef = 0.5 * dt * inv_dx

    @inbounds for m in 1:n_moments
        for i in 2:(n_cells+1)
            u_C = moments_current[m, i]

            # left interface flux (i - 1/2)
            if i == 2
                flux_left = wall_flux_left[m]                      # kinetic wall flux
            else
                f_L = fluxes_current[m, i-1];  f_C = fluxes_current[m, i]
                g_L = fluxes2_current[m, i-1]; g_C = fluxes2_current[m, i]
                u_L = moments_current[m, i-1]
                diff_lf = 0.5 * max_eigenvalue * (u_C - u_L)
                diff_lw = lw_coef * (g_C - g_L)
                flux_left = 0.5 * (f_L + f_C) - diff_lf + phi[i-1] * (diff_lf - diff_lw)
            end

            # right interface flux (i + 1/2)
            if i == n_cells + 1
                flux_right = wall_flux_right[m]                    # kinetic wall flux
            else
                f_C = fluxes_current[m, i];  f_R = fluxes_current[m, i+1]
                g_C = fluxes2_current[m, i]; g_R = fluxes2_current[m, i+1]
                u_R = moments_current[m, i+1]
                diff_lf = 0.5 * max_eigenvalue * (u_R - u_C)
                diff_lw = lw_coef * (g_R - g_C)
                flux_right = 0.5 * (f_C + f_R) - diff_lf + phi[i] * (diff_lf - diff_lw)
            end

            moments_next[m, i] = u_C - dt * inv_dx * (flux_right - flux_left)
        end
    end

    return nothing
end

"""
    upwind_lax_wendroff_wall_bc!(moments_next, moments_current, fluxes_current,
                                 fluxes2_current, fplus, fminus,
                                 wall_flux_left, wall_flux_right, phi,
                                 dt, dx, n_cells, n_moments)

Flux-limited Lax–Wendroff transport on the **kinetic upwind** low-order
baseline. Identical to `lax_wendroff_wall_bc!` except that the first-order flux
is `kinetic_upwind_wall_bc!`'s instead of Lax–Friedrichs':

    F = F_up + phi * (F_LW - F_up)
    F_up = fplus[:,L] + fminus[:,R]
    F_LW = 0.5*(u1_L + u1_R) - (dt/2dx)*(u2_R - u2_L)

`F_LW` is exactly the same high-order flux as in `lax_wendroff_wall_bc!` — the
low-order baseline only sets the anti-diffusive increment `F_LW - F_low`, so
`phi = 1` gives the same pure Lax–Wendroff flux for either baseline. What
changes is the size of the increment: against Lax–Friedrichs it is ~20–80x the
Lax–Wendroff term, against upwind only ~2–5x, so whatever spatial pattern `phi`
carries is imprinted an order of magnitude more weakly on the solution.

Written in increment form (`F_up + phi*(F_LW - F_up)`) rather than as
`0.5*(u1_L+u1_R) - 0.5*(D_R-D_L) + phi*(…)` with `D = fplus - fminus`: the two
are algebraically the same, but this form is bitwise `kinetic_upwind_wall_bc!`
at `phi = 0`, and it does not rely on `fplus + fminus == fluxes` holding in
floating point (it does not — the two are summed in different orders).
"""
function upwind_lax_wendroff_wall_bc!(moments_next, moments_current, fluxes_current,
                                      fluxes2_current, fplus, fminus,
                                      wall_flux_left, wall_flux_right, phi,
                                      dt, dx, n_cells, n_moments)
    inv_dx = 1.0 / dx
    lw_coef = 0.5 * dt * inv_dx

    @inbounds for m in 1:n_moments
        for i in 2:(n_cells+1)
            u_C = moments_current[m, i]

            # left interface flux (i - 1/2)
            if i == 2
                flux_left = wall_flux_left[m]                      # kinetic wall flux
            else
                f_up = fplus[m, i-1] + fminus[m, i]
                f_lw = 0.5 * (fluxes_current[m, i-1] + fluxes_current[m, i]) -
                       lw_coef * (fluxes2_current[m, i] - fluxes2_current[m, i-1])
                flux_left = f_up + phi[i-1] * (f_lw - f_up)
            end

            # right interface flux (i + 1/2)
            if i == n_cells + 1
                flux_right = wall_flux_right[m]                    # kinetic wall flux
            else
                f_up = fplus[m, i] + fminus[m, i+1]
                f_lw = 0.5 * (fluxes_current[m, i] + fluxes_current[m, i+1]) -
                       lw_coef * (fluxes2_current[m, i+1] - fluxes2_current[m, i])
                flux_right = f_up + phi[i] * (f_lw - f_up)
            end

            moments_next[m, i] = u_C - dt * inv_dx * (flux_right - flux_left)
        end
    end

    return nothing
end

"""
    wall_flux_denominator(M_wall, vx, Δv, n, normal_sign)

Half-space mass flux of the static wall Maxwellian,
`Σ_{normal_sign·v_x > 0} v_x M_wall Δv` — the denominator of the diffuse
re-emission factor `σ_w` in `kinetic_wall_flux!`. `M_wall` never changes, so
this is computed once per driver and passed via `wallbc`.
"""
function wall_flux_denominator(M_wall, vx, Δv, n, normal_sign)
    denom = 0.0
    @inbounds for k in 1:n
        vk = vx[k]
        if normal_sign * vk > 0.0
            denom += vk * M_wall[k] * Δv[k]
        end
    end
    return denom
end

"""
    kinetic_wall_flux!(wall_flux, fhalf, f_cell, M_wall, vx, Δv,
                       mmm_constraint, n, m_constraint, normal_sign, denom)

Diffuse-reflection wall flux (Baranger et al. 2019, eqns (10)–(14) and §3.1.1).
`normal_sign = +1` for the left wall (gas-side normal `+x`), `-1` for the right
wall (`-x`). `f_cell` is the reconstructed VDF in the boundary cell, `M_wall` the
density-1 wall Maxwellian, `vx` the per-node x-velocity, `Δv` the quadrature
weights, `denom` the precomputed half-space wall-Maxwellian mass flux
(`wall_flux_denominator`).

The per-node upwind interface flux is `F_k = v_x^+ f_left + v_x^- f_right`, where
the gas-side cell supplies the outgoing half and the wall Maxwellian (scaled by
`σ_w`) the incoming half. `σ_w` enforces zero net mass flux through the wall
(eqn (14)). The moment fluxes are `mmm_constraint * F`. Returns `σ_w`.
"""
function kinetic_wall_flux!(wall_flux, fhalf, f_cell, M_wall, vx, Δv,
                            mmm_constraint, n, m_constraint, normal_sign, denom)
    # σ_w from zero mass flux: incoming half (re-emitted, precomputed denom)
    # balances the outgoing half
    numer = 0.0   # Σ_outgoing v * f_cell * Δv
    @inbounds for k in 1:n
        vk = vx[k]
        if !(normal_sign * vk > 0.0)    # outgoing toward the wall -> in-cell VDF
            numer += vk * f_cell[k] * Δv[k]
        end
    end
    sigma_w = -numer / denom

    # per-node upwind flux F_k = v^+ f_left + v^- f_right at the wall interface
    @inbounds for k in 1:n
        vk = vx[k]
        vp = vk > 0.0 ? vk : 0.0
        vm = vk < 0.0 ? vk : 0.0
        if normal_sign > 0              # left wall: ghost (diffuse) on left, cell on right
            fhalf[k] = vp * (sigma_w * M_wall[k]) + vm * f_cell[k]
        else                            # right wall: cell on left, ghost (diffuse) on right
            fhalf[k] = vp * f_cell[k] + vm * (sigma_w * M_wall[k])
        end
    end

    # moment fluxes (mmm already carries Δv): wall_flux[j] = Σ_k mmm[j,k] F_k
    @inbounds for j in 1:m_constraint
        s = 0.0
        for k in 1:n
            s += mmm_constraint[j, k] * fhalf[k]
        end
        wall_flux[j] = s
    end

    return sigma_w
end

"""
    populate_fluxes_1d!(fluxes_current, moments_current, next_dir_index_x, n_cells_total, n_moments)

Populates the physical flux array for 1D transport.
Assumes `n_cells_total` includes the ghost cells (e.g., n_cells + 2).
"""
function populate_fluxes_1d!(fluxes_current, moments_work, next_dir_index_x, n_cells_total, n_moments)
    @inbounds for i in 1:n_cells_total
        for m in 1:n_moments
            flux_idx = next_dir_index_x[m]
            # Flux of constraint moment (a,b,c) is the next x-moment (a+1,b,c),
            # located in moments_work via the cross-set index next_dir_index_x.
            # With all = up_to(M+1) over constraint = up_to(M) this is always > 0.
            fluxes_current[m, i] = moments_work[flux_idx, i]
        end
    end

    return nothing
end

function compute_moments_from_vdf!(moments, col, gsol, w, mmm, n, m)
    @inbounds for j in 1:m
        s = 0.0
        for i in 1:n
            s += mmm[j, i] * gsol[i] * w[i]
        end
        moments[j, col] = s
    end

    return nothing
end

function convect_1D_LF!(moments_all, dt, dx, max_eigenvalue, n_t,
                       n_cells, n_moments_all, n_moments_constraint,
                       n_v, mmm_constraint, mmm_all, Δv, λ, omega,
                       A, rnorm_A, mvec, alpha, gsol, uvec, d_w, inv_dw,
                       y0, y_new, grad, H, Hreg, p, F_ch, info,
                       target_KL, moments_work, moments_next, fluxes, next_dir_index_x,
                       grid, moment_powers_constraint, vdf_mb, w_local, moms_MB,
                       mb_tp, Sx_tab, Sy_tab, Sz_tab, Qx_tab, Qy_tab, Qz_tab,
                       wallbc, io_freq, io; BGK_factor=1.0, mu_sref=1.0, sparse_stats_freq=250, threshold=1e-10, verbose=true,
                       flux::Val=Val(:lf), fluxes2=nothing, next2_dir_index_x=nothing,
                       phi=nothing, ind_n=nothing, ind_T=nothing,
                       fplus=nothing, fminus=nothing, mmm_plus=nothing, mmm_minus=nothing,
                       newton_tol=1e-9, fallback_rel_tol=1e2,
                       do_fallback=true)
    n = n_v^3
    m_constraint = n_moments_constraint
    m_all = n_moments_all

    # both are compile-time constants (flux is a Val), so the guarded work below
    # is branch-folded away for the schemes that do not need it
    needs_half = flux === Val(:upwind) || flux === Val(:upwind_lw)
    needs_lw   = flux === Val(:lf_lw)  || flux === Val(:upwind_lw)

    # highest per-axis power in the constraint set (table order for the
    # tensor-product Maxwellian moments)
    max_pow = maximum(p -> max(p[1], p[2], p[3]), moment_powers_constraint)

    # Find indices for the moments we need
    density_index = find_index(moment_powers_constraint, (0,0,0))
    vx_index = find_index(moment_powers_constraint, (1,0,0))
    vy_index = find_index(moment_powers_constraint, (0,1,0))
    vz_index = find_index(moment_powers_constraint, (0,0,1))
    Ex_index = find_index(moment_powers_constraint, (2,0,0))
    Ey_index = find_index(moment_powers_constraint, (0,2,0))
    Ez_index = find_index(moment_powers_constraint, (0,0,2))

    # reusable buffer for the per-cell constraint moments (avoids a per-cell view alloc)
    ref_moms_local = zeros(m_constraint)

    reset_timer!()

    sparsity = 0.0
    @inbounds for step in 1:n_t
        if step % 10 == 0
            println("step = $step / $n_t")
        end

        sparsity = 0.0
        
        for cell in 2:(n_cells+1)
            # Extract local moments for this cell into the reusable buffer
            for i in 1:m_constraint
                ref_moms_local[i] = moments_all[i, cell]
            end
            
            # T_local = 0.0
            # Compute local Maxwell-Boltzmann distribution
            @timeit "MB(T,v)" sol = find_MB_solution_T_and_v!(vdf_mb, grid, ref_moms_local, density_index,
                                            vx_index, vy_index, vz_index,
                                            Ex_index, Ey_index, Ez_index; tol=1e-11, maxiter=50)

            n_target = ref_moms_local[density_index]
            if sol.converged
                mb_ux = sol.ux; mb_uy = sol.uy; mb_uz = sol.uz; mb_T = sol.T
            else
                mb_T = find_MB_solution!(vdf_mb, grid, ref_moms_local, density_index, vx_index, vy_index, vz_index,
                                         Ex_index, Ey_index, Ez_index; tol=1e-11)
                mb_ux = ref_moms_local[vx_index] / n_target
                mb_uy = ref_moms_local[vy_index] / n_target
                mb_uz = ref_moms_local[vz_index] / n_target
            end

            # Get the local weighting function
            unroll!(w_local, vdf_mb.w)

            # Separable axes + per-axis moment tables of the matched Maxwellian
            # (tensor-product paths for moms_MB and rnorm_A)
            maxwell_boltzmann_axes!(mb_tp, grid, n_target, mb_ux, mb_uy, mb_uz, mb_T)
            axis_moment_table!(Sx_tab, grid.vx, grid.Δvx, mb_tp.gx, max_pow)
            axis_moment_table!(Sy_tab, grid.vy, grid.Δvy, mb_tp.gy, max_pow)
            axis_moment_table!(Sz_tab, grid.vz, grid.Δvz, mb_tp.gz, max_pow)

            # Compute moments of the local Maxwellian.
            # moms_MB[j] = C·Sx[a_j]·Sy[b_j]·Sz[c_j] (≡ mmm_constraint * w_local)
            separable_moments!(moms_MB, mb_tp, Sx_tab, Sy_tab, Sz_tab, moment_powers_constraint)

            # Local density and temperature for the collision time. The
            # temperature is the *matched Maxwellian's own* `mb_T`, already
            # solved for above: it is the code temperature in the sense of
            # docs/src/scaling.md, `T = (2/3)(<v²> - |u|²)` with `p = nT`, which
            # is exactly what `tau_BGK` expects. It is also the grid-consistent
            # value — it accounts for the truncated velocity grid, which the
            # analytic moment formula does not.
            n_local = moms_MB[density_index]
            T_local = mb_T

            # Compute collision time and apply BGK relaxation
            tau = tau_BGK(n_local, T_local, omega; mu_sref=mu_sref)
            nu = 1.0 / tau
            # Exact exponential integration of the BGK step on the residual
            # moments: δm ← exp(-ν Δt) δm. A convex combination of the cell
            # moments and moms_MB, hence unconditionally realizable — the
            # explicit-step restriction ν Δt ≤ 1 is gone.
            relax = exp(-dt * nu * BGK_factor)
            for i in 1:m_constraint
                moments_all[i, cell] = moms_MB[i] + relax * (moments_all[i, cell] - moms_MB[i])
            end
            
            # Recompute all arrays that depend on w_local
            compute_constraint_matrix!(A, mmm_constraint, w_local, n, m_constraint)

            # Row norms of A via the squared-weight axis tables: the rows of A
            # are moments of w², itself a separable Gaussian.
            axis_sq_moment_table!(Qx_tab, grid.vx, grid.Δvx, mb_tp.gx, max_pow)
            axis_sq_moment_table!(Qy_tab, grid.vy, grid.Δvy, mb_tp.gy, max_pow)
            axis_sq_moment_table!(Qz_tab, grid.vz, grid.Δvz, mb_tp.gz, max_pow)
            separable_row_norms!(rnorm_A, mb_tp, Qx_tab, Qy_tab, Qz_tab, moment_powers_constraint)

            for j in 1:n
                for i in 1:m_constraint
                    A[i,j] /= rnorm_A[i]
                end
            end
            
            compute_d_w_factors!(d_w, inv_dw, Δv, w_local, n)
            compute_alpha!(alpha, w_local, inv_dw, target_KL, 0.0, n)
            # H (m x m) and uvec (n) are free scratch here; newton_dual! refills both
            @timeit "f(moments): guess" dual_from_primal_guess!(y0, A, alpha, inv_dw, H, uvec, n, m_constraint)
            compute_alpha!(alpha, w_local, inv_dw, target_KL, λ, n)
            
            # Load cell moments into mvec
            for i in 1:m_constraint
                mvec[i] = moments_all[i, cell] / rnorm_A[i]
            end
            
            @timeit "f(moments)" newton_dual!(gsol, A, mvec, alpha, y0, y_new, d_w, inv_dw, uvec,
                         grad, H, Hreg, n, m_constraint, F_ch, p, info;
                         tol=newton_tol, maxiter=200, mu=1e-12, backtrack_rho=0.5, backtrack_c=1e-4)

            if info[:converged] != 1.0 && info[:gradnorm] > fallback_rel_tol * newton_tol
                if do_fallback
                    println("warning: reconstruction not converged (step=$step, cell=$cell, " *
                                "gradnorm=$(info[:gradnorm])); falling back to equilibrium closure")
                    fill!(gsol, 1.0)
                else
                    println("warning: reconstruction not converged (step=$step, cell=$cell, " *
                                "gradnorm=$(info[:gradnorm])), but retaining VDF as-is")
                end
            end


            if step % sparse_stats_freq == 0
                sparsity += sum(abs.(gsol) .< threshold) / n
            end
            
            # compute next-order
            compute_moments_from_vdf!(moments_work, cell, gsol, w_local, mmm_all, n, m_all)

            # half-space (v_x ≷ 0) moments of this cell's own VDF for the kinetic
            # upwind flux: A⁺[f_cell] and A⁻[f_cell]. Costs 2*m_constraint*n per
            # cell, negligible next to the Newton dual solve.
            if needs_half
                compute_moments_from_vdf!(fplus,  cell, gsol, w_local, mmm_plus,  n, m_constraint)
                compute_moments_from_vdf!(fminus, cell, gsol, w_local, mmm_minus, n, m_constraint)
            end

            # capture the boundary-cell VDFs (f = g * w) for the kinetic wall fluxes
            if cell == 2
                for k in 1:n
                    wallbc.f_left[k] = gsol[k] * w_local[k]
                end
            elseif cell == n_cells + 1
                for k in 1:n
                    wallbc.f_right[k] = gsol[k] * w_local[k]
                end
            end
        end

        # --- kinetic diffuse-reflection wall fluxes from the boundary-cell VDFs ---
        @timeit "BCs" kinetic_wall_flux!(wallbc.flux_left, wallbc.fhalf, wallbc.f_left, wallbc.M_left,
                           wallbc.vx, Δv, mmm_constraint, n, m_constraint, +1, wallbc.denom_left)
        @timeit "BCs" kinetic_wall_flux!(wallbc.flux_right, wallbc.fhalf, wallbc.f_right, wallbc.M_right,
                           wallbc.vx, Δv, mmm_constraint, n, m_constraint, -1, wallbc.denom_right)

        # fluxes = next-order x-moments, read from the reconstructed moments_work (interior interfaces)
        @timeit "compute fluxes" populate_fluxes_1d!(fluxes, moments_work, next_dir_index_x, n_cells+2, m_constraint)

        # second-next x-moments (a+2,b,c): the Lax-Wendroff correction term
        if needs_lw
            @timeit "compute fluxes" populate_fluxes_1d!(fluxes2, moments_work, next2_dir_index_x, n_cells+2, m_constraint)
            @timeit "limiter" shared_limiter!(phi, ind_n, ind_T, moments_all, density_index,
                                    vx_index, vy_index, vz_index,
                                    Ex_index, Ey_index, Ez_index, n_cells)
        end

        # transport: LF/upwind, optionally limited-LW corrected, in the interior;
        # kinetic fluxes at the two walls
        if flux === Val(:lf)
            @timeit "L-F" lax_friedrichs_wall_bc!(moments_next, moments_all, fluxes,
                                    wallbc.flux_left, wallbc.flux_right,
                                    dt, dx, max_eigenvalue, n_cells, m_constraint)
        elseif flux === Val(:upwind)
            @timeit "upwind" kinetic_upwind_wall_bc!(moments_next, moments_all, fplus, fminus,
                                    wallbc.flux_left, wallbc.flux_right,
                                    dt, dx, n_cells, m_constraint)
        elseif flux === Val(:lf_lw)
            @timeit "L-W" lax_wendroff_wall_bc!(moments_next, moments_all, fluxes, fluxes2,
                                    wallbc.flux_left, wallbc.flux_right, phi,
                                    dt, dx, max_eigenvalue, n_cells, m_constraint)
        elseif flux === Val(:upwind_lw)
            @timeit "L-W" upwind_lax_wendroff_wall_bc!(moments_next, moments_all, fluxes, fluxes2,
                                    fplus, fminus, wallbc.flux_left, wallbc.flux_right, phi,
                                    dt, dx, n_cells, m_constraint)
        else
            throw(ArgumentError("convect_1D_LF!: unknown flux $flux"))
        end

        # copy interior back into the state
        for m in 1:m_constraint
            for i in 2:(n_cells+1)
                moments_all[m, i] = moments_next[m, i]
            end
        end

        # time-evolution snapshot. phi is written as the limiter state that
        # produced this step: phi[i] is the interface between cells i and i+1, so
        # column j of the output is the interface to the right of interior cell j.
        if !isnothing(io) && (step % io_freq == 0)
            write_moment_snapshot!(io, step, step * dt, @view moments_all[:, 2:(n_cells+1)];
                                   phi_interior = needs_lw ? (@view phi[2:(n_cells+1)]) : nothing)
        end

        if step % sparse_stats_freq == 0
            println("Sparsity (%) = $(100 * sparsity / n_cells)")
        end
    end

    if verbose
        print_timer()
    end
    return 100 * sparsity / n_cells
end

"""
    maxwellian_constraint_moments(grid, mmm_constraint, n_dens, ux, uy, uz, T)

Constraint-moment vector of a Maxwellian with density `n_dens`, bulk velocity
`(ux,uy,uz)` and temperature `T`, sampled on `grid`.
"""
function maxwellian_constraint_moments(grid, mmm_constraint, n_dens, ux, uy, uz, T)
    vdf_tmp = VDF3D(grid)
    maxwell_boltzmann!(vdf_tmp, grid, n_dens, ux, uy, uz, T)
    f = unroll(vdf_tmp.w)
    return mmm_constraint * f
end

"""
    run_1d(; kwargs...)

Driver for the 1D Lax–Friedrichs moment solver. Builds the velocity grid,
moment power sets, measurement matrices, the cross-set flux index
`next_dir_index_x`, all Newton/reconstruction work arrays, and the spatial
state arrays (sized with one ghost cell on each side), sets a two-state
shock-tube initial condition, then advances `convect_1D_LF!`.

# Important
* `moment_powers_all` is built as `[constraint; extras]` so its first
  `m_constraint` entries are exactly `moment_powers_constraint` (the closure
  contract `mmm_all[1:m_constraint,:] == mmm_constraint`). Asserted below.
* `next_dir_index_x[m] = find_index(moment_powers_all, (a+1,b,c))` for each
  constraint power `(a,b,c)`. With `all = up_to(M+1)` over `constraint = up_to(M)`
  every entry is `> 0`.
* Boundary conditions are diffuse-reflection solid walls: at each wall interface
  the flux is a kinetic upwind flux built from the in-cell reconstructed VDF and
  the wall Maxwellian (`kinetic_wall_flux!`, Baranger et al. 2019). This enforces
  zero mass flux through the walls. The BGK term inside `convect_1D_LF_mm!` is
  the exact exponential relaxation of the residual moments (unconditional in
  `Δt`, no `ν Δt ≤ 1` restriction).

# Keyword arguments
* `max_moment_constraint::Int`: highest total order of evolved (constraint) moments.
* `n_v::Int`, `extent`: velocity grid points per axis and half-extent `[-extent, extent]`.
* `n_cells::Int`, `x_min`, `x_max`: spatial domain (interior cells).
* `lambda_unscaled`: L1 weight, scaled internally by `sum(Δv)/length(Δv)` (matches the reference scripts).
* `omega`: VHS viscosity-temperature exponent passed to `tau_BGK` (1/2 = hard
  sphere, 1 = Maxwell molecules).
* `mu_sref`: reference viscosity (viscosity at `T=1`) setting the BGK collision
  scale / Knudsen number; passed to `tau_BGK`. Larger `mu_sref` = more rarefied.
* `CFL`, `t_end`: time-step factor and end time; `dt = CFL*dx/max_eigenvalue`, `max_eigenvalue = extent`.
* `left_state`, `right_state`: NamedTuples `(n, ux, T)` for the two initial
  states. The tangential velocities `vy`, `vz` are optional extra fields and
  default to zero, so a plain `(n, ux, T)` state behaves exactly as before.
* `interface_x`: x of the initial discontinuity (defaults to domain midpoint).
  Ignored when `interpolate_init_solution = true` — there is no discontinuity.
* `interpolate_init_solution::Bool=false`: initial condition layout.
  * `false` (default) — the two-state discontinuity at `interface_x`: every cell
    left of it gets `left_state`, every cell right of it `right_state`.
  * `true` — linearly ramp the primitive variables `(n, ux, vy, vz, T)` from
    `left_state` at `x_min` to `right_state` at `x_max`, sampled at cell centres.

  Each cell is still initialized to an *exact Maxwellian* at its interpolated
  primitives, so the gas starts at local equilibrium everywhere. (Interpolating
  the moment vectors instead would be realizable — the realizable set is convex
  — but would put a bimodal mixture of two Maxwellians in every cell, a
  non-equilibrium state the closure then has to represent.)

  The endpoints are the domain *edges*, not the first and last cell centres, so
  the ramp lines up with the wall interfaces rather than being offset by
  `dx/2`.
* `wall_left`, `wall_right`: NamedTuples `(T, vy, vz)` for the diffuse-reflection
  walls (temperature and tangential velocity; normal velocity is zero).
* `flux::Symbol`: interior flux — a product of two independent choices, the
  first-order baseline (Lax–Friedrichs or kinetic upwind) and the optional
  limited Lax–Wendroff correction on top of it:

  | symbol       | low-order                        | correction     | `moment_powers_all` |
  |--------------|----------------------------------|----------------|---------------------|
  | `:lf`        | Lax–Friedrichs, `a = extent`     | none           | `up_to(M+1)`        |
  | `:lf_lw`     | Lax–Friedrichs, `a = extent`     | LW, shared phi | `up_to(M+2)`        |
  | `:upwind`    | kinetic upwind, `a_k = abs(v_k)` | none           | `up_to(M+1)`        |
  | `:upwind_lw` | kinetic upwind, `a_k = abs(v_k)` | LW, shared phi | `up_to(M+2)`        |

  * `:lf` (default) — `lax_friedrichs_wall_bc!`. Bitwise identical to the
    behaviour before this switch was added.
  * `:upwind` — `kinetic_upwind_wall_bc!`. Dissipates per velocity node rather
    than at the global maximum wave speed (~5x less numerical viscosity, see
    `run_1d_dvm`'s `dissipation=:upwind`), and makes the interior interfaces the
    same operator as the kinetic wall fluxes. Needs the two half-space moments
    `A±[f_cell]` of each cell's own VDF — 2 extra `m_constraint x n` reductions
    per cell, no extra dual solve, no interface reconstruction.
  * `:lf_lw` / `:upwind_lw` — flux-limited Lax–Wendroff on the corresponding
    baseline (`lax_wendroff_wall_bc!` / `upwind_lax_wendroff_wall_bc!`), second
    order in smooth regions, dropping to the baseline at extrema via a single
    shared scalar limiter per interface (`shared_limiter!`). The high-order flux
    is the same for both, so `phi = 1` gives the same pure Lax–Wendroff flux; the
    baseline only sets how large the limited increment is. Both require
    `moment_powers_all = up_to(M+2)` for the `(a+2,b,c)` correction moments, so
    `m_all` grows (M=4: 56 → 84) and the per-cell `compute_moments_from_vdf!`
    reduction costs proportionally more. The dual solve per cell is unchanged —
    the correction is built from each cell's own reconstruction, so it adds no
    Newton work and never needs a neighbouring cell's VDF.

  All four use the same `dt = CFL*dx/extent` (upwinding buys less diffusion, not
  a larger step) and the same kinetic wall fluxes, and all four conserve mass
  exactly (the limiter blends per interface, so interior fluxes still telescope).

Returns a NamedTuple with the final `moments_state`, the last-step `moments_work`,
spatial/grid metadata, the power sets and the flux used.
"""
function run_1d(; max_moment_constraint::Int=4,
                  n_v::Int=20, extent=4.0,
                  n_cells::Int=100, x_min=0.0, x_max=1.0,
                  lambda_unscaled=0.0, omega=0.5, mu_sref=1.0,
                  CFL=0.4, t_end=0.2,
                  left_state=(n=1.0, ux=0.0, T=1.0),
                  right_state=(n=1.0, ux=0.0, T=0.5),
                  interface_x=nothing,
                  wall_left=(T=1.0, vy=0.0, vz=0.0),
                  wall_right=(T=1.0, vy=0.0, vz=0.0),
                  target_KL=nothing,
                  io_path=nothing, io_freq::Int=1,
                  verbose=true, BGK_factor=1.0,
                  flux::Symbol=:lf,
                  interpolate_init_solution::Bool=false,
                  newton_tol=1e-9, fallback_rel_tol=1e2,
                  do_fallback=true)

    flux in (:lf, :lf_lw, :upwind, :upwind_lw) ||
        throw(ArgumentError("run_1d: flux must be one of :lf, :lf_lw, :upwind, :upwind_lw, got :$flux"))

    # the two axes of the flux: low-order baseline x second-order correction
    needs_half = flux in (:upwind, :upwind_lw)   # fplus/fminus, mmm_plus/mmm_minus
    needs_lw   = flux in (:lf_lw, :upwind_lw)    # fluxes2, next2_dir_index_x, phi, ind_*

    dx = (x_max - x_min) / n_cells

    # --- time step from CFL; max wave speed is the max |v_x| on the grid (= extent) ---
    # This is the same for all four fluxes, including the two upwind ones: the
    # fastest velocity node still has to satisfy the CFL condition, so upwinding
    # buys less numerical diffusion, not a larger time step. `max_eigenvalue` is
    # unused *in the flux* under :upwind/:upwind_lw but is still needed here.
    max_eigenvalue = extent
    dt = CFL * dx / max_eigenvalue
    n_t = max(1, ceil(Int, t_end / dt))

    domain_length = x_max - x_min
    Kn_l = Knudsen_number(left_state.n, left_state.T, omega, domain_length, mu_sref)
    Kn_r = Knudsen_number(right_state.n, right_state.T, omega, domain_length, mu_sref)
    println("Knudsen (left): $Kn_l | Knudsen (right): $Kn_r")
    tau_l = tau_BGK(left_state.n, left_state.T, omega; mu_sref=mu_sref)
    tau_r = tau_BGK(right_state.n, right_state.T, omega; mu_sref=mu_sref)
    println("tau_coll (left): $tau_l | tau_coll (right): $tau_r")
    println("dt: $dt | n_t: $n_t")

    # --- moment power sets: constraint first, then the extra higher-order powers ---
    # The first-order schemes need the (a+1,b,c) flux moments, so all = up_to(M+1).
    # The Lax-Wendroff variants additionally need the (a+2,b,c) correction moments,
    # so all = up_to(M+2). The extra order is only paid for when it is used — under
    # :lf the set, and hence every result, is exactly what it was before the switch
    # existed.
    extra_order = needs_lw ? 2 : 1
    moment_powers_constraint = all_powers_up_to_M_3D(max_moment_constraint)
    moment_powers_all = vcat(moment_powers_constraint,
                             [p for p in all_powers_up_to_M_3D(max_moment_constraint + extra_order)
                              if !(p in moment_powers_constraint)])
    m_constraint = length(moment_powers_constraint)
    m_all = length(moment_powers_all)
    @assert moment_powers_all[1:m_constraint] == moment_powers_constraint "moment_powers_all must begin with moment_powers_constraint"

    # --- velocity grid + quadrature weights ---
    grid = Grid3D([-extent, extent], [-extent, extent], [-extent, extent], n_v, n_v, n_v)
    Δv = unroll(grid_weights(grid))
    n = n_v^3

    # --- moment measurement matrices ---
    mmm_constraint = construct_moment_measurement_matrix_3D(grid, n_v, moment_powers_constraint)
    mmm_all = construct_moment_measurement_matrix_3D(grid, n_v, moment_powers_all)
    @assert mmm_all[1:m_constraint, :] ≈ mmm_constraint "mmm_all constraint block must match mmm_constraint"

    # --- flat per-node x-velocity (kinetic wall BCs + the masked matrices below) ---
    vx_flat = unroll([grid.vx[i] for i in 1:n_v, j in 1:n_v, l in 1:n_v])

    # --- masked measurement matrices for the kinetic upwind flux ---
    # A±_j[f] = Σ_{±v_x>0} m_j(v) v_x f Δv; mmm_constraint already carries Δv, so
    # only the extra v_x factor and the half-space mask are applied here. Strict
    # >/< is deliberate: a v_x = 0 node (present only for odd n_v) contributes
    # nothing to either half, and nothing to the (a+1,b,c) flux moment either,
    # so fplus + fminus == fluxes still holds (up to summation order).
    mmm_plus = needs_half ?
        [vx_flat[k] > 0 ? mmm_constraint[j, k] * vx_flat[k] : 0.0
         for j in 1:m_constraint, k in 1:n] : zeros((0, 0))
    mmm_minus = needs_half ?
        [vx_flat[k] < 0 ? mmm_constraint[j, k] * vx_flat[k] : 0.0
         for j in 1:m_constraint, k in 1:n] : zeros((0, 0))

    # --- cross-set flux index: flux of (a,b,c) is the next x-moment (a+1,b,c) in moment_powers_all ---
    next_dir_index_x = [find_index(moment_powers_all, (p[1] + 1, p[2], p[3]))
                        for p in moment_powers_constraint]
    @assert all(>(0), next_dir_index_x) "every next x-moment must exist in moment_powers_all"

    # --- second cross-set index (Lax-Wendroff only): the (a+2,b,c) moment ---
    next2_dir_index_x = needs_lw ?
        [find_index(moment_powers_all, (p[1] + 2, p[2], p[3])) for p in moment_powers_constraint] :
        Int[]
    @assert all(>(0), next2_dir_index_x) "every second-next x-moment must exist in moment_powers_all"

    # --- L1 weight scaling (matches reference scripts) ---
    λ = lambda_unscaled * sum(Δv) / length(Δv)

    target_KL = isnothing(target_KL) ? ones(n) : target_KL

    # --- Newton / reconstruction work arrays (mirror solve_iterate_over_L1_values) ---
    A = ones((m_constraint, n))
    rnorm_A = ones(m_constraint)
    alpha = zeros(n)
    gsol = zeros(n)
    uvec = zeros(n)
    d_w = zeros(n)
    inv_dw = zeros(n)
    y0 = zeros(m_constraint)
    y_new = zeros(m_constraint)
    mvec = zeros(m_constraint)
    grad = zeros(m_constraint)
    H = zeros((m_constraint, m_constraint))
    Hreg = zeros((m_constraint, m_constraint))
    p = zeros(m_constraint)
    F_ut = UpperTriangular(H)
    F_ch = Cholesky(F_ut)
    info::Dict{Symbol, Float64} = Dict(:iterations => 0.0, :converged => 0.0, :gradnorm => NaN)
    vdf_mb = VDF3D(grid)
    w_local = zeros(n)
    moms_MB = zeros(m_constraint)

    # --- tensor-product Maxwellian scratch: separable axes + per-axis tables ---
    mb_tp = MaxwellianTP(grid)
    Sx_tab = zeros(max_moment_constraint + 2)   # orders 0..M+1
    Sy_tab = zeros(max_moment_constraint + 2)
    Sz_tab = zeros(max_moment_constraint + 2)
    Qx_tab = zeros(max_moment_constraint + 2)
    Qy_tab = zeros(max_moment_constraint + 2)
    Qz_tab = zeros(max_moment_constraint + 2)

    # --- spatial arrays, sized with ghosts: columns 1 and n_cells+2 are ghosts ---
    moments_all  = zeros((m_constraint, n_cells + 2))   # evolved state
    moments_work = zeros((m_all,        n_cells + 2))   # reconstructed (flux source)
    fluxes       = zeros((m_constraint, n_cells + 2))
    moments_next = zeros((m_constraint, n_cells + 2))

    # --- Lax-Wendroff correction + limiter scratch (empty for the first-order fluxes) ---
    fluxes2 = needs_lw ? zeros((m_constraint, n_cells + 2)) : zeros((0, 0))
    phi     = needs_lw ? zeros(n_cells + 2) : zeros(0)   # phi[i] = interface (i, i+1)
    ind_n   = needs_lw ? zeros(n_cells + 2) : zeros(0)
    ind_T   = needs_lw ? zeros(n_cells + 2) : zeros(0)

    # --- kinetic upwind half-moments (empty for the Lax-Friedrichs baselines) ---
    fplus  = needs_half ? zeros((m_constraint, n_cells + 2)) : zeros((0, 0))
    fminus = needs_half ? zeros((m_constraint, n_cells + 2)) : zeros((0, 0))

    # --- kinetic wall BC data: density-1 wall Maxwellians (vx_flat built above) ---
    vdf_wall = VDF3D(grid)
    maxwell_boltzmann!(vdf_wall, grid, 1.0, 0.0, wall_left.vy, wall_left.vz, wall_left.T)
    E_wall = compute_E(vdf_wall, grid, 1.0)
    sol = find_T_and_v!(vdf_wall, grid, 0.0, wall_left.vy, wall_left.vz, E_wall,
                        0.0, wall_left.vy, wall_left.vz, (2.0/3.0) * E_wall; tol=1e-15, maxiter=50)
    
    if verbose
        println("Left wall sol: $(sol)")
    end
    # write the final distribution scaled to the target density
    maxwell_boltzmann!(vdf_wall, grid, 1.0, sol.ux, sol.uy, sol.uz, sol.T)

    # maxwell_boltzmann!(vdf_wall, grid, 1.0, 0.0, wall_left.vy,  wall_left.vz,  wall_left.T)
    M_wall_left = unroll(vdf_wall.w)

    
    maxwell_boltzmann!(vdf_wall, grid, 1.0, 0.0, wall_right.vy, wall_right.vz, wall_right.T)
    E_wall = compute_E(vdf_wall, grid, 1.0)
    sol = find_T_and_v!(vdf_wall, grid, 0.0, wall_right.vy, wall_right.vz, E_wall,
                        0.0, wall_right.vy, wall_right.vz, (2.0/3.0) * E_wall; tol=1e-15, maxiter=50)
    
    if verbose
        println("Right wall sol: $(sol)")
    end
    # write the final distribution scaled to the target density
    maxwell_boltzmann!(vdf_wall, grid, 1.0, sol.ux, sol.uy, sol.uz, sol.T)
    M_wall_right = unroll(vdf_wall.w)
    wallbc = (vx = vx_flat, M_left = M_wall_left, M_right = M_wall_right,
              f_left = zeros(n), f_right = zeros(n), fhalf = zeros(n),
              flux_left = zeros(m_constraint), flux_right = zeros(m_constraint),
              denom_left = wall_flux_denominator(M_wall_left, vx_flat, Δv, n, +1),
              denom_right = wall_flux_denominator(M_wall_right, vx_flat, Δv, n, -1))

    x = [x_min + (i - 0.5) * dx for i in 1:n_cells]     # interior cell centers

    # --- initial condition over interior cells 2..n_cells+1 ---
    # Tangential velocities are optional fields on the state NamedTuples, so a
    # plain (n, ux, T) state keeps behaving exactly as before.
    state_u(s) = (get(s, :ux, 0.0), get(s, :vy, 0.0), get(s, :vz, 0.0))
    lu, ru = state_u(left_state), state_u(right_state)

    if interpolate_init_solution
        # Linear ramp in the primitive variables from left_state at x_min to
        # right_state at x_max, sampled at cell centres. Each cell still gets an
        # exact Maxwellian, so the gas starts at local equilibrium everywhere —
        # unlike interpolating the moment vectors, which would give a bimodal
        # mixture of two Maxwellians that the closure then has to represent.
        span = x_max - x_min
        for (ci, xc) in enumerate(x)
            θ  = span > 0 ? (xc - x_min) / span : 0.0
            nd = (1 - θ) * left_state.n + θ * right_state.n
            ux = (1 - θ) * lu[1] + θ * ru[1]
            uy = (1 - θ) * lu[2] + θ * ru[2]
            uz = (1 - θ) * lu[3] + θ * ru[3]
            Tc = (1 - θ) * left_state.T + θ * right_state.T
            @views moments_all[:, ci+1] .=
                maxwellian_constraint_moments(grid, mmm_constraint, nd, ux, uy, uz, Tc)
        end
    else
        # two-state shock-tube discontinuity at `interface_x`
        interface = isnothing(interface_x) ? 0.5 * (x_min + x_max) : interface_x
        left_moms  = maxwellian_constraint_moments(grid, mmm_constraint, left_state.n,  lu[1], lu[2], lu[3], left_state.T)
        right_moms = maxwellian_constraint_moments(grid, mmm_constraint, right_state.n, ru[1], ru[2], ru[3], right_state.T)
        for (ci, xc) in enumerate(x)
            col = ci + 1
            @views moments_all[:, col] .= xc < interface ? left_moms : right_moms
        end
    end
    # ghost state columns are unused (wall fluxes replace them); left at zero.


    if verbose
        println("run_1d: M=$max_moment_constraint, m_constraint=$m_constraint, m_all=$m_all")
        println("  grid $(n_v)^3, extent [-$extent,$extent]; spatial n_cells=$n_cells, dx=$dx")
        println("  λ(scaled)=$λ, omega=$omega, CFL=$CFL, dt=$dt, n_t=$n_t, t_end≈$(n_t*dt)")
        println("  flux=:$flux, mu_sref=$mu_sref")
    end

    # --- open time-evolution output (streaming HDF5); write the t=0 frame ---
    io = nothing
    if !isnothing(io_path)
        io = init_moment_io(io_path, x, moment_powers_constraint, m_constraint, n_cells;
                            dt=dt, dx=dx, omega=omega, lambda=λ, io_freq=io_freq,
                            max_moment_constraint=max_moment_constraint, extent=extent,
                            x_min=x_min, x_max=x_max, n_cells=n_cells,
                            mu_sref=mu_sref, flux=string(flux),
                            interpolate_init_solution=string(interpolate_init_solution))
        write_moment_snapshot!(io, 0, 0.0, @view moments_all[:, 2:(n_cells+1)])
    end

    sparsity = 0.0
    try
        sparsity = convect_1D_LF!(moments_all, dt, dx, max_eigenvalue, n_t,
                       n_cells, m_all, m_constraint,
                       n_v, mmm_constraint, mmm_all, Δv, λ, omega,
                       A, rnorm_A, mvec, alpha, gsol, uvec, d_w, inv_dw,
                       y0, y_new, grad, H, Hreg, p, F_ch, info,
                       target_KL, moments_work, moments_next, fluxes, next_dir_index_x,
                       grid, moment_powers_constraint, vdf_mb, w_local, moms_MB,
                       mb_tp, Sx_tab, Sy_tab, Sz_tab, Qx_tab, Qy_tab, Qz_tab,
                       wallbc, io_freq, io; BGK_factor=BGK_factor, mu_sref=mu_sref, verbose=verbose,
                       flux=Val(flux), fluxes2=fluxes2, next2_dir_index_x=next2_dir_index_x,
                       phi=phi, ind_n=ind_n, ind_T=ind_T,
                       fplus=fplus, fminus=fminus, mmm_plus=mmm_plus, mmm_minus=mmm_minus,
                       newton_tol=newton_tol, fallback_rel_tol=fallback_rel_tol,
                       do_fallback=do_fallback)
    finally
        isnothing(io) || close_moment_io(io)
    end

    return (moments_state = moments_all,
            moments_work = moments_work,
            fplus = fplus, fminus = fminus,
            x = x, dx = dx, dt = dt, n_t = n_t,
            moment_powers_constraint = moment_powers_constraint,
            moment_powers_all = moment_powers_all,
            next_dir_index_x = next_dir_index_x,
            next2_dir_index_x = next2_dir_index_x,
            grid = grid,
            flux = flux, sparsity=sparsity,
            interpolate_init_solution = interpolate_init_solution)
end