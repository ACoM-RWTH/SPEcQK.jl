"""
    lax_friedrichs_dvm_wall_bc!(f_next, f_current, vx, fhalf_left, fhalf_right,
                                dt, dx, max_eigenvalue, n_cells, n,
                                dissipation::Val=Val(:global))

First-order transport of the per-node VDF `f` (size `(n, n_cells+2)`, interior
cells in columns `2..n_cells+1`). The two wall interfaces use the externally
supplied per-node kinetic upwind fluxes `fhalf_left`/`fhalf_right` (the `fhalf`
output of `kinetic_wall_flux!`). Ghost columns are never read.

Interior interfaces use a Rusanov flux
`0.5*v_k*(f_L+f_R) - 0.5*a_k*(f_R-f_L)` whose dissipation coefficient `a_k` is
selected by `dissipation`:

* `Val(:global)` — `a_k = max_eigenvalue` for every node, i.e. the same global
  dissipation as the moment solver, so the two transports are directly
  comparable. This is the default.
* `Val(:upwind)` — `a_k = |v_k|`, which reduces the Rusanov flux to the *exact*
  upwind flux for that node (each node is pure linear advection at speed `v_k`,
  so no approximate Riemann solver is needed). Written here directly in upwind
  form. The node-wise numerical diffusivity drops from
  `(max_eigenvalue*dx/2)(1 - CFL*v_k^2/max_eigenvalue^2)` to
  `(|v_k|*dx/2)(1 - CFL*|v_k|/max_eigenvalue)`

Both variants are conservative (flux form), and both are positivity-preserving
under the same `CFL <= 1` restriction already imposed by
`dt = CFL*dx/max_eigenvalue`, so the time step is unaffected by this choice.
"""
function lax_friedrichs_dvm_wall_bc!(f_next, f_current, vx, fhalf_left, fhalf_right,
                                     dt, dx, max_eigenvalue, n_cells, n,
                                     dissipation::Val=Val(:global))
    inv_dx = 1.0 / dx

    @inbounds for i in 2:(n_cells+1)
        for k in 1:n
            u_C = f_current[k, i]
            vk = vx[k]

            # left interface flux (i - 1/2)
            if i == 2
                flux_left = fhalf_left[k]                           # kinetic wall flux
            else
                u_L = f_current[k, i-1]
                flux_left = dvm_interface_flux(vk, u_L, u_C, max_eigenvalue, dissipation)
            end

            # right interface flux (i + 1/2)
            if i == n_cells + 1
                flux_right = fhalf_right[k]                         # kinetic wall flux
            else
                u_R = f_current[k, i+1]
                flux_right = dvm_interface_flux(vk, u_C, u_R, max_eigenvalue, dissipation)
            end

            f_next[k, i] = u_C - dt * inv_dx * (flux_right - flux_left)
        end
    end

    return nothing
end

"""
    dvm_interface_flux(vk, u_L, u_R, max_eigenvalue, dissipation::Val)

Interior interface flux for a single velocity node `k` of the DVM transport.
Dispatches on `dissipation` (see `lax_friedrichs_dvm_wall_bc!`) so the branch is
resolved at compile time and the inner loop stays free of it.
"""
@inline dvm_interface_flux(vk, u_L, u_R, max_eigenvalue, ::Val{:global}) =
    0.5 * vk * (u_L + u_R) - 0.5 * max_eigenvalue * (u_R - u_L)

# a_k = |v_k| collapses the Rusanov flux to exact upwinding: v>0 -> v*u_L, v<0 -> v*u_R
@inline dvm_interface_flux(vk, u_L, u_R, _max_eigenvalue, ::Val{:upwind}) =
    vk * (vk >= 0.0 ? u_L : u_R)

"""
    dvm_raw_moments!(ref_moms, f, col, vx, vy, vz, Δv, n)

Fills `ref_moms` with the 7 raw moments of the VDF in column `col` of `f`:
`[n, n*ux, n*uy, n*uz, <vx²>, <vy²>, <vz²>]` (layout matching the index
arguments `1..7` of `find_MB_solution_T_and_v!`).
"""
function dvm_raw_moments!(ref_moms, f, col, vx, vy, vz, Δv, n)
    fill!(ref_moms, 0.0)
    @inbounds for k in 1:n
        fk = f[k, col] * Δv[k]
        vxk = vx[k]; vyk = vy[k]; vzk = vz[k]
        ref_moms[1] += fk
        ref_moms[2] += vxk * fk
        ref_moms[3] += vyk * fk
        ref_moms[4] += vzk * fk
        ref_moms[5] += vxk * vxk * fk
        ref_moms[6] += vyk * vyk * fk
        ref_moms[7] += vzk * vzk * fk
    end
    return nothing
end

function convect_1D_DVM!(f, f_next, dt, dx, max_eigenvalue, n_t,
                         n_cells, n_v, mmm_constraint, m_constraint, Δv, omega,
                         grid, vdf_mb, w_local, ref_moms, mb_params,
                         vy, vz, wallbc, io_freq, io, moments_out, T_mb_out;
                         BGK_factor=1.0, mu_sref=1.0, verbose=true,
                         dissipation::Val=Val(:global))
    n = n_v^3
    vx = wallbc.vx

    reset_timer!()

    @inbounds for step in 1:n_t
        if step % 10 == 0
            println("step = $step / $n_t")
        end

        # --- BGK relaxation toward the grid-matched discrete Maxwellian ---
        for cell in 2:(n_cells+1)
            dvm_raw_moments!(ref_moms, f, cell, vx, vy, vz, Δv, n)

            # discrete Maxwellian matching density, velocity AND energy on the
            # cut-off grid — the relaxation below then conserves all five
            # collision invariants exactly at the discrete level
            @timeit "MB(T,v)" sol = find_MB_solution_T_and_v!(vdf_mb, grid, ref_moms,
                                            1, 2, 3, 4, 5, 6, 7; tol=1e-11, maxiter=50)

            if sol.converged
                mb_params[1, cell] = sol.ux
                mb_params[2, cell] = sol.uy
                mb_params[3, cell] = sol.uz
                mb_params[4, cell] = sol.T
            else
                T_fb = find_MB_solution!(vdf_mb, grid, ref_moms, 1, 2, 3, 4, 5, 6, 7; tol=1e-11)
                n_loc = ref_moms[1]
                mb_params[1, cell] = ref_moms[2] / n_loc
                mb_params[2, cell] = ref_moms[3] / n_loc
                mb_params[3, cell] = ref_moms[4] / n_loc
                mb_params[4, cell] = T_fb
            end

            unroll!(w_local, vdf_mb.w)

            # Local density and temperature for the collision time. As in the
            # moment solver, the temperature is the matched Maxwellian's own,
            # just stored into mb_params above: it is the code temperature of
            # docs/src/scaling.md (`T = (2/3)(<v²> - |u|²)`, `p = nT`), which is
            # what `tau_BGK` expects, and it is grid-consistent on the truncated
            # velocity grid.
            n_local = ref_moms[1]
            T_local = mb_params[4, cell]

            tau = tau_BGK(n_local, T_local, omega; mu_sref=mu_sref)
            nu = 1.0 / tau
            # Exact exponential integration of the BGK step: f ← M + e^{-νΔt}(f-M).
            # A convex combination of f and the matched Maxwellian, hence
            # positivity-preserving whenever f ≥ 0 — no ν Δt ≤ 1 restriction.
            relax = exp(-dt * nu * BGK_factor)
            for k in 1:n
                f[k, cell] = w_local[k] + relax * (f[k, cell] - w_local[k])
            end
        end

        # --- kinetic diffuse-reflection wall fluxes from the boundary-cell VDFs ---
        @timeit "BCs" kinetic_wall_flux!(wallbc.flux_left, wallbc.fhalf_left,
                           (@view f[:, 2]), wallbc.M_left,
                           wallbc.vx, Δv, mmm_constraint, n, m_constraint, +1, wallbc.denom_left)
        @timeit "BCs" kinetic_wall_flux!(wallbc.flux_right, wallbc.fhalf_right,
                           (@view f[:, n_cells+1]), wallbc.M_right,
                           wallbc.vx, Δv, mmm_constraint, n, m_constraint, -1, wallbc.denom_right)

        # transport: Rusanov/upwind in the interior, kinetic per-node fluxes at the two walls
        @timeit "L-F" lax_friedrichs_dvm_wall_bc!(f_next, f, wallbc.vx,
                                wallbc.fhalf_left, wallbc.fhalf_right,
                                dt, dx, max_eigenvalue, n_cells, n, dissipation)

        # copy interior back into the state
        for i in 2:(n_cells+1)
            for k in 1:n
                f[k, i] = f_next[k, i]
            end
        end

        # time-evolution snapshot: constraint moments of f + matched-Maxwellian T
        if !isnothing(io) && (step % io_freq == 0)
            for cell in 2:(n_cells+1)
                ci = cell - 1
                for j in 1:m_constraint
                    s = 0.0
                    for k in 1:n
                        s += mmm_constraint[j, k] * f[k, cell]
                    end
                    moments_out[j, ci] = s
                end

                # matched-Maxwellian temperature of the post-transport state,
                # warm-started from this step's relaxation-pass parameters
                dvm_raw_moments!(ref_moms, f, cell, vx, vy, vz, Δv, n)
                n_loc = ref_moms[1]
                ux_t = ref_moms[2] / n_loc
                uy_t = ref_moms[3] / n_loc
                uz_t = ref_moms[4] / n_loc
                E2_t = (ref_moms[5] + ref_moms[6] + ref_moms[7]) / n_loc
                T0 = mb_params[4, cell]
                if !(T0 > 0.0)
                    T0 = (2.0 / 3.0) * (E2_t - (ux_t^2 + uy_t^2 + uz_t^2))
                end
                sol_T = find_T_and_v!(vdf_mb, grid, ux_t, uy_t, uz_t, E2_t,
                                      mb_params[1, cell], mb_params[2, cell],
                                      mb_params[3, cell], T0;
                                      tol=1e-11, maxiter=50)
                T_mb_out[ci] = sol_T.T
            end
            write_moment_snapshot!(io, step, step * dt, moments_out, T_mb_out)
        end
    end

    if verbose
        print_timer()
    end
    return nothing
end

"""
    run_1d_dvm(; kwargs...)

Driver for the pure DVM (discrete velocity method) 1D Lax–Friedrichs solver.
Evolves the full per-node VDF `f(v_k, x_i)` directly — no maximum-entropy
reconstruction, no moment closure. Transport is node-wise Lax–Friedrichs with
the same global dissipation `max_eigenvalue = extent` as the moment solvers
(`dissipation=:global`, the default) or node-wise exact upwinding
(`dissipation=:upwind`, ~5x less numerical viscosity — see below);
the BGK collision step relaxes `f` exactly (`f ← M + e^{-νΔt}(f - M)`) toward
the grid-matched discrete Maxwellian (`find_MB_solution_T_and_v!`), so the
five collision invariants are conserved exactly at the discrete level.
Serves as the kinetic reference solution for `run_1d` / `run_1d_mm`.

# Important
* Boundary conditions are diffuse-reflection solid walls, fully reusing
  `kinetic_wall_flux!`: its per-node upwind flux `fhalf` is used directly as
  the wall-interface flux for every velocity node (the moment solvers use its
  projection onto the constraint moments).
* `max_moment_constraint` only selects which moments are written to the output
  file / returned — it does not affect the dynamics.

# Keyword arguments
* `max_moment_constraint::Int`: highest total order of output moments.
* `n_v::Int`, `extent`: velocity grid points per axis and half-extent `[-extent, extent]`.
* `n_cells::Int`, `x_min`, `x_max`: spatial domain (interior cells).
* `omega`: VHS viscosity-temperature exponent passed to `tau_BGK` (1/2 = hard
  sphere, 1 = Maxwell molecules).
* `mu_sref`: reference viscosity (viscosity at `T=1`) setting the BGK collision
  scale / Knudsen number; passed to `tau_BGK`. Larger `mu_sref` = more rarefied.
* `CFL`, `t_end`: time-step factor and end time; `dt = CFL*dx/max_eigenvalue`, `max_eigenvalue = extent`.
* `dissipation::Symbol`: interior flux dissipation, `:global` (default) or
  `:upwind`. `:global` uses `a_k = extent` for every velocity node, matching the
  moment solvers' dissipation exactly. `:upwind` uses `a_k = |v_k|`, the exact
  upwind flux for each node's linear advection; it cuts the velocity-averaged
  numerical diffusivity from `≈0.49*extent*dx` to `≈0.37*dx`.
* `left_state`, `right_state`: NamedTuples `(n, ux, T)` for the two initial
  states. The tangential velocities `vy`, `vz` are optional extra fields and
  default to zero, so a plain `(n, ux, T)` state behaves exactly as before.
* `interface_x`: x of the initial discontinuity (defaults to domain midpoint).
  Ignored when `interpolate_init_solution = true` — there is no discontinuity.
* `interpolate_init_solution::Bool=false`: initial condition layout, identical in
  meaning and implementation to [`run_1d`](@ref)'s keyword of the same name, so
  the two solvers can be started from the same state and compared.
  * `false` (default) — the two-state discontinuity at `interface_x`.
  * `true` — linearly ramp the primitive variables `(n, ux, vy, vz, T)` from
    `left_state` at `x_min` to `right_state` at `x_max`, sampled at cell centres,
    with each cell initialized to an exact Maxwellian at its interpolated
    primitives. The endpoints are the domain *edges*, so the ramp lines up with
    the wall interfaces rather than being offset by `dx/2`; setting
    `left_state = (n=1.0, ux=0.0, vy=-u_wall/2, T=T_wall)` and the mirrored
    `right_state` starts a Couette run on the linear (no-slip) profile instead
    of at rest.
* `wall_left`, `wall_right`: NamedTuples `(T, vy, vz)` for the diffuse-reflection
  walls (temperature and tangential velocity; normal velocity is zero).

Returns a NamedTuple with the final VDF `f`, the final constraint moments
`moments_state` (same `(m_constraint, n_cells+2)` layout as the moment
solvers), spatial/grid metadata and the power set.
"""
function run_1d_dvm(; max_moment_constraint::Int=4,
                  n_v::Int=20, extent=4.0,
                  n_cells::Int=100, x_min=0.0, x_max=1.0,
                  omega=0.5, mu_sref=1.0,
                  CFL=0.4, t_end=0.2,
                  left_state=(n=1.0, ux=0.0, T=1.0),
                  right_state=(n=1.0, ux=0.0, T=0.5),
                  interface_x=nothing,
                  wall_left=(T=1.0, vy=0.0, vz=0.0),
                  wall_right=(T=1.0, vy=0.0, vz=0.0),
                  io_path=nothing, io_freq::Int=1,
                  verbose=true, BGK_factor=1.0,
                  dissipation::Symbol=:global,
                  interpolate_init_solution::Bool=false)

    dx = (x_max - x_min) / n_cells
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

    dissipation in (:global, :upwind) ||
        throw(ArgumentError("run_1d_dvm: dissipation must be :global or :upwind, got :$dissipation"))
    dissipation_val = Val(dissipation)

    # --- output moment power set (diagnostics only — no closure here) ---
    moment_powers_constraint = all_powers_up_to_M_3D(max_moment_constraint)
    m_constraint = length(moment_powers_constraint)

    # --- velocity grid + quadrature weights ---
    grid = Grid3D([-extent, extent], [-extent, extent], [-extent, extent], n_v, n_v, n_v)
    Δv = unroll(grid_weights(grid))
    n = n_v^3

    # --- moment measurement matrix (output + wall-flux moment diagnostics) ---
    mmm_constraint = construct_moment_measurement_matrix_3D(grid, n_v, moment_powers_constraint)

    # --- flat per-node velocity components ---
    vx_flat = unroll([grid.vx[i] for i in 1:n_v, j in 1:n_v, l in 1:n_v])
    vy_flat = unroll([grid.vy[j] for i in 1:n_v, j in 1:n_v, l in 1:n_v])
    vz_flat = unroll([grid.vz[l] for i in 1:n_v, j in 1:n_v, l in 1:n_v])

    # --- work arrays ---
    vdf_mb = VDF3D(grid)
    w_local = zeros(n)
    ref_moms = zeros(7)
    mb_params = zeros(4, n_cells + 2)
    moments_out = zeros(m_constraint, n_cells)
    T_mb_out = zeros(n_cells)

    # --- spatial state, sized with ghosts: columns 1 and n_cells+2 are ghosts ---
    f      = zeros(n, n_cells + 2)
    f_next = zeros(n, n_cells + 2)

    # --- kinetic wall BC data: density-1 wall Maxwellians (grid-matched) ---
    vdf_wall = VDF3D(grid)

    maxwell_boltzmann!(vdf_wall, grid, 1.0, 0.0, wall_left.vy, wall_left.vz, wall_left.T)
    E_wall = compute_E(vdf_wall, grid, 1.0)
    sol = find_T_and_v!(vdf_wall, grid, 0.0, wall_left.vy, wall_left.vz, E_wall,
                        0.0, wall_left.vy, wall_left.vz, (2.0/3.0) * E_wall; tol=1e-15, maxiter=50)
    if verbose
        println("Left wall sol: $(sol)")
    end
    maxwell_boltzmann!(vdf_wall, grid, 1.0, sol.ux, sol.uy, sol.uz, sol.T)
    M_wall_left = unroll(vdf_wall.w)

    maxwell_boltzmann!(vdf_wall, grid, 1.0, 0.0, wall_right.vy, wall_right.vz, wall_right.T)
    E_wall = compute_E(vdf_wall, grid, 1.0)
    sol = find_T_and_v!(vdf_wall, grid, 0.0, wall_right.vy, wall_right.vz, E_wall,
                        0.0, wall_right.vy, wall_right.vz, (2.0/3.0) * E_wall; tol=1e-15, maxiter=50)
    if verbose
        println("Right wall sol: $(sol)")
    end
    maxwell_boltzmann!(vdf_wall, grid, 1.0, sol.ux, sol.uy, sol.uz, sol.T)
    M_wall_right = unroll(vdf_wall.w)

    wallbc = (vx = vx_flat, M_left = M_wall_left, M_right = M_wall_right,
              fhalf_left = zeros(n), fhalf_right = zeros(n),
              flux_left = zeros(m_constraint), flux_right = zeros(m_constraint),
              denom_left = wall_flux_denominator(M_wall_left, vx_flat, Δv, n, +1),
              denom_right = wall_flux_denominator(M_wall_right, vx_flat, Δv, n, -1))

    x = [x_min + (i - 0.5) * dx for i in 1:n_cells]     # interior cell centers

    # --- initial condition over interior cells 2..n_cells+1 ---
    # Mirrors `run_1d` exactly, so the two solvers can be initialized identically
    # and compared. Tangential velocities are optional fields on the state
    # NamedTuples, so a plain (n, ux, T) state keeps behaving as before.
    state_u(s) = (get(s, :ux, 0.0), get(s, :vy, 0.0), get(s, :vz, 0.0))
    lu, ru = state_u(left_state), state_u(right_state)
    vdf_tmp = VDF3D(grid)

    if interpolate_init_solution
        # Linear ramp in the primitive variables from left_state at x_min to
        # right_state at x_max, sampled at cell centres; every cell gets an exact
        # Maxwellian, so the gas starts at local equilibrium everywhere.
        span = x_max - x_min
        for (ci, xc) in enumerate(x)
            θ  = span > 0 ? (xc - x_min) / span : 0.0
            nd = (1 - θ) * left_state.n + θ * right_state.n
            ux = (1 - θ) * lu[1] + θ * ru[1]
            uy = (1 - θ) * lu[2] + θ * ru[2]
            uz = (1 - θ) * lu[3] + θ * ru[3]
            Tc = (1 - θ) * left_state.T + θ * right_state.T
            maxwell_boltzmann!(vdf_tmp, grid, nd, ux, uy, uz, Tc)
            @views f[:, ci+1] .= unroll(vdf_tmp.w)
        end
    else
        # two-state shock-tube discontinuity at `interface_x`
        interface = isnothing(interface_x) ? 0.5 * (x_min + x_max) : interface_x
        maxwell_boltzmann!(vdf_tmp, grid, left_state.n, lu[1], lu[2], lu[3], left_state.T)
        f_left0 = unroll(vdf_tmp.w)
        maxwell_boltzmann!(vdf_tmp, grid, right_state.n, ru[1], ru[2], ru[3], right_state.T)
        f_right0 = unroll(vdf_tmp.w)
        for (ci, xc) in enumerate(x)
            col = ci + 1
            @views f[:, col] .= xc < interface ? f_left0 : f_right0
        end
    end
    # ghost state columns are unused (wall fluxes replace them); left at zero.

    if verbose
        println("run_1d_dvm: M(output)=$max_moment_constraint, m_constraint=$m_constraint")
        println("  grid $(n_v)^3, extent [-$extent,$extent]; spatial n_cells=$n_cells, dx=$dx")
        println("  omega=$omega, CFL=$CFL, dt=$dt, n_t=$n_t, t_end≈$(n_t*dt)")
        println("  dissipation=:$dissipation, mu_sref=$mu_sref")
    end

    # --- open time-evolution output (streaming HDF5); write the t=0 frame ---
    io = nothing
    if !isnothing(io_path)
        io = init_moment_io(io_path, x, moment_powers_constraint, m_constraint, n_cells;
                            dt=dt, dx=dx, omega=omega, lambda=0.0, io_freq=io_freq,
                            max_moment_constraint=max_moment_constraint, extent=extent,
                            x_min=x_min, x_max=x_max, n_cells=n_cells,
                            mu_sref=mu_sref, dissipation=string(dissipation),
                            interpolate_init_solution=string(interpolate_init_solution))
        for cell in 2:(n_cells+1)
            ci = cell - 1
            for j in 1:m_constraint
                s = 0.0
                for k in 1:n
                    s += mmm_constraint[j, k] * f[k, cell]
                end
                moments_out[j, ci] = s
            end
            dvm_raw_moments!(ref_moms, f, cell, vx_flat, vy_flat, vz_flat, Δv, n)
            sol0 = find_MB_solution_T_and_v!(vdf_mb, grid, ref_moms,
                                             1, 2, 3, 4, 5, 6, 7; tol=1e-11, maxiter=50)
            T_mb_out[ci] = sol0.T
        end
        write_moment_snapshot!(io, 0, 0.0, moments_out, T_mb_out)
    end

    try
        convect_1D_DVM!(f, f_next, dt, dx, max_eigenvalue, n_t,
                        n_cells, n_v, mmm_constraint, m_constraint, Δv, omega,
                        grid, vdf_mb, w_local, ref_moms, mb_params,
                        vy_flat, vz_flat, wallbc, io_freq, io, moments_out, T_mb_out;
                        BGK_factor=BGK_factor, mu_sref=mu_sref, verbose=verbose,
                        dissipation=dissipation_val)
    finally
        isnothing(io) || close_moment_io(io)
    end

    # final constraint moments in the (m_constraint, n_cells+2) layout of the moment solvers
    moments_state = zeros(m_constraint, n_cells + 2)
    for cell in 2:(n_cells+1)
        for j in 1:m_constraint
            s = 0.0
            for k in 1:n
                s += mmm_constraint[j, k] * f[k, cell]
            end
            moments_state[j, cell] = s
        end
    end

    return (f = f,
            moments_state = moments_state,
            x = x, dx = dx, dt = dt, n_t = n_t,
            moment_powers_constraint = moment_powers_constraint,
            grid = grid,
            dissipation = dissipation,
            interpolate_init_solution = interpolate_init_solution)
end
