@testset "1D LF solver" begin
    # Transport (LF in the interior + kinetic diffuse-reflection wall fluxes) must
    # conserve total mass. We isolate transport from the (WIP, density-changing)
    # BGK relaxation by setting BGK_factor=0, so no relaxation is applied.
    n_cells = 8
    x_min, x_max = 0.0, 1.0
    dx = (x_max - x_min) / n_cells

    # total mass = ∫ ρ dx over the interior cells (columns 2..n_cells+1)
    domain_mass(res) = begin
        di = find_index(res.moment_powers_constraint, (0, 0, 0))
        sum(res.moments_state[di, 2:(n_cells + 1)]) * dx
    end

    @testset "uniform equilibrium conserves mass (BGK_factor=0)" begin
        res = run_1d(; max_moment_constraint=2, n_v=8, extent=5.0, n_cells=n_cells,
                       x_min=x_min, x_max=x_max, t_end=0.05,
                       left_state=(n=1.0, ux=0.0, T=1.0), right_state=(n=1.0, ux=0.0, T=1.0),
                       wall_left=(T=1.0, vy=0.0, vz=0.0), wall_right=(T=1.0, vy=0.0, vz=0.0),
                       BGK_factor=0.0, verbose=false)

        @test all(isfinite, res.moments_state)
        # density is 1 in every cell initially, so total mass = domain length = 1
        @test domain_mass(res) ≈ (x_max - x_min) rtol = 1e-10

        # a uniform equilibrium (zero bulk velocity) must remain uniform under transport
        di = find_index(res.moment_powers_constraint, (0, 0, 0))
        density = res.moments_state[di, 2:(n_cells + 1)]
        @test all(d -> isapprox(d, 1.0; rtol=1e-8), density)
    end

    @testset "uniform equilibrium conserves mass (BGK_factor=0), n(t=0) == 2" begin
        res = run_1d(; max_moment_constraint=2, n_v=8, extent=5.0, n_cells=n_cells,
                       x_min=x_min, x_max=x_max, t_end=0.05,
                       left_state=(n=2.0, ux=0.0, T=1.0), right_state=(n=2.0, ux=0.0, T=1.0),
                       wall_left=(T=1.0, vy=0.0, vz=0.0), wall_right=(T=1.0, vy=0.0, vz=0.0),
                       BGK_factor=0.0, verbose=false)

        @test all(isfinite, res.moments_state)
        # density is 1 in every cell initially, so total mass = domain length = 1
        @test domain_mass(res) ≈ 2.0 * (x_max - x_min) rtol = 1e-10

        # a uniform equilibrium (zero bulk velocity) must remain uniform under transport
        di = find_index(res.moment_powers_constraint, (0, 0, 0))
        density = res.moments_state[di, 2:(n_cells + 1)]
        @test all(d -> isapprox(d, 2.0; rtol=1e-8), density)
    end

    @testset "two-state transport conserves mass (BGK_factor=0)" begin
        # a temperature discontinuity drives transport; density redistributes but
        # total mass must stay constant (interior fluxes telescope, walls are
        # zero-mass-flux diffuse reflectors).
        res = run_1d(; max_moment_constraint=2, n_v=8, extent=5.0, n_cells=n_cells,
                       x_min=x_min, x_max=x_max, t_end=0.05,
                       left_state=(n=1.0, ux=0.0, T=1.0), right_state=(n=1.0, ux=0.0, T=0.5),
                       wall_left=(T=1.0, vy=0.0, vz=0.0), wall_right=(T=0.5, vy=0.0, vz=0.0),
                       BGK_factor=0.0, verbose=false)

        @test all(isfinite, res.moments_state)
        @test domain_mass(res) ≈ (x_max - x_min) rtol = 1e-10
    end
end

@testset "moment flux kernels" begin
    # A delta VDF at v_x = c has (a+1,b,c)-moment = c*u and (a+2,b,c)-moment = c^2*u,
    # and half-space moments fplus = c*u, fminus = 0 for c > 0 (swapped for c < 0),
    # so the kernels must reduce to scalar advection at speed c. This exercises the
    # flux formulas without involving the dual reconstruction. Both signs of c are
    # run, so the upwind split is actually exercised.
    a_max = 4.0

    # scheme ∈ (:lf, :upwind, :lf_lw, :upwind_lw); phi_val is ignored by the
    # first-order schemes
    function advect(N, nsteps, dt, dx, scheme; phi_val=1.0, c=1.0)
        u = zeros(1, N + 2); un = zeros(1, N + 2)
        f = zeros(1, N + 2); f2 = zeros(1, N + 2)
        fp = zeros(1, N + 2); fm = zeros(1, N + 2)
        phi = fill(phi_val, N + 2)
        cp = c > 0 ? c : 0.0
        cm = c < 0 ? c : 0.0
        xs = [(i - 0.5) * dx for i in 1:N]
        # smooth bump, far from both walls in either direction of travel
        u0(z) = exp(-200 * (z - 0.5)^2)
        for i in 1:N
            u[1, i+1] = u0(xs[i])
        end
        wl, wr = zeros(1), zeros(1)
        for _ in 1:nsteps
            @. f  = c * u
            @. f2 = c^2 * u
            @. fp = cp * u
            @. fm = cm * u
            if scheme === :lf
                lax_friedrichs_wall_bc!(un, u, f, wl, wr, dt, dx, a_max, N, 1)
            elseif scheme === :upwind
                kinetic_upwind_wall_bc!(un, u, fp, fm, wl, wr, dt, dx, N, 1)
            elseif scheme === :lf_lw
                lax_wendroff_wall_bc!(un, u, f, f2, wl, wr, phi, dt, dx, a_max, N, 1)
            elseif scheme === :upwind_lw
                upwind_lax_wendroff_wall_bc!(un, u, f, f2, fp, fm, wl, wr, phi,
                                             dt, dx, N, 1)
            else
                error("unknown scheme :$scheme")
            end
            u[:, 2:(N+1)] .= un[:, 2:(N+1)]
        end
        return xs, u[1, 2:(N+1)], u0
    end

    @testset "phi = 0 reproduces the matching low-order flux bitwise" begin
        dx = 1 / 50; dt = 0.4 * dx / a_max
        for c in (1.0, -1.0)
            for (lo, hi) in ((:lf, :lf_lw), (:upwind, :upwind_lw))
                _, u_lo, _ = advect(50, 20, dt, dx, lo; c=c)
                _, u_hi, _ = advect(50, 20, dt, dx, hi; phi_val=0.0, c=c)
                @test u_lo == u_hi
            end
        end
    end

    @testset "phi = 1 is second-order on a smooth profile" begin
        for scheme in (:lf_lw, :upwind_lw)
            errs = Float64[]
            for N in (100, 200, 400)
                dx = 1.0 / N; dt = 0.4 * dx / a_max; nsteps = round(Int, 0.1 / dt)
                xs, u, u0 = advect(N, nsteps, dt, dx, scheme)
                exact = [u0(xi - nsteps * dt) for xi in xs]
                push!(errs, sum(abs.(u .- exact)) * dx)
            end
            # observed orders are ~2 for both baselines; allow a little slack
            @test log2(errs[1] / errs[2]) > 1.8
            @test log2(errs[2] / errs[3]) > 1.8
        end
    end

    @testset "phi = 1 gives the same flux on either baseline" begin
        # F_LW is independent of the low-order baseline, so the two Lax-Wendroff
        # kernels must agree at phi = 1. Run on a state where fplus/fminus are
        # unrelated to a single advection speed, so a sign error in the split
        # cannot cancel: only F_LW = 0.5*(f_L+f_R) - (dt/2dx)*(g_R-g_L) survives.
        N, nm = 12, 3
        dx = 1 / N; dt = 0.4 * dx / a_max
        u  = [0.7 + 0.2 * sinpi(0.3 * i + 0.4 * m) for m in 1:nm, i in 1:(N+2)]
        fp = [0.4 + 0.3 * cospi(0.2 * i + 0.5 * m) for m in 1:nm, i in 1:(N+2)]
        fm = [-0.3 - 0.2 * sinpi(0.7 * i - 0.3 * m) for m in 1:nm, i in 1:(N+2)]
        f  = fp .+ fm
        f2 = [1.1 + 0.4 * cospi(0.5 * i + 0.1 * m) for m in 1:nm, i in 1:(N+2)]
        wl, wr = zeros(nm), zeros(nm)
        phi = ones(N + 2)
        un_lf = zeros(nm, N + 2); un_up = zeros(nm, N + 2)
        lax_wendroff_wall_bc!(un_lf, u, f, f2, wl, wr, phi, dt, dx, a_max, N, nm)
        upwind_lax_wendroff_wall_bc!(un_up, u, f, f2, fp, fm, wl, wr, phi, dt, dx, N, nm)
        @test un_lf[:, 2:(N+1)] ≈ un_up[:, 2:(N+1)] rtol = 1e-12
    end

    @testset "the upwind kernel is first-order upwind advection" begin
        N = 40; dx = 1 / N; dt = 0.4 * dx / a_max
        for c in (1.0, -1.0)
            _, u_k, u0 = advect(N, 15, dt, dx, :upwind; c=c)
            # hand-rolled scalar upwind with the same zero wall fluxes
            u = zeros(N + 2); un = zeros(N + 2)
            xs = [(i - 0.5) * dx for i in 1:N]
            for i in 1:N
                u[i+1] = u0(xs[i])
            end
            for _ in 1:15
                for i in 2:(N+1)
                    fl = i == 2     ? 0.0 : (c > 0 ? c * u[i-1] : c * u[i])
                    fr = i == N + 1 ? 0.0 : (c > 0 ? c * u[i]   : c * u[i+1])
                    un[i] = u[i] - dt * (1.0 / dx) * (fr - fl)
                end
                u[2:(N+1)] .= un[2:(N+1)]
            end
            @test u_k == u[2:(N+1)]
        end
    end

    @testset "the upwind kernel is less diffusive than Lax-Friedrichs" begin
        # same bump, same resolution: a_k = |c| = 1 instead of a = extent = 4
        N = 200; dx = 1.0 / N; dt = 0.4 * dx / a_max; nsteps = round(Int, 0.1 / dt)
        l1(scheme, c) = begin
            xs, u, u0 = advect(N, nsteps, dt, dx, scheme; c=c)
            exact = [u0(xi - c * nsteps * dt) for xi in xs]
            sum(abs.(u .- exact)) * dx
        end
        for c in (1.0, -1.0)
            @test l1(:upwind, c) < 0.5 * l1(:lf, c)
        end
    end
end

@testset "run_1d flux switch" begin
    n_cells = 8
    x_min, x_max = 0.0, 1.0
    dx = (x_max - x_min) / n_cells

    lf_run(; kwargs...) = run_1d(; max_moment_constraint=2, n_v=8, extent=4.0,
                                   n_cells=n_cells, x_min=x_min, x_max=x_max,
                                   t_end=0.03, mu_sref=0.05,
                                   left_state=(n=1.0, ux=0.0, T=1.0),
                                   right_state=(n=0.125, ux=0.0, T=0.8),
                                   verbose=false, kwargs...)

    runs = Dict(s => lf_run(; flux=s) for s in (:lf, :lf_lw, :upwind, :upwind_lw))
    res_lf, res_lw = runs[:lf], runs[:lf_lw]
    res_up, res_uplw = runs[:upwind], runs[:upwind_lw]
    mass0 = 0.5 * (x_max - x_min) * (1.0 + 0.125)

    domain_mass(res) = begin
        di = find_index(res.moment_powers_constraint, (0, 0, 0))
        sum(res.moments_state[di, 2:(n_cells + 1)]) * dx
    end

    @testset "all four fluxes are finite and conserve mass" begin
        for (s, res) in runs
            @test all(isfinite, res.moments_state)
            @test domain_mass(res) ≈ mass0 rtol = 1e-10
            @test res.flux === s
        end
    end

    @testset ":lf is the default and its moment set is unchanged" begin
        @test lf_run().moments_state == res_lf.moments_state       # bitwise
        # the first-order fluxes keep up_to(M+1); the LW variants extend to
        # up_to(M+2) for the correction moments
        for res in (res_lf, res_up)
            @test maximum(sum, res.moment_powers_all) == 3
            @test isempty(res.next2_dir_index_x)
        end
        for res in (res_lw, res_uplw)
            @test maximum(sum, res.moment_powers_all) == 4
            @test length(res.next2_dir_index_x) == length(res.moment_powers_constraint)
        end
    end

    @testset "next2_dir_index_x points at the (a+2,b,c) moment" begin
        for res in (res_lw, res_uplw)
            for (m, p) in enumerate(res.moment_powers_constraint)
                @test res.moment_powers_all[res.next2_dir_index_x[m]] ==
                      (p[1] + 2, p[2], p[3])
            end
        end
    end

    @testset "half-space moments sum to the (a+1,b,c) flux moment" begin
        # A⁺[f] + A⁻[f] = Σ_k m_j(v) v_x f Δv = the next x-moment, which the
        # Lax-Friedrichs path reads out of moments_work via next_dir_index_x.
        # Summation order differs, so this is ≈, not ==; the atol covers the
        # moments that vanish by symmetry, where the two orders cancel to
        # different round-off. Catches masking and sign errors in
        # mmm_plus / mmm_minus.
        for res in (res_up, res_uplw)
            @test !isempty(res.fplus)
            @test all(isapprox(res.fplus[m, i] + res.fminus[m, i],
                               res.moments_work[idx, i]; rtol=1e-10, atol=1e-12)
                      for (m, idx) in enumerate(res.next_dir_index_x)
                      for i in 2:(n_cells + 1))
        end
        @test isempty(res_lf.fplus) && isempty(res_lf.fminus)
    end

    @testset "the half-moments carry the expected signs" begin
        # positive x-flux of a positive VDF: A⁺[density] > 0 > A⁻[density]
        di = find_index(res_up.moment_powers_constraint, (0, 0, 0))
        for i in 2:(n_cells + 1)
            @test res_up.fplus[di, i] > 0.0
            @test res_up.fminus[di, i] < 0.0
        end
    end

    @testset "invalid flux is rejected" begin
        @test_throws ArgumentError lf_run(; flux=:lw)
        @test_throws ArgumentError lf_run(; flux=:lax_wendroff)
    end
end

@testset "moment :upwind tracks the DVM reference" begin
    # the moment solver and the pure DVM solver differ only in the closure, so on
    # a short run from the same initial condition with the same (upwind) flux they
    # must stay close. Coarse grid, few steps, BGK off to isolate transport.
    n_cells = 8
    x_min, x_max = 0.0, 1.0
    common = (; max_moment_constraint=4, n_v=8, extent=4.0, n_cells=n_cells,
                x_min=x_min, x_max=x_max, t_end=0.02, BGK_factor=0.0,
                left_state=(n=1.0, ux=0.0, T=1.0),
                right_state=(n=0.5, ux=0.0, T=0.8),
                verbose=false)

    res_m = run_1d(; common..., flux=:upwind)
    res_d = run_1d_dvm(; common..., dissipation=:upwind)
    res_g = run_1d_dvm(; common..., dissipation=:global)

    di = find_index(res_m.moment_powers_constraint, (0, 0, 0))
    rho(res) = res.moments_state[di, 2:(n_cells + 1)]

    # at M=4 over this short run the closure error is far below the difference
    # between the two dissipations, so this is a sharp check on the flux
    @test rho(res_m) ≈ rho(res_d) rtol = 1e-6
    # ... and it is not vacuous: the wrong pairing is ~17% off
    @test !isapprox(rho(res_m), rho(res_g); rtol=1e-2)
end

@testset "HDF5 output carries phi without disturbing the other writers" begin
    # /phi is written by the Lax-Wendroff variants only; every other caller —
    # including the DVM solver, which passes T_MB positionally — must keep
    # writing exactly what it wrote before, with an all-NaN /phi.
    n_cells = 8
    dir = mktempdir()
    common = (; max_moment_constraint=2, n_v=8, extent=4.0, n_cells=n_cells,
                x_min=0.0, x_max=1.0, t_end=0.03, mu_sref=0.05,
                left_state=(n=1.0, ux=0.0, T=1.0), right_state=(n=0.125, ux=0.0, T=0.8),
                verbose=false, io_freq=1)

    paths = Dict{Symbol,String}()
    for s in (:lf, :lf_lw, :upwind, :upwind_lw)
        paths[s] = joinpath(dir, "$s.h5")
        run_1d(; common..., flux=s, io_path=paths[s])
    end
    paths[:dvm] = joinpath(dir, "dvm.h5")
    run_1d_dvm(; common..., dissipation=:upwind, io_path=paths[:dvm])

    read_ds(p, name) = HDF5.h5open(f -> read(f[name]), p)

    for (s, p) in paths
        phi = read_ds(p, "phi")
        @test size(phi) == (n_cells, size(read_ds(p, "moments"), 3))
        # only the LW variants have a limiter to report
        @test all(isnan, phi) == !(s in (:lf_lw, :upwind_lw))
        if s in (:lf_lw, :upwind_lw)
            interior = phi[1:(n_cells - 1), end]      # last column is the wall interface
            @test all(0.0 .<= interior .<= 1.0)
            @test phi[n_cells, end] == 0.0
        end
    end

    # the DVM writer still fills T_MB (it passes it positionally); run_1d does not
    @test !any(isnan, read_ds(paths[:dvm], "T_MB"))
    @test all(isnan, read_ds(paths[:lf], "T_MB"))
end

@testset "interpolate_init_solution" begin
    n_cells = 10
    x_min, x_max = 0.0, 1.0
    base(; kwargs...) = run_1d(; max_moment_constraint=2, n_v=8, extent=5.0,
                                 n_cells=n_cells, x_min=x_min, x_max=x_max,
                                 t_end=0.0001, BGK_factor=0.0, verbose=false, kwargs...)
    L = (n=1.0, ux=0.0, T=1.0)
    R = (n=2.0, ux=0.0, T=1.5)

    # `run_1d` always takes at least one step, so `moments_state` is the state
    # *after* transport. The t=0 snapshot in the output file is the initial
    # condition itself, which is what these tests are about.
    dir = mktempdir()
    ic_counter = Ref(0)
    function initial_moments(; kwargs...)
        p = joinpath(dir, "ic$(ic_counter[] += 1).h5")
        res = base(; io_path=p, io_freq=1000, kwargs...)
        M = HDF5.h5open(f -> read(f["moments"]), p)
        return res, M[:, :, 1]                     # snapshot 1 is t = 0
    end

    @testset "default is off and leaves the two-state IC bitwise unchanged" begin
        @test base(; left_state=L, right_state=R).interpolate_init_solution == false
        @test base(; left_state=L, right_state=R, interpolate_init_solution=false).moments_state ==
              base(; left_state=L, right_state=R).moments_state
    end

    @testset "off: a sharp jump at the interface" begin
        res, M0 = initial_moments(; left_state=L, right_state=R, interpolate_init_solution=false)
        di = find_index(res.moment_powers_constraint, (0,0,0))
        rho = M0[di, :]
        @test rho[1] ≈ rho[n_cells ÷ 2] rtol = 1e-10        # flat on the left
        @test rho[n_cells ÷ 2 + 1] ≈ rho[end] rtol = 1e-10  # flat on the right
        @test rho[end] / rho[1] ≈ 2.0 rtol = 1e-6
    end

    @testset "on: a linear ramp between the two states" begin
        res, M0 = initial_moments(; left_state=L, right_state=R, interpolate_init_solution=true)
        di = find_index(res.moment_powers_constraint, (0,0,0))
        rho = M0[di, :]
        # endpoints are the domain edges, so cell centres sit half a cell inside
        dx = (x_max - x_min) / n_cells
        expect = [1.0 + ((x_min + (i - 0.5) * dx - x_min) / (x_max - x_min)) * (2.0 - 1.0)
                  for i in 1:n_cells]
        @test rho ≈ expect rtol = 1e-6
        @test all(diff(rho) .> 0)                             # monotone, no jump
        @test maximum(abs.(diff(diff(rho)))) < 1e-8           # and straight
    end

    @testset "tangential velocity ramps too (the Couette use case)" begin
        uw = 2.0
        res, M0 = initial_moments(; left_state=(n=1.0, ux=0.0, vy=-uw/2, T=1.0),
                                    right_state=(n=1.0, ux=0.0, vy=+uw/2, T=1.0),
                                    interpolate_init_solution=true)
        di = find_index(res.moment_powers_constraint, (0,0,0))
        vi = find_index(res.moment_powers_constraint, (0,1,0))
        uy = M0[vi, :] ./ M0[di, :]

        # Compare against Maxwellians built independently at the interpolated
        # primitives. Comparing against the requested vy directly would instead
        # be measuring `maxwell_boltzmann!`'s lattice sampling error, which at
        # n_v=8 is ~3% (the sampled-vs-discretely-matched distinction) and has
        # nothing to do with the interpolation under test.
        mmm = construct_moment_measurement_matrix_3D(res.grid, 8, res.moment_powers_constraint)
        dx = (x_max - x_min) / n_cells
        for i in 1:n_cells
            θ = (i - 0.5) * dx / (x_max - x_min)
            want = maxwellian_constraint_moments(res.grid, mmm, 1.0, 0.0, -uw/2 + θ*uw, 0.0, 1.0)
            @test M0[:, i] ≈ want rtol = 1e-12
        end

        # structural properties, robust to the sampling error. Linearity of the
        # ramp is already covered exactly by the per-cell comparison above; the
        # *measured* uy is not straight at n_v=8 because the sampling error is
        # itself a nonlinear function of the drift (-0.9 reads as -0.875, -0.1
        # as -0.115), which again is the sampler, not the interpolation.
        @test all(diff(uy) .> 0)                              # monotone ramp
        @test uy[1] ≈ -uy[end] rtol = 1e-10                   # antisymmetric
    end

    @testset "vy/vz are optional and default to zero" begin
        # a plain (n, ux, T) state must still give a quiescent gas
        res, M0 = initial_moments(; left_state=L, right_state=L, interpolate_init_solution=true)
        for p in ((1,0,0), (0,1,0), (0,0,1))
            vi = find_index(res.moment_powers_constraint, p)
            @test all(abs.(M0[vi, :]) .< 1e-12)
        end
    end

    @testset "the DVM driver initializes identically" begin
        # the whole point of duplicating the option is that both solvers can be
        # started from the same state; the DVM builds the same per-cell
        # Maxwellians, so its t=0 constraint moments must match the moment
        # solver's to round-off.
        dvm_base(; kwargs...) = run_1d_dvm(; max_moment_constraint=2, n_v=8, extent=5.0,
                                             n_cells=n_cells, x_min=x_min, x_max=x_max,
                                             t_end=0.0001, BGK_factor=0.0,
                                             verbose=false, kwargs...)
        for (lab, kw) in (("ramp",  (; left_state=(n=1.0, ux=0.0, vy=-1.0, T=1.0),
                                       right_state=(n=2.0, ux=0.0, vy=1.0, T=1.5),
                                       interpolate_init_solution=true)),
                          ("sharp", (; left_state=L, right_state=R,
                                       interpolate_init_solution=false)))
            @testset "$lab" begin
                _, M0 = initial_moments(; kw...)
                p = joinpath(dir, "dvm_$lab.h5")
                dvm_base(; kw..., io_path=p, io_freq=1000)
                D0 = HDF5.h5open(f -> read(f["moments"]), p)[:, :, 1]
                @test D0 ≈ M0 rtol = 1e-12
            end
        end

        @test dvm_base(; left_state=L, right_state=R).interpolate_init_solution == false
        # default off is bitwise the previous two-state behaviour
        @test dvm_base(; left_state=L, right_state=R, interpolate_init_solution=false).f ==
              dvm_base(; left_state=L, right_state=R).f
    end
end

@testset "shared limiter" begin
    # phi is a single scalar per interface; indicators are density and temperature
    n_cells = 6
    dens, vx, vy, vz, Ex, Ey, Ez = 1, 2, 3, 4, 5, 6, 7
    mom = zeros(7, n_cells + 2)
    phi = zeros(n_cells + 2); ind_n = zeros(n_cells + 2); ind_T = zeros(n_cells + 2)

    lim!() = shared_limiter!(phi, ind_n, ind_T, mom, dens, vx, vy, vz, Ex, Ey, Ez, n_cells)

    @testset "uniform state gives phi = 1 on every usable interface" begin
        for i in 2:(n_cells+1)
            mom[dens, i] = 1.0
            mom[Ex, i] = mom[Ey, i] = mom[Ez, i] = 1.0    # T = 1
        end
        lim!()
        @test all(phi[i] ≈ 1.0 for i in 3:(n_cells-1))
    end

    @testset "a jump drives phi to 0 at the discontinuity" begin
        for i in 2:(n_cells+1)
            nl = i <= (n_cells ÷ 2) + 1 ? 1.0 : 0.125
            mom[dens, i] = nl
            mom[Ex, i] = mom[Ey, i] = mom[Ez, i] = nl     # T = 1 either side
        end
        lim!()
        jump = (n_cells ÷ 2) + 1                          # interface (jump, jump+1)
        @test phi[jump] == 0.0
        @test all(0.0 <= phi[i] <= 1.0 for i in 2:n_cells)
    end

    @testset "a nearly-flat indicator does not read as a field of extrema" begin
        # regression: the flatness guard was scaled by |s| (rtol=1e-12), which for
        # T~0.55 sits ~7 orders below the truncation-scale differences in a flat
        # region, so it never fired and the limiter returned 0 all through the
        # middle of a Couette domain. The scale must come from the indicator's
        # variation across the domain instead.
        for i in 2:(n_cells+1)
            mom[dens, i] = 1.0 + 0.1 * (i - 2) / n_cells          # gentle real trend
            # T flat to 1 part in 1e5, wobbling at truncation scale
            Tl = 0.55 + 1e-5 * (iseven(i) ? 1.0 : -1.0)
            mom[Ex, i] = mom[Ey, i] = mom[Ez, i] = Tl
        end
        lim!()
        # the T wobble is 2e-5 against a span of 2e-5, i.e. entirely flat: it must
        # not veto the correction, and the smooth density trend must survive
        @test count(==(0.0), phi[3:(n_cells-1)]) == 0
        @test all(phi[i] > 0.5 for i in 3:(n_cells-1))
    end

    @testset "a smooth extremum still vetoes the correction" begin
        # Couette at even n_cells puts a steady, smooth density minimum exactly on
        # a cell interface. d_loc there is 0 (mirror symmetry) so the flat-band
        # branch fires and phi = 1, while the two flanking interfaces straddle the
        # extremum, get d_up*d_loc <= 0 and phi = 0: a low-viscosity slit walled in
        # by two high-viscosity interfaces. With a Lax-Friedrichs baseline that is
        # a factor ~20-80 in numerical viscosity, which is what puts wiggles in the
        # M_110 profile.
        #
        # This is the motivating case for a split phi+/phi- limiter; it is
        # deliberately *not* asserted on the existing shared limiter, whose
        # two-sided-min behaviour is pinned by the testsets above.
        for i in 2:(n_cells+1)
            xc = (i - 1.5) / n_cells
            mom[dens, i] = 1.0 + 0.2 * (xc - 0.5)^2               # smooth minimum at x = 0.5
            mom[Ex, i] = mom[Ey, i] = mom[Ez, i] = 1.0            # T flat
        end
        lim!()
        mid = (n_cells ÷ 2) + 1                                   # interface at the minimum
        @test phi[mid] ≈ 1.0                                      # flat-band branch
        # the flanking interfaces are the problem: smooth data, but phi = 0
        @test phi[mid-1] == 0.0
        @test phi[mid+1] == 0.0
        # what a direction-split limiter should give instead:
        @test_broken all(phi[i] > 0.5 for i in (mid-1, mid+1))
    end

    @testset "wall-adjacent interfaces fall back to Lax-Friedrichs" begin
        # (state here is whatever the previous testset left in `mom`/`phi`)
        # interface 2 has no cell 1 and interface n_cells has no cell n_cells+2,
        # but each still has one usable direction, so only a 2-cell domain is fully blind
        @test phi[1] == 0.0
        @test phi[n_cells+1] == 0.0
    end
end

@testset "DVM dissipation switch" begin
    n_cells = 8
    x_min, x_max = 0.0, 1.0
    dx = (x_max - x_min) / n_cells

    dvm_run(; kwargs...) = run_1d_dvm(; max_moment_constraint=2, n_v=8, extent=4.0,
                                        n_cells=n_cells, x_min=x_min, x_max=x_max,
                                        t_end=0.05, mu_sref=0.05,
                                        left_state=(n=1.0, ux=0.0, T=1.0),
                                        right_state=(n=0.125, ux=0.0, T=0.8),
                                        verbose=false, kwargs...)

    domain_mass(res) = begin
        di = find_index(res.moment_powers_constraint, (0, 0, 0))
        sum(res.moments_state[di, 2:(n_cells + 1)]) * dx
    end
    # left half at n=1, right half at n=0.125
    mass0 = 0.5 * (x_max - x_min) * (1.0 + 0.125)

    res_g = dvm_run(; dissipation=:global)
    res_u = dvm_run(; dissipation=:upwind)

    @testset "both variants are finite, positive and conservative" begin
        for res in (res_g, res_u)
            @test all(isfinite, res.f)
            @test minimum(res.f) >= 0.0                      # positivity under CFL <= 1
            @test domain_mass(res) ≈ mass0 rtol = 1e-10      # walls are zero-mass-flux
        end
    end

    @testset ":global is the default and is unchanged" begin
        @test res_g.dissipation === :global
        @test res_u.dissipation === :upwind
        @test dvm_run().f == res_g.f                         # bitwise, not just approx
    end

    @testset ":upwind is strictly less diffusive" begin
        # the two schemes differ only in numerical viscosity, so an initially
        # discontinuous profile stays sharper under :upwind. Measure the total
        # variation of the density profile: more diffusion => smaller TV.
        di = find_index(res_g.moment_powers_constraint, (0, 0, 0))
        tv(res) = sum(abs.(diff(res.moments_state[di, 2:(n_cells + 1)])))
        @test tv(res_u) > tv(res_g)
    end

    @testset "invalid dissipation is rejected" begin
        @test_throws ArgumentError dvm_run(; dissipation=:lax_friedrichs)
    end
end
