@testset "BGK collision scale" begin
    # The canonical code temperature is the one in
    # docs/src/scaling.md, and it is exactly the matched Maxwellian's own T.
    grid = Grid3D([-8.0, 8.0], [-8.0, 8.0], [-8.0, 8.0], 24, 24, 24)
    n = 24^3
    dv = unroll(grid_weights(grid))
    vxf = unroll([grid.vx[i] for i in 1:24, j in 1:24, l in 1:24])
    vyf = unroll([grid.vy[j] for i in 1:24, j in 1:24, l in 1:24])
    vzf = unroll([grid.vz[l] for i in 1:24, j in 1:24, l in 1:24])

    # T from raw moments, exactly as docs/src/scaling.md defines it
    function code_temperature(f)
        nn = sum(f .* dv)
        u = (sum(vxf .* f .* dv), sum(vyf .* f .* dv), sum(vzf .* f .* dv)) ./ nn
        E2 = sum((vxf.^2 .+ vyf.^2 .+ vzf.^2) .* f .* dv) / nn
        return (2.0 / 3.0) * (E2 - sum(abs2, u))
    end

    @testset "the scaling.md temperature is the Maxwellian's own T" begin
        # exact in the continuum; on the discrete grid the quadrature and the
        # truncated tails cost a few digits, worst for cold, fast-drifting states
        # (T=0.6 at u_y=-1.5 is off by 2e-4). That discretization gap is exactly
        # why the solvers pass the *matched* Maxwellian's mb_T rather than
        # recomputing T from the raw moments.
        for (T, uy) in ((1.0, 0.0), (1.0, 1.0), (1.3, 0.7), (0.6, -1.5))
            v = VDF3D(grid)
            maxwell_boltzmann!(v, grid, 1.0, 0.0, uy, 0.0, T)
            @test code_temperature(unroll(v.w)) ≈ T rtol = 1e-3
        end
    end

    @testset "the matched Maxwellian's T is that same temperature" begin
        # this is what the solvers now pass to tau_BGK
        powers = all_powers_up_to_M_3D(2)
        mmm = construct_moment_measurement_matrix_3D(grid, 24, powers)
        idx(p) = find_index(powers, p)
        for (T, uy) in ((1.0, 0.0), (1.3, 0.7), (0.6, -1.5))
            v = VDF3D(grid)
            maxwell_boltzmann!(v, grid, 1.1, 0.0, uy, 0.0, T)
            ref = mmm * unroll(v.w)
            vdf_mb = VDF3D(grid)
            sol = find_MB_solution_T_and_v!(vdf_mb, grid, ref, idx((0,0,0)),
                                            idx((1,0,0)), idx((0,1,0)), idx((0,0,1)),
                                            idx((2,0,0)), idx((0,2,0)), idx((0,0,2));
                                            tol=1e-11, maxiter=50)
            @test sol.converged
            @test sol.T ≈ T rtol = 1e-6
        end
    end

    @testset "tau_BGK is mu/p with p = nT" begin
        for (nn, T, om, mu) in ((1.0,1.0,0.5,0.3), (2.0,1.3,0.5,0.3), (0.7,0.8,1.0,1.5))
            tau = tau_BGK(nn, T, om; mu_sref=mu)
            @test tau ≈ (mu * T^om) / (nn * T) rtol = 1e-12   # tau = mu(T)/p
            @test tau * nn * T ≈ mu * T^om rtol = 1e-12       # recovers the VHS viscosity
        end
        # at n = T = 1 the collision time is exactly mu_sref -- but the n really
        # is needed, tau = mu_sref/n there, which is the thing easy to misremember
        @test tau_BGK(1.0, 1.0, 0.5; mu_sref=0.3) ≈ 0.3
        @test tau_BGK(2.0, 1.0, 0.5; mu_sref=0.3) ≈ 0.15
        # Maxwell molecules: T-independent
        @test tau_BGK(1.0, 3.0, 1.0; mu_sref=0.3) ≈ tau_BGK(1.0, 0.2, 1.0; mu_sref=0.3)
    end

    @testset "Knudsen_number tracks mu_sref * T^(omega-1/2)/n" begin
        # C_omega is an O(1) constant, so the *ratios* are what to pin
        kn(nn, T, om) = Knudsen_number(nn, T, om, 1.0, 0.3)
        @test kn(1.0, 1.0, 0.5) / kn(2.0, 1.0, 0.5) ≈ 2.0 rtol = 1e-12
        @test kn(1.0, 4.0, 1.0) / kn(1.0, 1.0, 1.0) ≈ 2.0 rtol = 1e-12   # T^(1/2)
        @test kn(1.0, 4.0, 0.5) ≈ kn(1.0, 1.0, 0.5) rtol = 1e-12         # T^0
        @test Knudsen_number(1.0, 1.0, 0.5, 2.0, 0.3) ≈ 0.5 * kn(1.0, 1.0, 0.5)
    end

    @testset "a strong shear flow no longer produces a negative temperature" begin
        # an old formula with a bug gave T = <v²>/3 - |u|², negative once |u| exceeded the
        # thermal speed -- e.g. a wall driven at u_y = 2 in the Couette setup
        v = VDF3D(grid)
        maxwell_boltzmann!(v, grid, 1.0, 0.0, 2.0, 0.0, 1.0)
        f = unroll(v.w)
        nn = sum(f .* dv)
        E2 = sum((vxf.^2 .+ vyf.^2 .+ vzf.^2) .* f .* dv) / nn
        @test E2 / 3 - 4.0 < 0.0                       # the old formula: negative
        @test code_temperature(f) ≈ 1.0 rtol = 1e-6    # the correct one: fine
        @test tau_BGK(nn, code_temperature(f), 0.5; mu_sref=0.3) > 0.0
    end
end
