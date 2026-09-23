@testset "find Maxwellian approximation" begin
    @testset "dens = 1.0" begin
        n_v = 24

        grid = Grid3D([-4.0, 4.0], [-4.0, 4.0], [-4.0, 4.0], n_v, n_v, n_v)
        vdf = VDF3D(grid)
        vdf_mb = VDF3D(grid)

        maxwell_boltzmann!(vdf, grid, 1.0, 0.0, 0.0, 0.0, 1.25)

        # test weights
        grid_dv = grid.Δvx[1] .* grid.Δvx[1] .* grid.Δvx[1]

        gw = SPEcQK.grid_weights(grid)
        @test isapprox(norm(gw .- grid_dv), 0.0, atol=1e-10)

        # density
        @test isapprox(sum(vdf.w .* gw), 1.0, atol=1e-10)

        velvec = [0.0, 0.0, 0.0]
        for k in 1:n_v
            for j in 1:n_v
                for i in 1:n_v
                    velvec += [grid.vx[i], grid.vy[j], grid.vz[k]] .* (vdf.w[i,j,k] * gw[i,j,k])
                end
            end
        end

        Ex = 0.0
        Ey = 0.0
        Ez = 0.0
        for k in 1:n_v
            for j in 1:n_v
                for i in 1:n_v
                    Ex += ((grid.vx[i] - velvec[1])^2) .* (vdf.w[i,j,k] * gw[i,j,k])
                    Ey += ((grid.vy[j] - velvec[2])^2) .* (vdf.w[i,j,k] * gw[i,j,k])
                    Ez += ((grid.vz[k] - velvec[3])^2) .* (vdf.w[i,j,k] * gw[i,j,k])
                end
            end
        end

        T_found = find_MB_solution!(vdf_mb, grid, [1.0, velvec[1], Ex, velvec[2], Ey, velvec[3], Ez], 1, 2, 4, 6,
                                    3, 5, 7; tol=1e-11)
        @test isapprox(T_found, 1.25; atol=1e-5)
    end

    @testset "dens = 1.35" begin
        n_v = 24

        grid = Grid3D([-4.0, 4.0], [-4.0, 4.0], [-4.0, 4.0], n_v, n_v, n_v)
        vdf = VDF3D(grid)
        vdf_mb = VDF3D(grid)

        maxwell_boltzmann!(vdf, grid, 1.35, 0.0, 0.0, 0.0, 1.25)

        # test weights
        grid_dv = grid.Δvx[1] .* grid.Δvx[1] .* grid.Δvx[1]

        gw = SPEcQK.grid_weights(grid)
        @test isapprox(norm(gw .- grid_dv), 0.0, atol=1e-10)

        # density
        @test isapprox(sum(vdf.w .* gw), 1.35, atol=1e-10)

        velvec = [0.0, 0.0, 0.0]
        for k in 1:n_v
            for j in 1:n_v
                for i in 1:n_v
                    velvec += [grid.vx[i], grid.vy[j], grid.vz[k]] .* (vdf.w[i,j,k] * gw[i,j,k])
                end
            end
        end

        Ex = 0.0
        Ey = 0.0
        Ez = 0.0
        for k in 1:n_v
            for j in 1:n_v
                for i in 1:n_v
                    Ex += ((grid.vx[i] - velvec[1])^2) .* (vdf.w[i,j,k] * gw[i,j,k])
                    Ey += ((grid.vy[j] - velvec[2])^2) .* (vdf.w[i,j,k] * gw[i,j,k])
                    Ez += ((grid.vz[k] - velvec[3])^2) .* (vdf.w[i,j,k] * gw[i,j,k])
                end
            end
        end

        T_found = find_MB_solution!(vdf_mb, grid, [1.35, velvec[1], Ex, velvec[2], Ey, velvec[3], Ez], 1, 2, 4, 6,
                                    3, 5, 7; tol=1e-11)
        @test isapprox(T_found, 1.25; atol=1e-5)
    end

    @testset "T and v (matches velocity + energy on cut-off grid)" begin
        n_v = 24
        grid = Grid3D([-4.0, 4.0], [-4.0, 4.0], [-4.0, 4.0], n_v, n_v, n_v)
        gw = SPEcQK.grid_weights(grid)

        # reference discrete Maxwellian at a known density, velocity and temperature
        n_b, ux_b, uy_b, uz_b, T_b = 1.35, 0.6, -0.4, 0.25, 1.3
        vdf = VDF3D(grid)
        maxwell_boltzmann!(vdf, grid, n_b, ux_b, uy_b, uz_b, T_b)

        # raw discrete moments of the reference (targets, NOT divided by density)
        dens = sum(vdf.w .* gw)
        px = 0.0; py = 0.0; pz = 0.0; Exx = 0.0; Eyy = 0.0; Ezz = 0.0
        for k in 1:n_v
            for j in 1:n_v
                for i in 1:n_v
                    f = vdf.w[i,j,k] * gw[i,j,k]
                    px += grid.vx[i] * f;  py += grid.vy[j] * f;  pz += grid.vz[k] * f
                    Exx += grid.vx[i]^2 * f; Eyy += grid.vy[j]^2 * f; Ezz += grid.vz[k]^2 * f
                end
            end
        end
        ref = [dens, px, py, pz, Exx, Eyy, Ezz]

        vdf_mb = VDF3D(grid)
        sol = find_MB_solution_T_and_v!(vdf_mb, grid, ref, 1, 2, 3, 4, 5, 6, 7; tol=1e-12, maxiter=50)

        @test sol.converged
        # the reference is itself a discrete Maxwellian, so (ux,uy,uz,T) are recovered
        @test isapprox(sol.ux, ux_b; atol=1e-6)
        @test isapprox(sol.uy, uy_b; atol=1e-6)
        @test isapprox(sol.uz, uz_b; atol=1e-6)
        @test isapprox(sol.T,  T_b;  atol=1e-6)

        # the produced distribution reproduces density, mean velocity and full 2nd moment
        d2 = sum(vdf_mb.w .* gw)
        qx = 0.0; qy = 0.0; qz = 0.0; E2 = 0.0
        for k in 1:n_v
            for j in 1:n_v
                for i in 1:n_v
                    f = vdf_mb.w[i,j,k] * gw[i,j,k]
                    qx += grid.vx[i] * f; qy += grid.vy[j] * f; qz += grid.vz[k] * f
                    E2 += (grid.vx[i]^2 + grid.vy[j]^2 + grid.vz[k]^2) * f
                end
            end
        end
        @test isapprox(d2, dens; rtol=1e-10)
        @test isapprox(qx / d2, px / dens; atol=1e-9)
        @test isapprox(qy / d2, py / dens; atol=1e-9)
        @test isapprox(qz / d2, pz / dens; atol=1e-9)
        @test isapprox(E2 / d2, (Exx + Eyy + Ezz) / dens; atol=1e-9)
    end
end