@testset "test grids and distributions" begin
    @testset "test grid" begin
        grid = Grid3D([-1.0, 1.0], [-2.0, 2.0], [-3.0, 3.0], 4, 5, 19)

        @test grid.n_vx == 4
        @test grid.n_vy == 5
        @test grid.n_vz == 19
        @test length(grid.vx) == 4
        @test length(grid.vy) == 5
        @test length(grid.vz) == 19
        @test grid.vx[1] == -1.0
        @test grid.vx[end] == 1.0
        @test grid.vy[1] == -2.0
        @test grid.vy[end] == 2.0
        @test grid.vz[1] == -3.0
        @test grid.vz[end] == 3.0

        @test isapprox(norm(grid.Δvx .- [2.0/3.0, 2.0/3.0, 2.0/3.0, 2.0/3.0]), 0.0, atol=1e-10) # nodes: -1, -0.3, 0.3, 1.0
        @test grid.Δvy == [1.0, 1.0, 1.0, 1.0, 1.0]
        @test isapprox(norm(grid.Δvz .- [1.0/3.0 for i in 1:19]), 0.0, atol=1e-10)

        @test isapprox(norm(grid.vx.^2 - grid.vxsq), 0.0, atol=1e-10)
        @test isapprox(norm(grid.vy.^2 - grid.vysq), 0.0, atol=1e-10)
        @test isapprox(norm(grid.vz.^2 - grid.vzsq), 0.0, atol=1e-10)

        vdf = VDF3D(grid)
        @test size(vdf.w) == (4, 5, 19)
    end

    @testset "test Maxwell-Boltzmann distribution" begin
        n_v = 24

        grid = Grid3D([-4.0, 4.0], [-4.0, 4.0], [-4.0, 4.0], n_v, n_v, n_v)
        vdf = VDF3D(grid)

        maxwell_boltzmann!(vdf, grid, 1.0, 0.5, -0.25, 0.4, 1.0)

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

        @test isapprox(norm(velvec .- [0.5, -0.25, 0.4]), 0.0, atol=1e-6)

        E = 0.0
        for k in 1:n_v
            for j in 1:n_v
                for i in 1:n_v
                    E += ((grid.vx[i] - velvec[1])^2 +
                          (grid.vy[j] - velvec[2])^2 +
                          (grid.vz[k] - velvec[3])^2) .* (vdf.w[i,j,k] * gw[i,j,k])
                end
            end
        end
        T = (2.0/3.0) * E

        @test isapprox(T, 1.0, atol=1e-5)
    end

    @testset "test Maxwell-Boltzmann distribution, dens != 1" begin
        n_v = 24

        grid = Grid3D([-4.0, 4.0], [-4.0, 4.0], [-4.0, 4.0], n_v, n_v, n_v)
        vdf = VDF3D(grid)

        maxwell_boltzmann!(vdf, grid, 1.75, 0.5, -0.25, 0.4, 1.0)

        # test weights
        grid_dv = grid.Δvx[1] .* grid.Δvx[1] .* grid.Δvx[1]

        gw = SPEcQK.grid_weights(grid)
        @test isapprox(norm(gw .- grid_dv), 0.0, atol=1e-10)

        # density
        @test isapprox(sum(vdf.w .* gw), 1.75, atol=1e-10)
        rho = sum(vdf.w .* gw)

        velvec = [0.0, 0.0, 0.0]
        for k in 1:n_v
            for j in 1:n_v
                for i in 1:n_v
                    velvec += [grid.vx[i], grid.vy[j], grid.vz[k]] .* (vdf.w[i,j,k] * gw[i,j,k])
                end
            end
        end

        @test isapprox(norm((velvec ./ rho) .- [0.5, -0.25, 0.4]), 0.0, atol=1e-6)

        E = 0.0
        for k in 1:n_v
            for j in 1:n_v
                for i in 1:n_v
                    E += ((grid.vx[i] - (velvec[1] / rho))^2 +
                          (grid.vy[j] - (velvec[2] / rho))^2 +
                          (grid.vz[k] - (velvec[3] / rho))^2) .* (vdf.w[i,j,k] * gw[i,j,k])
                end
            end
        end
        T = (2.0/3.0) * E / rho

        @test isapprox(T, 1.0, atol=1e-5)
    end
end
