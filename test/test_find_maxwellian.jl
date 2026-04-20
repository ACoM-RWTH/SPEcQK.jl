@testset "find Maxwellian approximation" begin
        n_v = 24

        grid = Grid3D([-4.0, 4.0], [-4.0, 4.0], [-4.0, 4.0], n_v, n_v, n_v)
        vdf = VDF3D(grid)
        vdf_mb = VDF3D(grid)

        maxwell_boltzmann!(vdf, grid, 0.0, 0.0, 0.0, 1.25)

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

        T_found = find_MB_solution!(vdf_mb, grid, [velvec[1], Ex, velvec[2], Ey, velvec[3], Ez], 1, 3, 5,
                                    2, 4, 6; tol=1e-11)
        @test isapprox(T_found, 1.25; atol=1e-5)
end