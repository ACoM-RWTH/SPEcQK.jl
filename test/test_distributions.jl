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
    end
end
