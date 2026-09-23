@testset "test dual solver" begin
    @testset "Maxwell-Boltzmann reconstruction with no L1 regularization" begin
        # Set up a small grid for fast testing
        n_v = 12
        extent = 3.0
        
        grid = Grid3D([-extent, extent], [-extent, extent], [-extent, extent], n_v, n_v, n_v)
        gw = grid_weights(grid)
        Δv = unroll(gw)
        
        # Create a Maxwell-Boltzmann distribution with known parameters
        vdf_hidden_truth = VDF3D(grid)
        T_true = 1.0
        vx0_true = 0.1
        vy0_true = -0.05
        vz0_true = 0.02
        maxwell_boltzmann!(vdf_hidden_truth, grid, 1.0, vx0_true, vy0_true, vz0_true, T_true)
        
        # Unroll the true distribution
        vdf_hidden_truth_unrolled = unroll(vdf_hidden_truth.w)
        
        # Set up moment constraints - use moments up to M=4
        max_moment_constraint = 4
        moment_powers_constraint = all_powers_up_to_M_3D(max_moment_constraint)
        
        # Construct moment measurement matrix
        mmm_constraint = construct_moment_measurement_matrix_3D(grid, n_v, moment_powers_constraint)
        
        # Compute reference moments from the true distribution
        ref_moms_constraint = mmm_constraint * vdf_hidden_truth_unrolled
        
        # We'll use a weighting function of all ones (classical EQMom)
        w = ones(n_v^3)
        target_KL = ones(n_v^3)
        
        # Set up solver arrays
        n = n_v^3
        m = length(moment_powers_constraint)
        
        A = zeros((m, n))
        rnorm_A = zeros(m)
        mvec = copy(ref_moms_constraint)
        
        alpha = zeros(n)
        gsol = zeros(n)
        uvec = zeros(n)
        d_w = zeros(n)
        inv_dw = zeros(n)
        
        y0 = zeros(m)
        y_new = zeros(m)
        grad = zeros(m)
        H = zeros((m, m))
        Hreg = zeros((m, m))
        p = zeros(m)
        F_ut = UpperTriangular(H)
        F_ch = Cholesky(F_ut)
        
        info = Dict(:iterations => 0.0, :converged => 0.0, :gradnorm => NaN)
        
        # Solve with no L1 regularization (λ = 0)
        λ = 0.0
        
        full_solve_with_init!(gsol, target_KL,
                               A, rnorm_A, ref_moms_constraint, mvec, mmm_constraint,
                               alpha, uvec, d_w, inv_dw, y0, y_new, grad, H, Hreg,
                               Δv, w, λ, n, m, F_ch, p, info;
                               tol=1e-9, maxiter=100,
                               mu=1e-12, backtrack_rho=0.5, backtrack_c=1e-4, find_y0=true)
        
        # Check that the solver converged
        @test info[:converged] == 1.0
        
        # The solution gsol should be close to the original distribution
        # Since we're using λ=0 and target_KL=ones, we expect gsol ≈ vdf_hidden_truth_unrolled
        # But note: gsol is the primal solution, which needs to be compared properly
        
        # The moments of the solution should match the reference moments
        computed_moments = mmm_constraint * (gsol .* w)
        max_moment_error = maximum(abs.(computed_moments - ref_moms_constraint))

        @test max_moment_error < 1e-6 * maximum(abs.(ref_moms_constraint))
        
        # Check that the solution is not far from the initial distribution
        # We compare the L2 norm of the difference
        gsol_scaled = gsol
        truth_scaled = vdf_hidden_truth_unrolled
        
        # Normalize both to have the same L1 norm (density)
        norm_gsol = sum(gsol_scaled)
        norm_truth = sum(truth_scaled)
        
        if norm_gsol > 0 && norm_truth > 0
            gsol_normalized = gsol_scaled / norm_gsol
            truth_normalized = truth_scaled / norm_truth
            
            # Compute L2 distance between normalized distributions
            l2_dist = sqrt(sum((gsol_normalized - truth_normalized).^2))
            
            # The L2 distance should be reasonably small
            # With λ=0, the solution should be close to the original distribution
            @test l2_dist < 1e-6
        end
    end

    @testset "Maxwell-Boltzmann reconstruction with small L1 regularization and preconditioning" begin
        # Test with small L1 regularization
        n_v = 12
        extent = 3.0
        
        grid = Grid3D([-extent, extent], [-extent, extent], [-extent, extent], n_v, n_v, n_v)
        gw = grid_weights(grid)
        Δv = unroll(gw)
        
        vdf_hidden_truth = VDF3D(grid)
        maxwell_boltzmann!(vdf_hidden_truth, grid, 1.0, 0.0, 0.0, 0.0, 1.0)
        vdf_hidden_truth_unrolled = unroll(vdf_hidden_truth.w)
        
        max_moment_constraint = 4
        moment_powers_constraint = all_powers_up_to_M_3D(max_moment_constraint)
        mmm_constraint = construct_moment_measurement_matrix_3D(grid, n_v, moment_powers_constraint)
        ref_moms_constraint = mmm_constraint * vdf_hidden_truth_unrolled
        
        ndens_index = find_index(moment_powers_constraint, (0,0,0))

        vx_index = find_index(moment_powers_constraint, (1,0,0))
        vy_index = find_index(moment_powers_constraint, (0,1,0))
        vz_index = find_index(moment_powers_constraint, (0,0,1))

        Ex_index = find_index(moment_powers_constraint, (2,0,0))
        Ey_index = find_index(moment_powers_constraint, (0,2,0))
        Ez_index = find_index(moment_powers_constraint, (0,0,2))

        # used to store weighting M-B distribution
        vdf_mb = VDF3D(grid)
        T_MB = find_MB_solution!(vdf_mb, grid, ref_moms_constraint, ndens_index, vx_index, vy_index, vz_index,
                                Ex_index, Ey_index, Ez_index; tol=1e-11)
        w = unroll(vdf_mb.w)

        target_KL = ones(n_v^3)
        
        n = n_v^3
        m = length(moment_powers_constraint)
        
        A = zeros((m, n))
        rnorm_A = zeros(m)
        mvec = copy(ref_moms_constraint)
        
        alpha = zeros(n)
        gsol = zeros(n)
        uvec = zeros(n)
        d_w = zeros(n)
        inv_dw = zeros(n)
        
        y0 = zeros(m)
        y_new = zeros(m)
        grad = zeros(m)
        H = zeros((m, m))
        Hreg = zeros((m, m))
        p = zeros(m)
        F_ut = UpperTriangular(H)
        F_ch = Cholesky(F_ut)
        
        info = Dict(:iterations => 0.0, :converged => 0.0, :gradnorm => NaN)
        
        # Solve with small L1 regularization
        λ = 1e-6
        
        full_solve_with_init!(gsol, target_KL,
                               A, rnorm_A, ref_moms_constraint, mvec, mmm_constraint,
                               alpha, uvec, d_w, inv_dw, y0, y_new, grad, H, Hreg,
                               Δv, w, λ, n, m, F_ch, p, info;
                               tol=1e-9, maxiter=100,
                               mu=1e-12, backtrack_rho=0.5, backtrack_c=1e-4, find_y0=true)
        
        @test info[:converged] == 1.0

        # sparsity
        @test sum(gsol .< 1e-8) > 0
        
        # Check that the constraints are satisfied
        computed_moments = mmm_constraint * (gsol .* w)
        max_moment_error = maximum(abs.(computed_moments - ref_moms_constraint))

        @test max_moment_error < 1e-6 * maximum(abs.(ref_moms_constraint))
    end
end
