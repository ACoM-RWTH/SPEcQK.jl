@muladd begin
"""
    solve_iterate_over_L1_values(lambda_L1_arr,
                                 w, target_KL, Δv,
                                 mmms_unrolled_concatenated, mmms_unrolled_concatenated_test,
                                 ref_moms, ref_moms_test, n_v; threshold=1e-6,
                                 tol=1e-9, maxiter=100,
                                 mu=1e-12, backtrack_rho=0.5, backtrack_c=1e-4, find_y0=true,
                                 verbose=2, warmup=true)

Iterate over a range of L1 regularization values and solve the optimization problem for each value, writing
the resulting VDFs and predicted moments to arrays. This allocates all required intermediate arrays.
Important: it is assumed that `mmms_unrolled_concatenated[1,:]` (i.e. the first row of the moment measurement matrix)
corresponds to computing the zeroth moment (i.e. the density).

# Positional arguments:
* `lambda_L1_arr`: array of L1 regularization values to iterate over
* `w`: weighting function values in a vector of length `n`
* `target_KL`: target distribution for K-L divergence, if equal to `ones(n)` then classical EQMom is recovered
* `Δv`: quadrature weights in a vector of length `n`
* `mmms_unrolled_concatenated`: moment measurement matrix (of size `m x n`) for reference moments (that act as constraints)
* `mmms_unrolled_concatenated_test`: moment measurement matrix (of size `m_test x n`) for test moments (that are predicted)
* `ref_moms`: vector of length `m` of reference moment values (that act as constraints)
* `ref_moms_test`: vector of length `m_test` of test moment values (that are predicted)
* `n_v`: number of velocity nodes in each direction

# Keyword arguments:
* `threshold`: threshold for sparsity, values smaller than this value will be set to 0
* `tol`: termination criteria based on gradient norm
* `maxiter`: maximum number of Newton iterations
* `mu` : regularization added to Hessian if it's singular / ill-conditioned
* `backtrack_rho` : step length shrink factor (0<rho<1)
* `backtrack_c` : Armijo parameter
* `find_y0`: if `true`, compute an initial guess for the dual solution
* `verbose`: produce printed output if `verbose > 0`
* `warmup`: if `true`, run a quick iteration of the solver to force compilation

# Returns
* `solutions`: array of shape `(n_v, n_v, n_v, n_lambda_vals)` containing the VDFs for each value in `lambda_L1_arr`
* `constraint_predictions`: array of shape `(m, n_lambda_vals)` containing the predicted reference/constraint moments for each value in `lambda_L1_arr`
* `test_predictions`: array of shape `(m_test, n_lambda_vals)` containing the predicted test moments for each value in `lambda_L1_arr`
* `sparsity`: array of shape `(n_lambda_vals)` containing the number of zero entries of the sparseified (see `threshold`) VDF for each value in `lambda_L1_arr`
"""
function solve_iterate_over_L1_values(lambda_L1_arr,
                                      w, target_KL, Δv,
                                      mmms_unrolled_concatenated, mmms_unrolled_concatenated_test,
                                      ref_moms, ref_moms_test, n_v; threshold=1e-6,
                                      tol=1e-9, maxiter=100,
                                      mu=1e-12, backtrack_rho=0.5, backtrack_c=1e-4, find_y0=true,
                                      verbose=2, warmup=true)
    
    m_test = size(mmms_unrolled_concatenated_test)[1]  # these are used for testing prediction quality
    m = size(mmms_unrolled_concatenated)[1]  # these are the constraints
    n = size(mmms_unrolled_concatenated)[2]

    A = ones((m, n))
    rnorm_A = ones(m)
    # setup stuff for Newton solver

    alpha = zeros(n)
    gsol = zeros(n)
    uvec = zeros(n)
    d_w = zeros(n)
    inv_dw = zeros(n)
    ref_moms_copy = zeros(m)

    y0 = zeros(m)
    y_new = zeros(m)
    mvec = zeros(m)

    grad = zeros(m)
    H = zeros((m, m))
    Hreg = zeros((m, m))
    p = zeros(m)
    F_ut = UpperTriangular(H)
    F_ch = Cholesky(F_ut)

    for i in 1:m
        ref_moms_copy[i] = ref_moms[i]
    end

    info::Dict{Symbol, Float64} = Dict(:iterations => 0.0, :converged => 0.0, :gradnorm=>NaN)

    reset_timer!()

    n_lambda_vals = length(lambda_L1_arr)
    solutions = zeros((n_v, n_v, n_v, n_lambda_vals))
    constraint = zeros((m, n_lambda_vals))
    test_predictions = zeros((m_test, n_lambda_vals))
    sparsity = zeros(n_lambda_vals)

    if warmup
        # run for just a few iterations to JIT compile the code
        @timeit "full solve: warmup" full_solve_with_init!(gsol, target_KL,
                                                           A, rnorm_A, ref_moms, mvec, mmms_unrolled_concatenated,
                                                           alpha, uvec, d_w, inv_dw, y0, y_new, grad, H, Hreg,
                                                           Δv, w, 0.0, n, m, F_ch, p, info;
                                                           tol=1e-5, maxiter=3,
                                                           mu=1e-12, backtrack_rho=0.5, backtrack_c=1e-4, find_y0=true)
    end

    for (i, λ) in enumerate(lambda_L1_arr)
        if verbose > 0
            println("\nSolving for λ = $λ ($i/$(length(lambda_L1_arr)))")
        end
        @timeit "full solve" full_solve_with_init!(gsol, target_KL,
                                                   A, rnorm_A, ref_moms, mvec, mmms_unrolled_concatenated,
                                                   alpha, uvec, d_w, inv_dw, y0, y_new, grad, H, Hreg,
                                                   Δv, w, λ, n, m, F_ch, p, info;
                                                   tol=tol, maxiter=maxiter,
                                                   mu=mu, backtrack_rho=backtrack_rho,
                                                   backtrack_c=backtrack_c, find_y0=find_y0)

        gsparse = copy(gsol)
        gsparse[gsparse .< threshold] .= 0.0

        # assume that the first moment is the mass
        gsparse = gsparse .* ref_moms[1] / (dot(mmms_unrolled_concatenated[1,:], (gsparse .* w)))

        solutions[:,:,:,i] .= rollup(copy(gsol), n_v, n_v, n_v)
        constraint_moments_predicted = mmms_unrolled_concatenated * (gsparse .* w)
        next_moments_predicted = mmms_unrolled_concatenated_test * (gsparse .* w)

        constraint[:,i] .= constraint_moments_predicted
        test_predictions[:,i] .= next_moments_predicted

        if verbose > 0
            println("-----------------")
            println("lambda = $(λ)")
            println(info)
            println("Min post-sparsity: $(minimum(gsparse[gsparse .> 0.0]))")
            println("Max absolute error in constraints: ", maximum(abs.(constraint_moments_predicted .- ref_moms)))
            println("Max absolute error in next moments: ", maximum(abs.(next_moments_predicted .- ref_moms_test)))
        end

        sparsity[i] = sum(gsol .< threshold)
    end
    
    if verbose > 0
        print_timer()
    end
    return solutions, constraint_moments_predicted, next_moments_predicted, sparsity
end
end