
"""
    A wrapper that looks at sparsity, computes errors, etc.

        # assume that the first moment in mmms_unrolled_concatenated is the mass 
"""
function solve_iterate_over_L1_values(lambda_L1_arr,
                                      w, g0, Δv, y0, y_new,
                                      A,
                                      mmms_unrolled_concatenated, mmms_unrolled_concatenated_test, mmms_unrolled_concatenated_full,
                                      ref_moms, ref_moms_test, ref_moms_full,
                                      result_matrix, predicted_moment_values, sparsity, errors, threshold;
                                      tol=1e-9, maxiter=100,
                                      mu=1e-12, backtrack_rho=0.5, backtrack_c=1e-4, find_y0=true,
                                      verbose=2, use_KL=false)
    
    m_test = size(mmms_unrolled_concatenated_test)[1]  # these are used for testing prediction quality
    m = size(mmms_unrolled_concatenated)[1]  # these are the constraints
    n = size(mmms_unrolled_concatenated)[2]

    rnorm_A = ones(m)
    # setup stuff for Newton solver

    alpha = zeros(n)
    gsol = zeros(n)
    uvec = zeros(n)
    d_w = zeros(n)
    inv_dw = zeros(n)
    ref_moms_copy = zeros(m)

    grad = zeros(m)
    H = zeros((m, m))
    Hreg = zeros((m, m))
    p = zeros(m)
    F_ut = UpperTriangular(H)
    F_ch = Cholesky(F_ut)

    for i in 1:m
        ref_moms_copy[i] = ref_moms[i]
    end

    ξ = ones(n)  # corresponds to Maximum Entropy

    if use_KL
        ξ = w .* g0  # corresponds to K-L
    end

    info::Dict{Symbol, Float64} = Dict(:iterations => 0.0, :converged => 0.0, :gradnorm=>NaN)

    reset_timer!()

    n_lambda_vals = length(lambda_L1_arr)
    solutions = zeros((n_v, n_v, n_v, n_lambda_vals))
    constraint = zeros((m, n_lambda_vals))
    test_predictions = zeros((m_test, n_lambda_vals))
    sparsity = zeros(n_lambda_vals)

    for (i, λ) in enumerate(lambda_L1_arr)
        # @timeit "full solve" full_solve_with_init!(????)

        gsparse = copy(gsol)
        gsparse[gsparse .< threshold] .= 0.0

        # assume that the first moment is the mass
        gsparse = gsparse .* ref_moms[1] / (dot(mmms_unrolled_concatenated[1,:], (gsparse .* w)))

        solutions[:,:,:,i] .= rollup(copy(g), n_v, n_v, n_v)
        constraint_moments_predicted = mmms_unrolled_concatenated * (gsparse .* w)
        next_moments_predicted = mmms_unrolled_concatenated_test * (gsparse .* w)

        constraint[:,i] .= constraint_moments_predicted
        test_predictions[:,i] .= next_moments_predicted

        if verbose > 0
            println("-----------------")
            println("lambda = $(lambda_L1)")
            println(info)
            println("Min post-sparsity: $(minimum(gsparse[gsparse .> 0.0]))")
            println("Max absolute error in constraints: ", maximum(abs.(constraint_moments_predicted .- ref_moms)))
            println("Max absolute error in next moments: ", maximum(abs.(next_moments_predicted .- ref_moms_test)))
        end

        sparsity[i] = sum(gsol .< threshold)
    end

    print_timer()
    return solutions, constraint_moments_predicted, next_moments_predicted, sparsity
end