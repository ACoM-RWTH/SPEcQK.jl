# dual_solver.jl
# Self-rolled Newton solver for the dual of
#   min_g sum_i d_i*(w_i g_i log(w_i g_i) + lambda * g_i)
#   s.t. A*g = b, g >= 0
# Dual (maximize) reduced to unconstrained y in R^m; we solve the minimization
# of f(y) = -phi(y) using Newton's method with analytic gradient and Hessian.

using LinearAlgebra
using TimerOutputs

@muladd begin
"""
    compute_constraint_matrix!(A, mmms_unrolled_concatenated, w, n, m)

Compute the matrix of constraints based on the moment measurement matrix and weighting function.

# Positional arguments:
* `A`: The constraint matrix to be computed of size `n x m`
* `mmms_unrolled_concatenated`: unrolled moment measurement matrix of size `m x n`
* `w`: the weighting function of size `n`
* `n`: the number of quadrature points
* `m`: the number of moment constraints
"""
function compute_constraint_matrix!(A, mmms_unrolled_concatenated, w, n, m)
    @inbounds for j in 1:n
        @inbounds for i in 1:m
            A[i,j] = mmms_unrolled_concatenated[i,j] * w[j]
        end
    end
end

"""
    compute_d_w_factors!(d_w::AbstractVector, inv_dw::AbstractVector, Δv, w, n)

Compute the weighting factors (quadrature weight times weighting function) and their inverses.

# Positional arguments:
* `d_w`: vector to store weighting factors
* `inv_dw`: vector to store inverses of weighting factors
* `Δv`: vector of quadrature weights
* `w`: vector of weighting function values
* `n`: number of quadrature points
"""
function compute_d_w_factors!(d_w::AbstractVector, inv_dw::AbstractVector,
                              Δv, w, n)
    @inbounds for i in 1:n
        d_w[i] = Δv[i] * w[i]
        inv_dw[i] = 1.0 / d_w[i]
    end
end

"""
    dual_from_primal_guess!(A, alpha, inv_dw)

Compute the dual solution from a primal solution that is constant and equal to 1.

# Positional arguments:
* `A` - the constraints matrix (size m x n)
* `alpha` - see definition of dual problem for definition of alpha (size n)
* `inv_dw` - see definition of dual problem for definition of alpha (size n)

# Returns:
* `y0` - the dual solution guess (of size m)
"""
function dual_from_primal_guess!(A, alpha, inv_dw)
    # TODO: use pre-allocated values!
    # we have
    # gsol[i] = 0.0
    # for j in 1:m
    #     gsol[i] += A[j,i] * y[j]
    # end
    # gsol[i] *= inv_dw[i]
    # gsol[i] = alpha[i] * exp(clamp(gsol[i], -EXP_CLAMP, EXP_CLAMP))

    # or succinctly: g_i = α_i * exp(inv_dw_i * (A^T y)_i)
    # we want g_i === 1
    # (A^T y)_i = ln(1/α_i) / inv_dw_i = -ln(α_i) / inv_dw_i
    # so we solve A A^T y = -A ln(α) / inv_dw

    M = A * A'
    rhs = -A * (log.(alpha) ./ inv_dw)
    return M \ rhs
end

"""
    compute_alpha!(alpha::AbstractVector, w::AbstractVector,
                   inv_dw::AbstractVector, ξ::AbstractVector, λ, n)

Compute the vector `alpha` of length `n` using the given parameters.
See definition of dual problem for definition of quantities involved.

# Positional arguments:
* `alpha` - results will be written here
* `w` - weighting function
* `inv_dw` - vectors of inverses of `Δv[i] * w[i]`
* `ξ` - target distribution in K-L divergence (vector of ones for classical SEQMom)
* `λ` - strength of L1 regularization
* `n` - length of vectors
"""
function compute_alpha!(alpha::AbstractVector, w::AbstractVector, inv_dw::AbstractVector,
                        ξ::AbstractVector, λ, n)
    @inbounds for i in 1:n
        alpha[i] = ξ[i] * exp(-1 - λ * inv_dw[i]) / w[i]
    end
end


# Evaluate f(y) = -phi(y), grad f, and Hessian H = -∇^2 φ = A * D * A'
# where D = diag( e ./ (d.^2 .* w.^2) ) with e = alpha .* exp( (A' * y) ./ (w.*d) )

"""
    f_grad_hess!(grad::AbstractVector, H::AbstractMatrix, A::AbstractMatrix,
                 mvec::AbstractVector,
                 alpha::AbstractVector, y::AbstractVector,
                 d_w, inv_dw, uvec,
                 n, m)

Evaluate f(y) = -phi(y), grad f, and Hessian H = -∇^2 φ = A * D * A'
where D = diag( e ./ (d.^2 .* w.^2) ) with e = alpha .* exp( (A' * y) ./ (w.*d) )

KKT:
d/dg_i (Δv[i] * w_i * g_i log(g_i w_i) + lambda g_i) = (A^T y)_I
Δv[i] * w_i log(g_i w_i) + Δv[i] * w_i + lambda = (A^T y)_i
log(g_i w_i) = (A^T y)_i * inv_dw_i - lambda * inv_dw_i - 1
g_i = (1/w_i) * exp((A^T y)_i * inv_dw_i - lambda * inv_dw_i - 1)
y = 0, lambda = 0
g_i = exp()

phi = ... - sum_i d_i exp(A^t y * inv_dw_i - lambda * inv_dw_i - 1)
g = (1/w_i) * exp((A^T y)_i * inv_dw_i - lambda * inv_dw_i - 1)
alpha =  (1/w_i) * exp(- lambda * inv_dw_i - 1)
phi = sum_i (w_i d_i) g_i

# Positional arguments:
* `grad`: vector of length `m` where gradient will be stored
* `H`: matrix of size `m x m` where Hessian will be stored
* `A`: constraints matrix of size `m x n`
* `mvec`: moment vector of length `m`
* `alpha`: vector of length `n` (see problem description for definition)
* `y`: vector of length `m`, current dual solution
* `d_w`: vector of length `n` containing the quadrature weights times weighting function
* `inv_dw`: vector of length `n` containing inverses of elements of `d_w`
* `uvec`: vector of length `n` used for intermediate storage
* `n`: number of quadrature points
* `m`: number of moment measurements

# Returns:
* Value of target function
"""
function f_grad_hess!(grad::AbstractVector, H::AbstractMatrix, A::AbstractMatrix,
                      mvec::AbstractVector,
                      alpha::AbstractVector, y::AbstractVector,
                      d_w, inv_dw, uvec,
                      n, m)
    fval = 0.0

    fill!(uvec, 0.0)
    fill!(H, 0.0)

    @inbounds for i in 1:n
        s = 0.0
        @simd for j in 1:m
            s += A[j,i] * y[j]
        end

        s *= inv_dw[i]
        s = clamp(s, -EXP_CLAMP, EXP_CLAMP)
        uvec[i] = alpha[i] * exp(s)
        fval += uvec[i] * d_w[i]
    end

    fval -= dot(y, mvec)
    
    # grad .= -mvec
    @inbounds for j in 1:m
        grad[j] = -mvec[j]
    end
    @inbounds for i in 1:n
        @simd for j in 1:m
            grad[j] += A[j,i] * uvec[i]
        end
    end
    
    # Hessian: H = A * Diag( e ./ (d.^2 .* w.^2) ) * A'
    # Dvec = uvec .* (inv_dw.^2) # length n

    @inbounds for k in 1:n
        wk = uvec[k] * inv_dw[k] # inv_dw_sq[k]
        @inbounds @simd for j in 1:m
            Ajk_wk = A[j,k] * wk
            @inbounds for i in 1:j
                H[i,j] += A[i,k] * Ajk_wk
            end
        end
    end

    @inbounds for j in 1:m
        @inbounds @simd for i in 1:j
            H[j,i] = H[i,j]
        end
    end

    return fval
end


"""
    f_grad_hess!(A::AbstractMatrix,
                 mvec::AbstractVector,
                 alpha::AbstractVector, y::AbstractVector,
                 d_w, inv_dw,
                 n, m)

Evaluate f(y) = -phi(y) only.

# Positional arguments:
* `A`: moment measurement matrix of size `m x n`
* `mvec`: moment vector of length `m`
* `alpha`: vector of length `n` (see problem description for definition)
* `y`: vector of length `m`, current dual solution
* `d_w`: vector of length `n` containing the quadrature weights times weighting function
* `inv_dw`: vector of length `n` containing inverses of elements of `d_w`
* `n`: number of quadrature points
* `m`: number of moment measurements

# Returns:
* Value of target function
"""
function f_only(A::AbstractMatrix, mvec::AbstractVector,
                alpha::AbstractVector, y::AbstractVector,
                d_w, inv_dw,
                n, m)
    # Inputs: A (m x n), b (m), w (n), d (n), alpha (n), y (m)
    fval = 0.0

    @inbounds for i in 1:n
        s = 0.0
        @simd for j in 1:m
            s += A[j,i] * y[j]
        end

        s *= inv_dw[i]
        s = alpha[i] * exp(clamp(s, -EXP_CLAMP, EXP_CLAMP))
        fval += s * d_w[i]
    end

    fval -= dot(y, mvec)

    return fval
end


"""
newton_dual!(gsol::AbstractVector,
             A::AbstractMatrix, mvec::AbstractVector, alpha::AbstractVector, 
             y::AbstractVector, y_new::AbstractVector,
             d_w::AbstractVector, inv_dw::AbstractVector,
             uvec::AbstractVector,
             grad::AbstractVector, H::AbstractMatrix, Hreg::AbstractMatrix,
             n, m, F_ch, p::AbstractVector, info; tol=1e-9, maxiter=100,
             mu=1e-12, backtrack_rho=0.5, backtrack_c=1e-4))

Solves the dual problem by minimizing f(y) = -phi(y) with Newton's method.
Returns (y, g, primal_obj, info)

# Positional arguments:
* `gsol`: vector of length `n` to store primal solution
* `A`: moment measurement matrix of size `m x n`
* `mvec`: moment vector of length `m` (constraints)
* `alpha`: vector of length `n` (see problem description for definition)
* `y`: vector of length `m` to store dual solution; initial solution guess should be given here
* `y_new`: vector of length `m` to store new dual solution (in Armijo search)
* `d_w`: vector of length `n` containing the quadrature weights times weighting function
* `inv_dw`: vector of length `n` containing inverses of elements of `d_w`
* `uvec`: vector of length `n` to store intermediate values
* `grad`: vector of length `m` where gradient will be stored
* `H`: matrix of size `m x m` where Hessian will be stored
* `Hreg`: matrix of size `m x m` where regularized Hessian will be stored
* `n`: number of quadrature points
* `m`: number of moment measurements
* `F_ch`: Cholesky factorization object (mutated)
* `p`: vector of length `m` used for backtracking line search
* `info`: dictionary to store information about solution (`:iterations`, `:converged`, `:gradnorm`)

# Keyword arguments:
* `tol`: termination criteria based on gradient norm
* `maxiter`: maximum number of Newton iterations
* `mu` : regularization added to Hessian if it's singular / ill-conditioned
* `backtrack_rho` : step length shrink factor (0<rho<1)
* `backtrack_c` : Armijo parameter
"""
function newton_dual!(gsol::AbstractVector,
                      A::AbstractMatrix, mvec::AbstractVector, alpha::AbstractVector, 
                      y::AbstractVector, y_new::AbstractVector,
                      d_w::AbstractVector, inv_dw::AbstractVector,
                      uvec::AbstractVector,
                      grad::AbstractVector, H::AbstractMatrix, Hreg::AbstractMatrix,
                      n, m, F_ch, p::AbstractVector, info; tol=1e-9, maxiter=100,
                      mu=1e-12, backtrack_rho=0.5, backtrack_c=1e-4)

    info[:iterations] = 0
    info[:converged] = 0
    info[:gradnorm] = NaN

    gn = 0.0
    k = 1

    @inbounds for k in 1:maxiter
        fval = f_grad_hess!(grad, H, A, mvec, alpha, y, d_w, inv_dw, uvec,
                            n, m)
        gn = norm(grad)
        info[:iterations] = convert(Float64, k)

        if gn < tol
            info[:converged] = 1.0
            break
        end
        
        # Ensure H is SPD: add mu*I (increase mu if necessary)
        # Try a Cholesky factorization with growing regularization until success
        tau = mu
        success = false
        max_mu_tries = 10

        for j in 1:m
            @simd for i in 1:m
                Hreg[i,j] = H[i,j]
            end
        end
        for _ in 1:max_mu_tries
            # form H_reg = H + tau*I
            @simd for i in 1:m
                Hreg[i,i] = H[i,i] + tau
            end
            
            F_ch = cholesky(Hreg; check=false)

            if issuccess(F_ch)
                success = true
                break
            end
            tau *= 10.0
        end
        if !success
            error("Hessian regularization failed to produce SPD matrix; try larger mu (currently $tau)")
        end

        # Solve Hreg * p = -grad  => Newton step p
        @simd for i in 1:m
            p[i] = grad[i]
        end
        # p .= grad

        ldiv!(F_ch, p)  # in-place solve
        p = -p

        # Backtracking line search (Armijo) on f(y); ensure y + t*p reduces f
        t = 1.0
        fy = fval
        # directional derivative: grad' * p
        # dirder = dot(grad, p)
        dirder = 0.0
        @simd for i in 1:m
            dirder += grad[i] * p[i]
        end
        if dirder >= 0
            # Not a descent direction (shouldn't happen for Newton on convex f), fall back to -grad direction
            # p = -grad
            p .= -grad
            @simd for i in 1:m
                p[i] = -grad[i]
            end

            dirder = -gn^2
            # dirder = -dot(grad, grad)
            # dirder = 0.0
            # @inbounds for i in 1:m
            #     dirder -= grad[i]^2
            # end
        end

        # Armijo loop
        max_ls = 40
        ls_iter = 0
        while ls_iter < max_ls
            y_new .= y .+ t * p

            fnew = f_only(A, mvec, alpha, y_new, d_w, inv_dw,
                          n, m)
            if fnew <= fy + backtrack_c * t * dirder
                # sufficient decrease
                y .= y_new
                break
            else
                t *= backtrack_rho
                ls_iter += 1
            end
        end
        if ls_iter >= max_ls
            # line search failed; accept small step in direction of p scaled
            y .+= t * p
        end
    end

    # Recover primal solution g
    @inbounds @simd for i in 1:n
        gsol[i] = 0.0
        for j in 1:m
            gsol[i] += A[j,i] * y[j]
        end
        gsol[i] *= inv_dw[i]
        gsol[i] = alpha[i] * exp(clamp(gsol[i], -EXP_CLAMP, EXP_CLAMP))
    end

    info[:gradnorm] = gn

    return
end

"""
    full_solve_with_init!(gsol, target_KL,
                          A, rnorm_A, ref_moms_constraint, mvec, mmms_unrolled_concatenated,
                          alpha, uvec, d_w, inv_dw, y0, y_new, grad, H, Hreg,
                          Δv, w, λ, n, m, F_ch, p, info;
                          tol=1e-9, maxiter=100,
                          mu=1e-12, backtrack_rho=0.5, backtrack_c=1e-4, find_y0=true)

Solve the SPEQMom problem: minimizing K-L divergence to a target vector + L1 regularization whilst
satisfying linear constraints. This function sets up the constraint matrix, initial solution, etc.

# Positional arguments
* `gsol`: vector of length `n` containing computed solution
* `target_KL`: vector of length `n` containing target distribution for K-L divergence. If vector of ones,
this recovers classical entropic quadrature.
* `A`: matrix of size `m x n` that will store constraints
* `rnorm_A`: vector of size `m` to store row-wise norms of `A`
* `ref_moms_constraint`: moment vector of length `m` of constraint values (remains unchanged)
* `mvec`: moment vector of length `m` to store scaled constraint values (is mutated)
* `mmms_unrolled_concatenated`: unrolled moment measurement matrix of size `m x n`
* `alpha`: vector of size `n` to store pre-computed values used in optimization routine
* `uvec`: vector of length `n` used for intermediate storage
* `d_w`: vector of length `n` containing the quadrature weights times weighting function
* `inv_dw`: vector of length `n` containing inverses of elements of `d_w`
* `y0`: vector of length `m` to store previous dual solution, also used to store computed guess (if `find_y0 == true`)
* `y_new`: vector of length `m` to store updated dual solution (in Armijo search)
* `grad`: vector of length `m` where gradient will be stored
* `H`: matrix of size `m x m` where Hessian will be stored
* `Hreg`: matrix of size `m x m` where regularized Hessian will be stored
* `Δv`: vector of quadrature weights
* `w`: vector of weighting function values
* `n`: number of quadrature points
* `m`: number of moment measurements
* `F_ch`: Cholesky factorization object (mutated)
* `p`: vector of length `m` used for backtracking line search
* `info`: dictionary to store information about solution (`:iterations`, `:converged`, `:gradnorm`)

# Keyword arguments:
* `tol`: termination criteria based on gradient norm
* `maxiter`: maximum number of Newton iterations
* `mu` : regularization added to Hessian if it's singular / ill-conditioned
* `backtrack_rho` : step length shrink factor (0<rho<1)
* `backtrack_c` : Armijo parameter
* `find_y0`: if `true`, compute an initial guess for the dual solution
"""
function full_solve_with_init!(gsol, target_KL,
                               A, rnorm_A, ref_moms_constraint, mvec, mmms_unrolled_concatenated,
                               alpha, uvec, d_w, inv_dw, y0, y_new, grad, H, Hreg,
                               Δv, w, λ, n, m, F_ch, p, info;
                               tol=1e-9, maxiter=100,
                               mu=1e-12, backtrack_rho=0.5, backtrack_c=1e-4, find_y0=true)

    compute_constraint_matrix!(A, mmms_unrolled_concatenated, w, n, m)

    fill!(rnorm_A, 0.0)

    for j in 1:n
        for i in 1:m
            Asq = A[i,j]^2
            rnorm_A[i] += Asq
        end
    end

    for i in 1:m
        rnorm_A[i] = sqrt(rnorm_A[i])
    end

    for j in 1:n
        for i in 1:m
            A[i,j] /= (rnorm_A[i])
        end
    end
    for i in 1:m
        mvec[i] = ref_moms_constraint[i] / rnorm_A[i]
    end

    compute_d_w_factors!(d_w, inv_dw, Δv, w, n)

    if find_y0
        compute_alpha!(alpha, w, inv_dw, target_KL, 0.0, n)
        y0 = dual_from_primal_guess!(A, alpha, inv_dw)
    end
    compute_alpha!(alpha, w, inv_dw, target_KL, λ, n)

    newton_dual!(gsol,
                 A, mvec, alpha, 
                 y0, y_new,
                 d_w, inv_dw,
                 uvec,
                 grad, H, Hreg,
                 n, m, F_ch, p, info; tol=tol, maxiter=maxiter,
                 mu=mu, backtrack_rho=backtrack_rho, backtrack_c=backtrack_c)
end
end