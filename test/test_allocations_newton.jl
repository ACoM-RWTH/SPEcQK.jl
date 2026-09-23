# Allocation regression tests for the dual Newton solvers and the warm-start
# guess. These kernels run once per cell per time step, so any per-call heap
# allocation is a real hot-loop cost (a `t * p` temporary in the line search
# and an `lu!` pivot vector in the guess used to allocate here).
#
# Each solver is exercised on a small, deterministically-constructed *feasible*
# problem: pick a dual `y_true`, push it through the KKT primal map to get `g`,
# and set the right-hand side to `A*g`. Starting the measured solve from a cold
# `y0 = 0` then forces several real Newton iterations, hitting the line-search
# and Cholesky paths where the temporaries lived.

using SPEcQK
using SPEcQK: newton_dual!, dual_from_primal_guess!, EXP_CLAMP
using Test
using LinearAlgebra
using StableRNGs

# Moment matrix with unit-norm rows, mirroring the solver's internal row scaling.
function _random_normalized_A(m, n, rng)
    A = randn(rng, m, n)
    for i in 1:m
        A[i, :] ./= norm(@view A[i, :])
    end
    return A
end

# Classic solver primal map: g_i = α_i · exp(inv_dw_i · (Aᵀy)_i), target_KL = 1.
# `d_w = 1` keeps the exponential dual well conditioned so the cold start from
# `y0 = 0` converges reliably (in production this solve is warm-started).
function _setup_classic(n, m, λ, rng)
    A = _random_normalized_A(m, n, rng)
    d_w = fill(1.0, n)                  # Δv .* w with w = 1
    inv_dw = 1.0 ./ d_w
    alpha = @. exp(-1.0 - λ * inv_dw)   # / w, w = 1
    y_true = 0.05 .* randn(rng, m)
    g = zeros(n)
    for i in 1:n
        s = clamp(dot(@view(A[:, i]), y_true) * inv_dw[i], -EXP_CLAMP, EXP_CLAMP)
        g[i] = alpha[i] * exp(s)
    end
    mvec = A * g
    return A, mvec, alpha, d_w, inv_dw
end

# Micro-macro primal map: g_i = exp(inv_dw_i · soft_λ((Aᵀy)_i)) - 1.
function _setup_mm(n, m, λ, rng)
    A = _random_normalized_A(m, n, rng)
    d_w = fill(1.0, n)                  # Δv .* w with w = 1 (well conditioned)
    inv_dw = 1.0 ./ d_w
    y_true = 0.1 .* randn(rng, m)
    g = zeros(n)
    for i in 1:n
        a = dot(@view(A[:, i]), y_true)
        aa = abs(a)
        if aa >= λ
            s = clamp(copysign(aa - λ, a) * inv_dw[i], -EXP_CLAMP, EXP_CLAMP)
            g[i] = exp(s) - 1.0
        end
    end
    mvec = A * g
    return A, mvec, d_w, inv_dw
end

# --- function barriers -------------------------------------------------------
# Buffers are passed as arguments so `@allocated` reflects only the kernel's own
# allocations, not boxing from the enclosing (global) test scope. Each helper
# does one warmup solve to compile, then measures a second cold solve.

function _alloc_newton_dual!(gsol, A, mvec, alpha, y0, y_new, d_w, inv_dw, uvec,
                             grad, H, Hreg, n, m, F_ch, p, info)
    fill!(y0, 0.0)
    newton_dual!(gsol, A, mvec, alpha, y0, y_new, d_w, inv_dw, uvec,
                 grad, H, Hreg, n, m, F_ch, p, info; tol=1e-10, maxiter=200)
    fill!(y0, 0.0)
    bytes = @allocated newton_dual!(gsol, A, mvec, alpha, y0, y_new, d_w, inv_dw, uvec,
                                    grad, H, Hreg, n, m, F_ch, p, info; tol=1e-10, maxiter=200)
    return bytes, info[:converged], info[:iterations]
end

function _alloc_guess!(y0, A, alpha, inv_dw, M_scratch, tmp_n, n, m)
    dual_from_primal_guess!(y0, A, alpha, inv_dw, M_scratch, tmp_n, n, m)
    return @allocated dual_from_primal_guess!(y0, A, alpha, inv_dw, M_scratch, tmp_n, n, m)
end

@testset "Newton solver allocations" begin
    n = 800
    m = 10

    @testset "newton_dual! (classic) is allocation-free" begin
        rng = StableRNG(1)
        λ = 0.0
        A, mvec, alpha, d_w, inv_dw = _setup_classic(n, m, λ, rng)

        gsol = zeros(n); y0 = zeros(m); y_new = zeros(m); uvec = zeros(n)
        grad = zeros(m); H = zeros(m, m); Hreg = zeros(m, m); p = zeros(m)
        F_ch = Cholesky(UpperTriangular(H))
        info = Dict(:iterations => 0.0, :converged => 0.0, :gradnorm => NaN)

        bytes, converged, iters = _alloc_newton_dual!(gsol, A, mvec, alpha, y0, y_new,
                                                      d_w, inv_dw, uvec, grad, H, Hreg,
                                                      n, m, F_ch, p, info)

        @test converged == 1.0   # sanity: we measured a real, converged solve
        @test iters > 1          # sanity: line-search / Cholesky path was exercised
        @test bytes == 0
    end

    @testset "dual_from_primal_guess! is allocation-free" begin
        rng = StableRNG(3)
        λ = 0.0
        A, mvec, alpha, d_w, inv_dw = _setup_classic(n, m, λ, rng)

        y0 = zeros(m)
        M_scratch = zeros(m, m)   # scratch for A*A' and its factorization
        tmp_n = zeros(n)

        bytes = _alloc_guess!(y0, A, alpha, inv_dw, M_scratch, tmp_n, n, m)

        @test all(isfinite, y0)   # sanity: produced a finite guess
        @test bytes == 0
    end
end
