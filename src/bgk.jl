"""
BGK (Bhatnagar-Gross-Krook) collision operator utilities.
"""

"""
    tau_BGK(n, T, omega; mu_sref=1.0)

BGK collision (relaxation) time scale from the standard relation `τ = μ/p`,
with a VHS (variable-hard-sphere) viscosity law `μ(T) = mu_sref * T^omega` and
the ideal-gas pressure `p = n*T`.

    τ = μ / p = mu_sref * T^omega / (n*T) = mu_sref * T^(omega-1) / n

so that the resulting BGK viscosity reproduces `μ = τ*n*T = mu_sref * T^omega`.
This is the standard single-relaxation BGK model (Prandtl number 1).

Limits: `omega = 1/2` (hard spheres) gives `τ ∝ T^(-1/2)/n`; `omega = 1`
(Maxwell molecules) gives `τ ∝ 1/n`, independent of temperature.

!!! note "Which temperature"
    `T` is the **code temperature** of `docs/src/scaling.md`, i.e. the same `T`
    that parameterizes [`maxwell_boltzmann!`](@ref)'s `exp(-|v-u|²/T)`, for which
    `p = nT` holds and the per-direction velocity variance is `T/2`. In terms of
    raw moments that is

        T = (2/3) * ( (M200 + M020 + M002)/n - |u|² )

    **not** `(<v²> - |u|²)/3`. Callers should
    pass the matched Maxwellian's own temperature (`mb_T` from
    [`find_MB_solution_T_and_v!`](@ref)) rather than recomputing it: that is this
    same quantity, and it is additionally consistent with the truncated velocity
    grid. No further rescaling is applied here — `mu_sref` and `T` go in as-is.

# Positional arguments:
* `n`: number density
* `T`: code temperature (see the note above)
* `omega`: VHS viscosity-temperature exponent (1/2 = hard sphere, 1 = Maxwell)

# Keyword arguments:
* `mu_sref`: reference viscosity (= viscosity at `T = 1`); sets the collision
  scale / Knudsen number. Larger `mu_sref` -> longer `τ` -> more rarefied.

# Returns:
* `tau`: collision relaxation time
"""
function tau_BGK(n, T, omega; mu_sref=1.0)
    return mu_sref * T^(omega - 1.0) / n
end

"""
    Knudsen_number(n, T, omega, L, mu_sref)

Compute Knudsen number based on a characteristic length scale `L` and the
collision relaxation time `tau_BGK`.

`Kn = λ/L` with `λ ≈ C_ω * mu_sref * T^(ω-1/2) / n`, the mean free path being the
distance travelled in one collision time at the code-unit thermal speed `√T`
(see the Knudsen-number section of `docs/src/scaling.md`). `C_ω` is the standard
VHS constant — Bird eqn. (4.65) / Boyd & Schwartzentruber (D.21) — so this is the
true `λ/L`, not the `O(1)`-approximate `mu_sref * T^(ω-1/2)/n`.

`T` is the same **code temperature** as for [`tau_BGK`](@ref); see the note
there. `mu_sref` and `T` are used as-is, with no further rescaling.
"""
function Knudsen_number(n, T, omega, L, mu_sref)
    C_omega = ((5.0 - 2.0*omega) * (7.0 - 2.0*omega)) / (5.0 * (1.0 + omega) * sqrt(2.0 * pi))

    lambda = C_omega * mu_sref * T^(omega - 0.5) / n
    return lambda / L
end