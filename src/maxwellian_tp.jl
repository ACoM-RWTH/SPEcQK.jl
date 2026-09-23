@muladd begin

"""
    MaxwellianTP{N_vx, N_vy, N_vz}

Separable (tensor-product) representation of a discrete Maxwellian on a
`Grid3D`: `M(v_ijk) = C * gx[i] * gy[j] * gz[k]`, with per-axis Gaussian
factors `gx[i] = exp(-(vx_i - ux)² / T)` (same for y/z) and the overall scale
`C` chosen so the discrete density matches the requested one. Any raw moment
of the Maxwellian then factorizes into per-axis 1D sums, dropping the cost
from `O(n_v³)` to `O(n_v)` (see `axis_moment_table!` / `separable_moments!`).
"""
mutable struct MaxwellianTP{N_vx, N_vy, N_vz}
    gx::MVector{N_vx, Float64}
    gy::MVector{N_vy, Float64}
    gz::MVector{N_vz, Float64}
    C::Float64
end

MaxwellianTP(grid::Grid3D{N_vx, N_vy, N_vz}) where {N_vx, N_vy, N_vz} =
    MaxwellianTP{N_vx, N_vy, N_vz}(zeros(MVector{N_vx, Float64}),
                                   zeros(MVector{N_vy, Float64}),
                                   zeros(MVector{N_vz, Float64}), 0.0)

"""
    maxwell_boltzmann_axes!(mb::MaxwellianTP, grid, ndens, vx0, vy0, vz0, T)

Fill the per-axis Gaussian factors of `mb` for a Maxwellian with streaming
velocity `(vx0, vy0, vz0)` and temperature `T`, and set `mb.C` so the discrete
density (with the grid quadrature weights) equals `ndens`. Costs `3 n_v` exps.
"""
function maxwell_boltzmann_axes!(mb::MaxwellianTP{N_vx, N_vy, N_vz}, grid::Grid3D{N_vx, N_vy, N_vz},
                                 ndens, vx0, vy0, vz0, T) where {N_vx, N_vy, N_vz}
    inv_T = 1.0 / T

    Sx0 = 0.0
    @inbounds for i in 1:N_vx
        d = grid.vx[i] - vx0
        g = exp(-d * d * inv_T)
        mb.gx[i] = g
        Sx0 += g * grid.Δvx[i]
    end

    Sy0 = 0.0
    @inbounds for j in 1:N_vy
        d = grid.vy[j] - vy0
        g = exp(-d * d * inv_T)
        mb.gy[j] = g
        Sy0 += g * grid.Δvy[j]
    end

    Sz0 = 0.0
    @inbounds for k in 1:N_vz
        d = grid.vz[k] - vz0
        g = exp(-d * d * inv_T)
        mb.gz[k] = g
        Sz0 += g * grid.Δvz[k]
    end

    mb.C = ndens / (Sx0 * Sy0 * Sz0)
    return mb
end

"""
    axis_moment_table!(S, v, Δv, g, A)

Fill the per-axis raw-moment table `S[a+1] = Σ_i v_i^a g_i Δv_i` for
`a = 0..A` in a single sweep (running power).
"""
function axis_moment_table!(S::AbstractVector, v, Δv, g, A::Int)
    @inbounds for a in 1:(A + 1)
        S[a] = 0.0
    end
    @inbounds for i in eachindex(v)
        p = g[i] * Δv[i]
        S[1] += p
        vi = v[i]
        for a in 2:(A + 1)
            p *= vi
            S[a] += p
        end
    end
    return S
end

"""
    axis_sq_moment_table!(Q, v, Δv, g, A)

Fill the squared-weight table `Q[a+1] = Σ_i v_i^{2a} (g_i Δv_i)²` for
`a = 0..A`. Used for the row norms of the constraint matrix
`A[i,j] = mmm[i,j] * w[j]`: `‖row (a,b,c)‖² = C² Qx[a+1] Qy[b+1] Qz[c+1]`.
"""
function axis_sq_moment_table!(Q::AbstractVector, v, Δv, g, A::Int)
    @inbounds for a in 1:(A + 1)
        Q[a] = 0.0
    end
    @inbounds for i in eachindex(v)
        p = g[i] * Δv[i]
        p *= p
        Q[1] += p
        vsq = v[i] * v[i]
        for a in 2:(A + 1)
            p *= vsq
            Q[a] += p
        end
    end
    return Q
end

"""
    separable_moments!(moms, mb, Sx, Sy, Sz, moment_powers)

Raw moments of the separable Maxwellian from the per-axis tables:
`moms[j] = C * Sx[a_j+1] * Sy[b_j+1] * Sz[c_j+1]` for each power
`(a_j, b_j, c_j)` in `moment_powers`. Replaces `mmm * w` at `O(m)` cost.
"""
function separable_moments!(moms::AbstractVector, mb::MaxwellianTP, Sx, Sy, Sz, moment_powers)
    C = mb.C
    @inbounds for j in eachindex(moment_powers)
        a, b, c = moment_powers[j]
        moms[j] = C * Sx[a + 1] * Sy[b + 1] * Sz[c + 1]
    end
    return moms
end

"""
    separable_moments!(moms, col, mb, Sx, Sy, Sz, moment_powers)

Column variant: writes into `moms[j, col]` (matches the layout of
`compute_moments_from_vdf!`).
"""
function separable_moments!(moms::AbstractMatrix, col::Integer, mb::MaxwellianTP, Sx, Sy, Sz, moment_powers)
    C = mb.C
    @inbounds for j in eachindex(moment_powers)
        a, b, c = moment_powers[j]
        moms[j, col] = C * Sx[a + 1] * Sy[b + 1] * Sz[c + 1]
    end
    return moms
end

"""
    separable_row_norms!(rnorm, mb, Qx, Qy, Qz, moment_powers)

Row norms of the unnormalized constraint matrix `A[i,j] = mmm[i,j] * w[j]`
for `w` the separable Maxwellian `mb`, from the squared-weight tables
(`axis_sq_moment_table!`): `rnorm[i] = C * sqrt(Qx[a+1] Qy[b+1] Qz[c+1])`.
"""
function separable_row_norms!(rnorm::AbstractVector, mb::MaxwellianTP, Qx, Qy, Qz, moment_powers)
    C = mb.C
    @inbounds for j in eachindex(moment_powers)
        a, b, c = moment_powers[j]
        rnorm[j] = C * sqrt(Qx[a + 1] * Qy[b + 1] * Qz[c + 1])
    end
    return rnorm
end

"""
    densify!(vdf::VDF3D, mb::MaxwellianTP)

Outer-product fill of the dense VDF from the separable factors:
`vdf.w[i,j,k] = C * gx[i] * gy[j] * gz[k]`. One multiply sweep, no exps.
"""
function densify!(vdf::VDF3D{N_vx, N_vy, N_vz}, mb::MaxwellianTP{N_vx, N_vy, N_vz}) where {N_vx, N_vy, N_vz}
    @inbounds for k in 1:N_vz
        cgz = mb.C * mb.gz[k]
        for j in 1:N_vy
            gyz = cgz * mb.gy[j]
            for i in 1:N_vx
                vdf.w[i, j, k] = mb.gx[i] * gyz
            end
        end
    end
    return nothing
end

"""
    axis_shifted_raw_sums(v, Δv, u, inv_T)

Per-axis raw sums `s_a = Σ_i v_i^a exp(-(v_i-u)² inv_T) Δv_i` for `a = 0..4`
in one sweep. Returns the tuple `(s0, s1, s2, s3, s4)`. Workhorse of the
tensor-product `find_T_and_v!` / `find_T!` Newton iterations.
"""
@inline function axis_shifted_raw_sums(v::SVector{N, Float64}, Δv::SVector{N, Float64}, u, inv_T) where {N}
    s0 = 0.0; s1 = 0.0; s2 = 0.0; s3 = 0.0; s4 = 0.0
    @inbounds for i in 1:N
        vi = v[i]
        d = vi - u
        g = exp(-d * d * inv_T) * Δv[i]
        s0 += g
        p = vi * g; s1 += p
        p *= vi;    s2 += p
        p *= vi;    s3 += p
        p *= vi;    s4 += p
    end
    return s0, s1, s2, s3, s4
end

end
