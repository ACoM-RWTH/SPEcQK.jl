using SpecialFunctions

@muladd begin

"""
    scale_vdf!(vdf::VDF3D{N_vx, N_vy, N_vz}, grid::Grid3D{N_vx,N_vy,N_vz}, n0) where {N_vx, N_vy, N_vz}

Scale the velocity distribution function `vdf` by the density `n0`.
"""
@inline function scale_vdf!(vdf::VDF3D{N_vx, N_vy, N_vz}, grid::Grid3D{N_vx,N_vy,N_vz}, n0) where {N_vx, N_vy, N_vz}
    # scale so that density is 1, n0 is the computed density
    @inbounds for k in 1:grid.n_vz
        for j in 1:grid.n_vy
            for i in 1:grid.n_vx
                vdf.w[i,j,k] = vdf.w[i,j,k] / n0
            end
        end
    end
end

"""
    maxwell_boltzmann!(vdf::VDF3D{N_vx, N_vy, N_vz}, grid::Grid3D{N_vx,N_vy,N_vz}, vx0, vy0, vz0, T::Number) where {N_vx, N_vy, N_vz}

Compute the Maxwell-Boltzmann distribution on a given velocity grid `grid` with a streaming velocity
`[vx0, vy0, vz0]` and temperature `T`.
The distribution is stored in `vdf.w` and is normalized to 1.
**Note**: due to discretization and grid cut-off, actual velocity and temperature values might be different
than those passed to the function.
"""
function maxwell_boltzmann!(vdf::VDF3D{N_vx, N_vy, N_vz}, grid::Grid3D{N_vx,N_vy,N_vz}, vx0, vy0, vz0, T::Number) where {N_vx, N_vy, N_vz}
    n = 0.0
    inv_T = 1.0/T
    @inbounds for k in 1:N_vz
        pow_z = (grid.vz[k]-vz0)^2
        for j in 1:N_vy
            pow_y = (grid.vy[j]-vy0)^2
            for i in 1:N_vx
                vdf.w[i,j,k] = exp(-((grid.vx[i]-vx0)^2+pow_y+pow_z)*inv_T)
                n += vdf.w[i,j,k] * grid.Δvx[i] * grid.Δvy[j] * grid.Δvz[k]
            end
        end
    end

    scale_vdf!(vdf, grid, n)  # scale so that density is 1
end

"""
    maxwell_boltzmann!(vdf::VDF3D{N_vx, N_vy, N_vz}, grid::Grid3D{N_vx,N_vy,N_vz}, vx0, vy0, vz0, T::Number) where {N_vx, N_vy, N_vz}

Compute the Maxwell-Boltzmann distribution on a given velocity grid `grid` with a zero streaming velocity
and temperature `T`.
The distribution is stored in `vdf.w` and is normalized to 1.
**Note**: due to discretization and grid cut-off, actual velocity and temperature values might be different
than those passed to the function.
"""
function maxwell_boltzmann!(vdf::VDF3D{N_vx, N_vy, N_vz}, grid::Grid3D{N_vx,N_vy,N_vz}, T::Number) where {N_vx, N_vy, N_vz}
    maxwell_boltzmann!(vdf, grid, 0.0, 0.0, 0.0, T)
end

"""
    druyvesteyn!(vdf::VDF3D{N_vx, N_vy, N_vz}, grid::Grid3D{N_vx,N_vy,N_vz}, v0, T) where {N_vx, N_vy, N_vz}

Compute the Druyvesteyn distribution on a given velocity grid `grid` with a streaming velocity
`v0` and temperature `T`.
The distribution is stored in `vdf.w` and is normalized to 1.
**Note**: due to discretization and grid cut-off, actual velocity and temperature values might be different
than those passed to the function.
"""
function druyvesteyn!(vdf::VDF3D{N_vx, N_vy, N_vz}, grid::Grid3D{N_vx,N_vy,N_vz}, v0, T) where {N_vx, N_vy, N_vz}
    n = 0.0
    # alpha = (4 * 3 * T / (2 * gamma(5.0/4.0)))^(-4.0/5.0)
    alpha = ((2 * gamma(5.0/4.0) / (3 * T * gamma(3.0/4.0))))^2
    @inbounds for k in 1:grid.n_vz
        pow_z = (grid.vz[k]-v0[3])^2
        for j in 1:grid.n_vy
            pow_y = (grid.vy[j]-v0[2])^2
            for i in 1:grid.n_vx
                vdf.w[i,j,k] = exp(-alpha * ((grid.vx[i]-v0[1])^2+pow_y+pow_z)^2)
                n += vdf.w[i,j,k] * grid.Δvx[i] * grid.Δvy[j] * grid.Δvz[k]
            end
        end
    end

    scale_vdf!(vdf, grid, n)  # scale so that density is 1
end

"""
    bimodal!(vdf::VDF3D{N_vx, N_vy,N_vz}, grid::Grid3D{N_vx,N_vy,N_vz}, alpha1, v1, T1, alpha2, v2, T2) where {N_vx, N_vy, N_vz}

Compute a bimodal distribution on a given velocity grid `grid` with two Maxwellian components.
The first component has weight `alpha1`, streaming velocity `v1`, and temperature `T1`.
The second component has weight `alpha2`, streaming velocity `v2`, and temperature `T2`.
The distribution is stored in `vdf.w` and is normalized to 1.
"""
function bimodal!(vdf::VDF3D{N_vx, N_vy,N_vz}, grid::Grid3D{N_vx,N_vy,N_vz}, alpha1, v1, T1, alpha2, v2, T2) where {N_vx, N_vy, N_vz}
    n = 0.0
    for k in 1:grid.n_vz
        pow_z = (grid.vz[k])^2
        for j in 1:grid.n_vy
            pow_y = (grid.vy[j])^2
            for i in 1:grid.n_vx
                vdf.w[i,j] = alpha1 * exp(-((grid.vx[i]-v1)^2+pow_y+pow_z)/T1)
                vdf.w[i,j] += alpha2 * exp(-((grid.vx[i]-v2)^2+pow_y+pow_z)/T2)
                n += vdf.w[i,j,k] * grid.Δvx[i] * grid.Δvy[j] * grid.Δvz[k]
            end
        end
    end

    scale_vdf!(vdf, grid, n)
end

"""
    get_mott_smith_params(M, gamma)

Compute the parameters for the Mott-Smith distribution given the Mach number `M` and the adiabatic index `gamma`.

# Returns
* `rho1`: density of the first component
* `v1`: velocity of the first component
* `T1`: temperature of the first component
* `rho2`: density of the second component
* `v2`: velocity of the second component
* `T2`: temperature of the second component
"""
function get_mott_smith_params(M, gamma)
    # f1 = rho1 * exp(-(v-v1)^2/T1)
    # f2 = rho2 * exp(-(v-v2)^2/T2)

    rho_RH = M^2 * (gamma + 1) / (2 + M^2 * (gamma - 1))
    p_RH = (1 - gamma + 2 * gamma * M^2) / (1 + gamma)
    # return rho1, v1, T1, rho2, v2, T2
    return 1.0, sqrt(gamma) * M, 1.0, rho_RH, sqrt(gamma) * M / rho_RH, p_RH / rho_RH
end

"""
    mott_smith_mixing(x)

Compute the mixing ratio for the Mott-Smith distribution given the position `x`.
"""
function mott_smith_mixing(x)
    return 1.0 / (1.0 + exp(x))
end

"""
    mott_smith!(vdf::VDF3D{N_vx, N_vy,N_vz}, grid::Grid3D{N_vx,N_vy,N_vz}, x, rho1, v1, T1, rho2, v2, T2) where {N_vx, N_vy, N_vz}

Compute the Mott-Smith distribution on a given velocity grid `grid` with parameters `rho1`, `v1`, `T1`, `rho2`, `v2`, `T2`,
and mixing position `x`.
The distribution is stored in `vdf.w` and is normalized to 1.
"""
function mott_smith!(vdf::VDF3D{N_vx, N_vy,N_vz}, grid::Grid3D{N_vx,N_vy,N_vz}, x, rho1, v1, T1, rho2, v2, T2) where {N_vx, N_vy, N_vz}
    mixratio = mott_smith_mixing(x)

    # f_MB(T) ~ T^(-3/2), we incorporate that directly into the mixing factor
    return bimodal!(vdf, grid, mixratio * rho1 * T1^(-3/2), v1, 1.0 * T1, (1.0 - mixratio) * rho2 * T2^(-3/2), v2, T2)
end

"""
    mott_smith!(vdf::VDF3D{N_vx, N_vy,N_vz}, grid::Grid3D{N_vx,N_vy,N_vz}, x, M, gamma) where {N_vx, N_vy, N_vz}

Compute the Mott-Smith distribution on a given velocity grid `grid` with Mach number `M`, adiabatic index `gamma`,
and mixing position `x`.
The distribution is stored in `vdf.w` and is normalized to 1.
"""
function mott_smith!(vdf::VDF3D{N_vx, N_vy,N_vz}, grid::Grid3D{N_vx,N_vy,N_vz}, x, M, gamma) where {N_vx, N_vy, N_vz}
    # mixratio = mott_smith_mixing(x)
    rho1, v1, T1, rho2, v2, T2 = get_mott_smith_params(M, gamma)

    return mott_smith!(vdf, grid, x, rho1, v1, T1, rho2, v2, T2)
end
end