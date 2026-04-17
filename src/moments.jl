function all_powers_up_to_M_3D(M)
    res::Vector{Tuple{Int64, Int64, Int64}} = []
    for i in 0:M
        for j in 0:M
            for k in 0:M
                if i+j+k <= M
                    push!(res, (i,j,k))
                end
            end
        end
    end

    return res
end

function construct_moment_measurement_matrix_3D(grid, n_v, moment_powers)
    mmms_unrolled_concatenated = zeros((length(moment_powers), n_v^3))

    @inbounds for i in 1:length(moment_powers)
        mmm_tmp = moment_measurement_tensor(grid, moment_powers[i][1], moment_powers[i][2], moment_powers[i][3])
        mmms_unrolled_concatenated[i,:] = mmm_tmp
    end

    return mmms_unrolled_concatenated
end


function moment_measurement_tensor(grid::Grid3D{N_vx,N_vy,N_vz}, m_vx, m_vy, m_vz) where {N_vx, N_vy, N_vz}
    res = ones((grid.n_vx, grid.n_vy, grid.n_vz))

    for k in 1:grid.n_vz
        pow_z = (grid.vz[k])^m_vz
        for j in 1:grid.n_vy
            pow_y = (grid.vy[j])^m_vy
            for i in 1:grid.n_vx
                res[i, j, k] = (grid.vx[i])^m_vx * pow_y * pow_z * grid.Δvx[i] * grid.Δvy[j] * grid.Δvz[k]
            end
        end
    end

    return res
end

function moment(vdf::VDF3D{N_vx, N_vy, N_vz}, grid, m_vx, m_vy, m_vz) where {N_vx, N_vy, N_vz}
    mom = 0.0
    @inbounds for k in 1:grid.n_vz
        pow_z = (grid.vz[k])^m_vz
        for j in 1:grid.n_vy
            pow_y = (grid.vy[j])^m_vy
            for i in 1:grid.n_vx
                mom += vdf.w[i,j,k] * (grid.vx[i])^m_vx * pow_y * pow_z * grid.Δvx[i] * grid.Δvy[j] * grid.Δvz[k]
            end
        end
    end

    return mom
end

function moment(vdf::VDF3D{N_vx, N_vy, N_vz}, mmm) where {N_vx, N_vy, N_vz}
    mom = 0.0
    @inbounds for k in 1:N_vz
        for j in 1:N_vy
            for i in 1:N_vx
                mom += vdf.w[i,j,k] * mmm[i,j,k]
            end
        end
    end

    return mom
end

function compute_moments(vdf::VDF3D{N_vx, N_vy, N_vz}, mmms) where {N_vx, N_vy, N_vz}
    moms = zeros(length(mmms))

    for i in eachindex(mmms)
        moms[i] = moment(vdf, mmms[i])
    end

    return moms
end