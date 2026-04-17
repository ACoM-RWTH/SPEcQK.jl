using StaticArrays

function get_dv(v_grid_unrolled::AbstractVector)
    dv = zeros(length(v_grid_unrolled))
    dv[1] = v_grid_unrolled[2] - v_grid_unrolled[1]
    dv[end] = v_grid_unrolled[end] - v_grid_unrolled[end-1]
    for i in 2:length(v_grid_unrolled)-1
        dv[i] = 0.5 * (v_grid_unrolled[i] - v_grid_unrolled[i-1] + v_grid_unrolled[i+1] - v_grid_unrolled[i])
    end
    return dv
end

struct Grid3D{N_vx, N_vy, N_vz}
    n_vx::Int64
    n_vy::Int64
    n_vz::Int64
    vx::SVector{N_vx, Float64}
    vy::SVector{N_vy, Float64}
    vz::SVector{N_vz, Float64}
    Δvx::SVector{N_vx, Float64}
    Δvy::SVector{N_vy, Float64}
    Δvz::SVector{N_vz, Float64}
    vxsq::SVector{N_vx, Float64}
    vysq::SVector{N_vy, Float64}
    vzsq::SVector{N_vz, Float64}
end

function Grid3D(vx_lims, vy_lims, vz_lims, n_vx, n_vy, n_vz)
    @assert vx_lims[2] > vx_lims[1]
    @assert vy_lims[2] > vy_lims[1]
    @assert vz_lims[2] > vz_lims[1]
    @assert n_vx >= 2
    @assert n_vy >= 2
    @assert n_vz >= 2
    vx = LinRange(vx_lims[1], vx_lims[2], n_vx)
    vy = LinRange(vy_lims[1], vy_lims[2], n_vy)
    vz = LinRange(vz_lims[1], vz_lims[2], n_vz)
    return Grid3D{n_vx, n_vy, n_vz}(n_vx, n_vy, n_vz, vx, vy, vz, get_dv(vx), get_dv(vy), get_dv(vz), vx.^2, vy.^2, vz.^2)
end

struct VDF3D{N_vx, N_vy, N_vz}
    w::Array{Float64,3}
end

function VDF3D(grid::Grid3D{N_vx,N_vy,N_vz}) where {N_vx, N_vy, N_vz}
    return VDF3D{grid.n_vx, grid.n_vy, grid.n_vz}(zeros((grid.n_vx, grid.n_vy, grid.n_vz)))
end

function grid_weights(grid::Grid3D{N_vx, N_vy, N_vz}) where {N_vx, N_vy, N_vz}
    w = zeros((grid.n_vx, grid.n_vy, grid.n_vz))

    for k in 1:grid.n_vz
        for j in 1:grid.n_vy
            for i in 1:grid.n_vx
                w[i,j,k] = grid.Δvx[i] * grid.Δvy[j] * grid.Δvz[k]
            end
        end
    end

    return w
end