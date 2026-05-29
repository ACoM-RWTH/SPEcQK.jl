function lax_friedrichs_1d!(moments_next, moments_current, dt, dx, max_eigenvalue,
                                n_cells, n_moments)
    inv_dx = 1.0 / dx
    
    @inbounds for m in 1:n_moments
        for i in 1:n_cells
            u_i = moments_current[m, i]
            u_ip1 = moments_current[m, i+1]
            flux_num = 0.5 * max_eigenvalue * (u_i + u_ip1) - 0.5 * max_eigenvalue * (u_ip1 - u_i)
            moments_next[m, i] = moments_current[m, i] - dt * inv_dx * flux_num
        end
    end
    
    return nothing
end

function compute_moments_from_vdf!(moments, gsol, w, mmm, n, m)
    @inbounds for i in 1:n
        gsol[i] *= w[i]
    end
    
    @inbounds for j in 1:m
        s = 0.0
        for i in 1:n
            s += mmm[j, i] * gsol[i]
        end
        moments[j] = s
    end
    
    return nothing
end

function dummy_ghost_cell_left!(moments_ghost, moments_inner, n_moments)
    @inbounds for m in 1:n_moments
        moments_ghost[m] = moments_inner[m]
    end
    return nothing
end

function dummy_ghost_cell_right!(moments_ghost, moments_inner, n_moments)
    @inbounds for m in 1:n_moments
        moments_ghost[m] = moments_inner[m]
    end
    return nothing
end

function convect_1D_LF!(moments_all, dt, dx, max_eigenvalue, N,
                       n_cells, n_moments_all, n_moments_constraint,
                       n_v, mmm_constraint, mmm_all, Δv, λ,
                       A, rnorm_A, mvec, alpha, gsol, uvec, d_w, inv_dw,
                       y0, y_new, grad, H, Hreg, p, F_ch, info,
                       target_KL, moments_work, moments_next,
                       grid, moment_powers_constraint, vdf_mb, w_local)
    n = n_v^3
    m_constraint = n_moments_constraint
    m_all = n_moments_all
    
    # Find indices for the moments we need
    density_index = find_index(moment_powers_constraint, (0,0,0))
    vx_index = find_index(moment_powers_constraint, (1,0,0))
    vy_index = find_index(moment_powers_constraint, (0,1,0))
    vz_index = find_index(moment_powers_constraint, (0,0,1))
    Ex_index = find_index(moment_powers_constraint, (2,0,0))
    Ey_index = find_index(moment_powers_constraint, (0,2,0))
    Ez_index = find_index(moment_powers_constraint, (0,0,2))
    
    @inbounds for step in 1:N
        for cell in 1:n_cells
            # Extract local moments for this cell
            ref_moms_local = @view moments_all[1:m_constraint, cell]
            
            # Compute local Maxwell-Boltzmann distribution
            find_MB_solution!(vdf_mb, grid, ref_moms_local, vx_index, vy_index, vz_index,
                             Ex_index, Ey_index, Ez_index; tol=1e-11)
            
            # Get the local weighting function
            unroll!(w_local, vdf_mb.w)
            
            # Recompute all arrays that depend on w_local
            compute_constraint_matrix!(A, mmm_constraint, w_local, n, m_constraint)
            
            fill!(rnorm_A, 0.0)
            for j in 1:n
                for i in 1:m_constraint
                    rnorm_A[i] += A[i,j]^2
                end
            end
            
            for i in 1:m_constraint
                rnorm_A[i] = sqrt(rnorm_A[i])
            end
            
            for j in 1:n
                for i in 1:m_constraint
                    A[i,j] /= rnorm_A[i]
                end
            end
            
            compute_d_w_factors!(d_w, inv_dw, Δv, w_local, n)
            compute_alpha!(alpha, w_local, inv_dw, target_KL, λ, n)
            y0 .= dual_from_primal_guess!(A, alpha, inv_dw)
            compute_alpha!(alpha, w_local, inv_dw, target_KL, λ, n)
            
            # Load cell moments into mvec
            for i in 1:m_constraint
                mvec[i] = moments_all[i, cell] / rnorm_A[i]
            end
            
            newton_dual!(gsol, A, mvec, alpha, y0, y_new, d_w, inv_dw, uvec,
                         grad, H, Hreg, n, m_constraint, F_ch, p, info;
                         tol=1e-9, maxiter=100, mu=1e-12, backtrack_rho=0.5, backtrack_c=1e-4)
            
            compute_moments_from_vdf!(view(moments_work, :, cell), gsol, w_local, mmm_all, n, m_all)
        end
        
        for m in 1:m_all
            dummy_ghost_cell_left!(view(moments_work, m, 0:0), view(moments_work, m, 1:1), 1)
            dummy_ghost_cell_right!(view(moments_work, m, n_cells+1:n_cells+1), view(moments_work, m, n_cells:n_cells), 1)
        end
        
        lax_friedrichs_1d!(moments_next, moments_work, dt, dx, max_eigenvalue, n_cells, m_all)
        
        for m in 1:m_all
            for i in 1:n_cells
                moments_all[m, i] = moments_next[m, i]
            end
        end
    end
    
    return nothing
end
