using SPEcQK

function run_sod(;
                n_v     = 21,
                extent  = 5.0,     # must comfortably cover vy = u_wall + a few thermal speeds
                n_cells = 10,
                max_moment_constraint = 6,
                CFL     = 0.8,
                t_end   = 0.2,
                BGK_factor = 1.0,  # collisions on
                mu_sref     = 0.3,  # BGK collision scale (viscosity at T=1); larger = more rarefied
                io_freq = 10,
                dissipation = :global,
                io_path = joinpath(@__DIR__, "..", "..", "output", "sod", "dvm",
                "sod_dvm_$(mu_sref)_$(max_moment_constraint)_$(n_cells)_$(n_v)_$(dissipation)_$(BGK_factor).h5"))

    mkpath(dirname(io_path))

    left_state  = (n = 1.0,   ux = 0.0, T = 1.0)   # p_L = 0.5
    right_state = (n = 0.125, ux = 0.0, T = 0.8)   # p_R = 0.05  → 10:1 pressure, 8:1 density
    wall_left   = (T = 1.0, vy = 0.0, vz = 0.0)    # walls in equilibrium with adjacent gas
    wall_right  = (T = 0.8, vy = 0.0, vz = 0.0)

    return run_1d_dvm(;
                    max_moment_constraint = max_moment_constraint,
                    n_v = n_v, extent = extent,
                    n_cells = n_cells, x_min = 0.0, x_max = 1.0,
                    CFL = CFL, t_end = t_end,
                    left_state  = left_state, right_state = right_state,
                    wall_left   = wall_left, wall_right = wall_right,
                    interpolate_init_solution = false,
                    BGK_factor  = BGK_factor, mu_sref = mu_sref,
                    io_path     = io_path, io_freq = io_freq,
                    verbose     = true,
                    dissipation = dissipation,
                    )
end

# run_couette(lambda_unscaled=0.0)
# run_sod(mu_sref=0.002, dissipation=:upwind, t_end=0.1, n_cells=400)
# # Kn-L ~ 0.025
run_sod(mu_sref=0.02, dissipation=:upwind, n_v=22, t_end=0.1, n_cells=400)

# # Kn-L ~ 0.25
# run_sod(mu_sref=0.2, dissipation=:upwind, t_end=0.1, n_cells=400)

# # Kn-L ~ inf
run_sod(mu_sref=0.2, BGK_factor=0.0, n_v=22, dissipation=:upwind, t_end=0.1, n_cells=400)