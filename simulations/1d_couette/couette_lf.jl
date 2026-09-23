using SPEcQK

# 1D Couette flow (Baranger et al. 2019, §4.1).
#
# The gas lies between two diffuse-reflection walls held at the same temperature
# `T_wall`. The left wall is at rest; the right wall slides tangentially with
# velocity `u_wall` in the y-direction (so u_wall·n = 0). The gas starts at rest
# in equilibrium with the walls and is dragged into a shear flow by the moving
# wall: its diffuse-reflection Maxwellian (centered at vy = u_wall) injects
# y-momentum at the right boundary, which Lax–Friedrichs transport and BGK
# relaxation spread across the domain toward the steady Couette profile.
#
# "1D" refers to physical space x ∈ [0, 1]; the velocity space is the full 3D
# Grid3D. Output (moment field vs x over time) is streamed to an HDF5 file.

function run_couette(; lambda_unscaled = 0.0,
                       T_wall  = 1.0,
                       T_wall_left  = T_wall,   # the two walls may be held at
                       T_wall_right = T_wall,   # different temperatures
                       u_wall  = 2.0,
                       n_ref   = 1.0,           # domain-mean density of the initial state
                       n_v     = 16,
                       extent  = 6.0,     # must comfortably cover vy = u_wall + a few thermal speeds
                       n_cells = 10,
                       max_moment_constraint = 3,
                       CFL     = 0.8,
                       t_end   = 4.0,
                       BGK_factor = 1.0,  # collisions on
                       mu_sref     = 0.3,  # BGK collision scale (viscosity at T=1); larger = more rarefied
                       io_freq = 200,
                       flux    = :upwind,   # :lf, :lf_lw, :upwind, :upwind_lw
                       io_path = joinpath(@__DIR__, "..", "..", "output",
                                          "tscouette_$(mu_sref)_$(max_moment_constraint)_$(n_cells)_$(n_v)_$(lambda_unscaled)_$(flux).h5"),
                       do_fallback=true)

    mkpath(dirname(io_path))

    # the two diffuse-reflection walls: equal and opposite tangential velocity
    wall_left  = (T = T_wall_left,  vy = -u_wall/2, vz = 0.0)
    wall_right = (T = T_wall_right, vy =  u_wall/2, vz = 0.0)

    # Initial condition: the states the two walls would impose, ramped linearly
    # across the domain (`interpolate_init_solution`), instead of a gas at rest.
    # That removes the slow bulk shear mode which otherwise dominates the whole
    # transient -- at mu_sref = 0.015 relaxation from rest takes t ~ 300.
    #
    # The density follows from assuming uniform pressure. At steady state the
    # x-momentum balance gives d(p_xx)/dx = 0, so the normal stress is exactly
    # constant across the gap, and p = n*T equals it to leading order in Kn --
    # so constant pressure is the right assumption here. The pressure level p0
    # is fixed by asking for a domain-mean density of `n_ref`; because the
    # density is ramped *linearly*, its mean is the average of the two
    # endpoints, which makes p0 the harmonic mean of the wall temperatures.
    #
    # For equal wall temperatures this collapses to n = n_ref everywhere, i.e.
    # exactly the uniform initial condition used before. When they differ, p is
    # exact at the two walls but bulges in between by (1+r)^2/(4r) for a
    # temperature ratio r (12.5% at r = 2), since a linear n times a linear T is
    # quadratic; that excess relaxes on the fast acoustic timescale, not the
    # slow viscous one.
    p0 = 2 * n_ref * wall_left.T * wall_right.T / (wall_left.T + wall_right.T)
    left_state  = (n = p0 / wall_left.T,  ux = 0.0,
                   vy = wall_left.vy,  vz = wall_left.vz,  T = wall_left.T)
    right_state = (n = p0 / wall_right.T, ux = 0.0,
                   vy = wall_right.vy, vz = wall_right.vz, T = wall_right.T)

    res = run_1d(; lambda_unscaled = lambda_unscaled,
                    max_moment_constraint = max_moment_constraint,
                    n_v = n_v, extent = extent,
                    n_cells = n_cells, x_min = 0.0, x_max = 1.0,
                    CFL = CFL, t_end = t_end,
                    left_state  = left_state, right_state = right_state,
                    wall_left   = wall_left, wall_right = wall_right,
                    interpolate_init_solution = true,
                    BGK_factor  = BGK_factor, mu_sref = mu_sref,
                    io_path     = io_path, io_freq = io_freq,
                    verbose     = true,
                    flux = flux, do_fallback=do_fallback)

    log_path = joinpath(@__DIR__, "..", "..", "output",
                        "couette_log_$(mu_sref)_$(max_moment_constraint)_$(n_cells)_$(n_v)_$(lambda_unscaled)_$(flux).log")

    open(log_path, "w") do io
        println(io, res.sparsity)
    end
end

# run_couette(lambda_unscaled=1e-8, do_fallback=false, t_end=0.1)


# run_couette(lambda_unscaled=0.0)
# Kn ~0.4, M=4

const Mmax = 6

for λ in [0.0, 1e-6, 1e-4, 1e-2]
    # Kn ~0.4
    run_couette(mu_sref=0.3, lambda_unscaled=λ, max_moment_constraint=Mmax, flux=:lf, t_end=10.0, n_cells=50,
                do_fallback=false)
    run_couette(mu_sref=0.3, lambda_unscaled=λ, max_moment_constraint=Mmax, flux=:lf, t_end=10.0, n_cells=100,
                do_fallback=false)
    run_couette(mu_sref=0.3, lambda_unscaled=λ, max_moment_constraint=Mmax, flux=:upwind, t_end=10.0, n_cells=50,
                do_fallback=false)
    run_couette(mu_sref=0.3, lambda_unscaled=λ, max_moment_constraint=Mmax, flux=:upwind, t_end=10.0, n_cells=100,
                do_fallback=false)
    run_couette(mu_sref=0.3, lambda_unscaled=λ, max_moment_constraint=Mmax, flux=:upwind_lw, t_end=10.0, n_cells=50,
                do_fallback=false)
    run_couette(mu_sref=0.3, lambda_unscaled=λ, max_moment_constraint=Mmax, flux=:upwind_lw, t_end=10.0, n_cells=100,
                do_fallback=false)
end

# # Kn ~ 0.02
# run_couette(mu_sref=0.015, max_moment_constraint=Mmax, flux=:lf, t_end=12.0, n_cells=50)
# run_couette(mu_sref=0.015, max_moment_constraint=Mmax, flux=:lf, t_end=12.0, n_cells=100)
# run_couette(mu_sref=0.015, max_moment_constraint=Mmax, flux=:upwind, t_end=12.0, n_cells=50)
# run_couette(mu_sref=0.015, max_moment_constraint=Mmax, flux=:upwind, t_end=12.0, n_cells=100)
# run_couette(mu_sref=0.015, max_moment_constraint=Mmax, flux=:upwind_lw, t_end=12.0, n_cells=50)
# run_couette(mu_sref=0.015, max_moment_constraint=Mmax, flux=:upwind_lw, t_end=12.0, n_cells=100)
