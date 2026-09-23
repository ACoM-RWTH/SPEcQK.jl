#!/usr/bin/env python
"""
Plot 1D moment-solver output (HDF5 written by SPEcQK's run_1d / init_moment_io).

Produces a 2x2 figure at a chosen snapshot, overlaying one line per input file:
    1: density                    -- the (0,0,0) moment
    2: y-velocity                 -- (0,1,0) / density
    3: temperature                -- (2/3) * thermal energy from the 2nd moments
    4: <given moment> / density   -- the moment order passed on the command line

Example:
    python plot_1D.py --files a.h5 b.h5 --labels "M=2" "M=3" \\
        --moment 1,1,0 --timestep 10 --length 1.0 --output couette.png
"""

import argparse

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_moment(s):
    """'1,2,1' -> (1, 2, 1)."""
    parts = [int(p) for p in s.replace(" ", "").split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"moment must be 'a,b,c', got {s!r}")
    return tuple(parts)


def find_row(powers, target):
    """Row index of moment power `target` (a,b,c) in the (m,3) powers array."""
    for i, p in enumerate(powers):
        if tuple(int(v) for v in p) == tuple(target):
            return i
    raise ValueError(f"moment {tuple(target)} not present in file")


def read_snapshot(path, timestep):
    """Return (powers (m,3), snapshot (n_cells, m), time, step) at `timestep`."""
    with h5py.File(path, "r") as f:
        powers = f["moment_powers"][...]          # (m, 3)
        moments = f["moments"]                     # (n_snap, n_cells, m)
        n_snap = moments.shape[0]
        if not (-n_snap <= timestep < n_snap):
            raise IndexError(
                f"{path}: timestep {timestep} out of range (n_snapshots={n_snap})"
            )
        snap = moments[timestep, :, :]             # (n_cells, m)
        time = float(f["times"][timestep])
        step = int(f["steps"][timestep])
    return powers, snap, time, step


def derived_fields(powers, snap, moment):
    """Compute (density, uy, temperature, moment/density) profiles for one file."""
    col = lambda a, b, c: snap[:, find_row(powers, (a, b, c))]

    rho = col(0, 0, 0)
    ux, uy, uz = col(1, 0, 0) / rho, col(0, 1, 0) / rho, col(0, 0, 1) / rho

    # thermal energy <c^2> = <v^2> - u^2, then T = (2/3) * energy
    energy = (col(2, 0, 0) + col(0, 2, 0) + col(0, 0, 2)) / rho - (ux**2 + uy**2 + uz**2)
    temperature = (2.0 / 3.0) * energy

    single = col(*moment) / rho
    return rho, uy, temperature, single


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--files", nargs="+", required=True, help="HDF5 output files")
    ap.add_argument("--labels", nargs="+", required=True, help="legend label per file")
    ap.add_argument("--moment", type=parse_moment, required=True,
                    help="moment order for subplot 4, e.g. '1,2,1'")
    ap.add_argument("--timestep", type=int, required=True,
                    help="snapshot index (0-based; negatives count from the end)")
    ap.add_argument("--length", type=float, required=True, help="domain length")
    ap.add_argument("--output", default="plot_1D.png", help="output image path")
    args = ap.parse_args()

    if len(args.files) != len(args.labels):
        ap.error(f"got {len(args.files)} files but {len(args.labels)} labels")

    a, b, c = args.moment
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    ax_rho, ax_uy, ax_T, ax_m = axes.flat

    times = []
    for path, label in zip(args.files, args.labels):
        powers, snap, time, step = read_snapshot(path, args.timestep)
        times.append(time)

        n_cells = snap.shape[0]
        dx = args.length / n_cells
        x = (np.arange(n_cells) + 0.5) * dx      # cell centers

        rho, uy, T, single = derived_fields(powers, snap, args.moment)

        ax_rho.plot(x, rho, label=label)
        ax_uy.plot(x, uy, label=label)
        ax_T.plot(x, T, label=label)
        ax_m.plot(x, single, label=label)

    ax_rho.set(title="Density", ylabel=r"$\rho$  $(0,0,0)$")
    ax_uy.set(title="y-velocity", ylabel=r"$u_y = M_{010}/\rho$")
    ax_T.set(title="Temperature", ylabel=r"$T = \frac{2}{3}\,e_{\mathrm{th}}$")
    ax_m.set(title=f"Moment ({a},{b},{c}) / density",
             ylabel=rf"$M_{{{a}{b}{c}}}/\rho$")
    for ax in (ax_T, ax_m):
        ax.set_xlabel("x")
    for ax in axes.flat:
        ax.set_xlim(0.0, args.length)
        ax.grid(True, alpha=0.3)

    ax_rho.legend()   # legend on the first subplot only

    t_txt = f"t = {times[0]:.4g}" if len(set(f"{t:.6g}" for t in times)) == 1 \
        else f"snapshot {args.timestep}"
    fig.suptitle(f"1D moment solver — {t_txt}")
    fig.tight_layout()
    fig.savefig(args.output, dpi=130)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
