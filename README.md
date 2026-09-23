# SPEcQK.jl

This is the repository of the Julia package SPEcQK.jl ("SParse Entropic Quadrature for Kinetics"), which implements the sparse and low-rank kinetic distribution estimation algorithm described in the preprint "Sparse and low-rank kinetic distribution estimation" by G. Oblapenko, L. Theisen, R.-P. Wilhelm, M. Torrilhon, M. Herty.

The code is released under the MPL 2.0 license, see LICENSE for details.

Documentation available [on the NRW Gitlab](https://georgii.oblapenko.pages.git.nrw/specqk)

**The current documentation is largely AI-generated, this will be cleaned-up and made more human-readable in upcoming releases**.

## Installation, tests, running examples

Currently, the package can be used as follows:
- Clone the repository
- Navigate to the project directory
- Run `julia --project=. -e 'using Pkg; Pkg.instantiate()'`

To run tests:
- `julia --project=. -e 'using Pkg; Pkg.test()'`

To run a simulation file in `simulations`
- Create a directory `output` by running `mkdir output` (the directory is already in `.gitignore`)
- Run a simulation file, e.g. `julia --project=. simulations/paper_2026_sparse_and_lowrank/mott_smith.jl'`

## Reproducibility of papers

### Sparse and low-rank kinetic distribution estimation
Details on reproducing the results of "Sparse and low-rank kinetic distribution estimation" can be found in [reproducibility.md](simulations/paper_2026_sparse_and_lowrank/README.md).

### Sparse entropic quadrature for moment equations
Details on reproducing the results of "Sparse entropic quadrature for moment equations" can be found in [reproducibility.md](simulations/paper_2026_sparse_entropic/README.md).

## Acknowledgments
Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) within the SFB 1481 (442047500) "Sparsity and Singular Structures", Project B04 "Sparsity Patterns in Kinetic
Theory".