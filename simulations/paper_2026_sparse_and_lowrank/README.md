# Reproducing results from "Sparse and low-rank kinetic distribution estimation"

The `*.jl` files in this directory produce the results from the paper "Sparse and low-rank kinetic distribution estimation" by G. Oblapenko, L. Theisen, R.-P. Wilhelm, M. Torrilhon, M. Herty. To run them, first create an `output` subdirectory in the root directory of the project. Then, run `julia --project=. simulations/paper_2026_sparse_and_lowrank/<file_name>.jl` from the root directory of the project (note that running the `nufi.jl` and `nufi_KL_w.jl` files requires the NuFI simulation data to be downloaded [from Zenodo](https://)). The parameters of
the simulations are listed [below in this readme](#Main-simulation-parameters).

The results can also be downloaded in HDF5 format from [Zenodo](https://). The output format description is listed [below in this readme](#Simulation-output-format).

## Maxwell-Boltzmann and Druyvesteyn distributions
The data for the Maxwell-Boltzmann and Druyvesteyn distributions are produced by running the `mb_and_druyvesteyn.jl` file.
The default naming scheme for the output is `<distribution_name>_constraint_M_upto<m_constraint>_<n_v>.h5`,
where `<distribution_name>` is either `Maxwell_Boltzmann` or `Druyvesteyn`, `<m_constraint>` is maximum order of the moment constraints used, and `<n_v>` is the number of velocity grid points in each direction.

## Mott-Smith distribution
The data for the Mott-Smith distributions are produced by running the `mb_and_druyvesteyn.jl` file.
The default naming scheme for the output is `Mott_Smith_constraint_M_upto<m_constraint>_<n_v>.h5`,
where `<distribution_name>` is either `Maxwell_Boltzmann` or `Druyvesteyn`, `<m_constraint>` is maximum order of the moment constraints used, and `<n_v>` is the number of velocity grid points in each direction.
The parameter study over different values of the epsilon threshold parameters produces files named 
`Mott_Smith<sparse_threshold>_constraint_M_upto<m_constraint>_<n_v>.h5`, where `<sparse_threshold>` is the value of the epsilon threshold parameter.

## Reconstruction of NuFI data
The sparse reconstructions of the data produced by the Numerical Flow Iteration (NuFI) solver are produced by running
the `nufi.jl` and `nufi_KL_w.jl` files. To run the files, one first needs to download the distribution data produced by NuFI,
available [on Zenodo](https://).
One can run simulation files `nufi.jl` and `nufi_KL_w.jl` by calling `julia --project=. nufi.jl PATH_TO_NUFI_DATA` (or  `julia --project=. nufi_KL_w.jl PATH_TO_NUFI_DATA`), where `PATH_TO_NUFI_DATA` is the path to the folder containing the NuFI data.

`nufi.jl` reconstructs the NuFI data based on entropy minimization with L1 regularization, using a Maxwell-Boltzmann distribution
as the weighting function. `nufi_KL_w.jl` reconstructs the NuFI data based on Kullback-Leibler divergence minimization with L1 regularization,
using the original distribution produced by NuFI as the weighting function.

## Post-processing script
The `plotting.ipynb` file is a Jupyter notebook that can be used to post-process the results of the simulations. It is written in Python and requires the following packages:
- jupyter
- numpy
- matplotlib
- h5py

The plotting parameters (fonts and fontsizes) are set at the top of the notebook, `path_to_output_dir` should be set to the path of the output directory of the simulations relative to the notebook. One should create a `plots` directory in the root directory of the package to store the plots.
To execute all the cells of the notebook, one can run the following command in the terminal: `jupyter notebook plotting.ipynb`.

### Notes on simulation parameters and output format

## Main simulation parameters
All of the simulation files have a `run` function, which takes the following parameters:
- `target_vdf`: a function that takes a `VDF3D` and `Grid3D` instance and evaluates the target distribution function at the velocity grid points
- `output_prefix`: path to output directory (relative to the root directory of the project)
- `target_vdf_name`: name to prepend to the output files
- `n_v`: number of velocity grid points in each dimension
- `extent`: extent of the velocity grid in each dimension
- `lambda_values_unscaled`: array of (unscaled) values of the regularization strength `lambda` to iterate over
- `max_moment_constraint`: maximum total order of the moment constraints
- `output`: if `true`, HDF5 output will be written
- `threshold`: value of the threshold below which values of the predicted distribution `g` will be set to 0 (default: `1e-7`)

The parameters are set after the `run` function definition in each simulation file, and the `run` function is called with these parameters.

## Simulation output format
The output is in HDF5 format, with the following fields (depending on row- or column major-ordering of the language used to read the files,
the order of dimensions may be reversed):
- `grid`: the velocity grid of dimensions `n_vx x n_vy x n_vz x 3`, the last dimension contains the x/y/z velocity components at a specific grid node
- `lambdas`: the array of (unscaled) values of the regularization strength `lambda` used in the simulations that have been iterated over
- `moment_measurement_matrix_constraints`: matrix of dimension `n_constraints x (n_vx x n_vy x n_vz)` containing the measurement matrix for the constraints
- `moment_measurement_matrix_test`: matrix of dimension `n_test x (n_vx x n_vy x n_vz)` containing the measurement matrix for the moments used for testing
- `predicted_M_a,b,c` (multiple values): vector of length `n_lambda` of predicted values of moments of order (`a,b,c`) for each `lambda` value
- `ref_M_constraint_a,b,c`: value of reference moment value of order (`a,b,c`) used as constraint
- `ref_M_test_a,b,c`:  value of reference moment value of order (`a,b,c`) used for testing (not acting as constraint)
- `solutions`: array of dimension `n_vx x n_vy x n_vz x n_lambda`, each slice of the last dimension contains the solution `g` for a specific `lambda` value
- `sparsity_degree`: 
- `vdf_hidden_truth`: the hidden truth velocity distribution function, an array of dimension `n_vx x n_vy x n_vz`
- `vdf_hidden_truth_unrolled`: the hidden truth velocity distribution function unrolled into a vector of length `n_vx x n_vy x n_vz`
- `weighting_function`: the weighting function `w` used, an array of dimension `n_vx x n_vy x n_vz`

A full reconstructed distribution is given by the product of the weighting function and the solution, i.e `w .* solutions[:,:,:,i]`, where
`i` is an index denoting which value of `lambda` was used, i.e. `lambdas[i]`.