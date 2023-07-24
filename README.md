# Unbiased Hamiltonian Monte Carlo

[![Static Badge](https://img.shields.io/badge/Documentation-darkgreen)](https://www.helmuthnaumer.com/HMC_unbiased)


This package represents a quick implementation of the unbiased Hamiltonian Monte Carlo algorithm.


The algorithm comes from the paper:

    J Heng, P E Jacob, Unbiased Hamiltonian Monte Carlo With Couplings, 
    Biometrika, Volume 106, Issue 2, June 2019, Pages 287–302, 
    https://doi.org/10.1093/biomet/asy074

## Installation

The project depends on [`jax`](https://jax.readthedocs.io/en/latest/) and [`jaxtyping`](https://github.com/google/jaxtyping).
It is recommended that you install `jax` seperately according to their install instructions.

`HMC_unbiased` can be installed through pip using:

    pip install git+https://github.com/Helmuthn/HMC_unbiased.git
