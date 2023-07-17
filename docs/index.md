# Overview

This project represents a quick implementation of unbiased 
Hamiltonian Monte Carlo (HMC).

The algorithm comes from the paper:

    J Heng, P E Jacob, Unbiased Hamiltonian Monte Carlo With Couplings, 
    Biometrika, Volume 106, Issue 2, June 2019, Pages 287–302, 
    https://doi.org/10.1093/biomet/asy074


## Theory

The general idea in the algorithm is to couple two HMC processes such that when they coincide, they can be used to produce unbiased estimates of the mean of the process.

This is done by incorporating a ranodm walk Metropolis-Hastings model, using maximally coupled Gaussian steps.

## Usage
