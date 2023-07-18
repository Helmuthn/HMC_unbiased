# Overview

This project represents a quick implementation of unbiased 
Hamiltonian Monte Carlo (HMC).

It can be installed through pip using:

    pip install git+https://github.com/Helmuthn/HMC_unbiased.git

The algorithm comes from the paper:

    J Heng, P E Jacob, Unbiased Hamiltonian Monte Carlo With Couplings, 
    Biometrika, Volume 106, Issue 2, June 2019, Pages 287–302, 
    https://doi.org/10.1093/biomet/asy074


## Theory

The general idea in the algorithm is to couple two HMC processes such that when they coincide, they can be used to produce unbiased estimates of the mean of the process.
This class of methods takes advantage of a telescoping sum.

First, construct two Markov chains $(X_i, Y_i)$ such that they both converge to a desired stationary distribution $\pi$, have an identical marginal distribution for each step, and are coupled such that if $X_i = Y_{i+1}$ for some $i$, then $X_j = Y_{j+1}$ for all $j > i$.

As $X_i$ is asymptotically unbiased, let $\tilde X$ be a random variable from the limiting distribution, and

$$
\begin{align}
\mathbb{E}\left[\tilde X\right] &= \mathbb{E}\left[X_i + \sum_{j=i}^{\infty}(X_{j+1} - X_{j})\right]\\\\
&= \mathbb{E}\left[X_i\right] + \sum_{j=i}^{\infty}\mathbb{E}\left[X_{j+1} - X_{j}\right]\\\\
&= \mathbb{E}\left[X_i\right] + \sum_{j=i}^{\infty}\mathbb{E}\left[Y_{j+1} - X_{j}\right]\\\\
&= \mathbb{E}\left[X_i\right] + \sum_{j=i}^{T}\mathbb{E}\left[Y_{j+1} - X_{j}\right],
\end{align}
$$
where $T$ is the meeting time of the two processes.
Equation (3) comes from noting that $X_i$ and $Y_i$ have the same distribution by construction, and Equation (4) comes from the constraint that $Y_{j+1} = X_j$ for any timesteps after $T$.

Beyond verifying technical assumptions described in the paper, the main challenge in this class of algorithms is to ensure that the chains meet in finite time with high probability.

The authors do this through the incorporation of a maximally coupled random walk model.
That is, with some probability $\gamma$, rather than take a step of HMC, instead step according to a Metropolis-Hastings random walk with a proposal distribution being a pair of maximally coupled Gaussian distributions.


## Usage
