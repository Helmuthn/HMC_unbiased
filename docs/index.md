# Overview

This project represents a quick implementation of unbiased 
Hamiltonian Monte Carlo (HMC).


The algorithm comes from the paper:

    J Heng, P E Jacob, "Unbiased Hamiltonian Monte Carlo With Couplings", 
    Biometrika, Volume 106, Issue 2, June 2019, Pages 287–302, 
    https://doi.org/10.1093/biomet/asy074

## Installation

The project depends on [`jax`](https://jax.readthedocs.io/en/latest/) and [`jaxtyping`](https://github.com/google/jaxtyping).
It is recommended that you install `jax` seperately according to their install instructions.

`HMC_unbiased` can be installed through pip using:

    pip install git+https://github.com/Helmuthn/HMC_unbiased.git



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

There are two main high-level functions in this implementation: `unbiased_HMC_chain` and `unbiased_HMC_step`.
These functions construct a the full MCMC chain until convergence and an individual step, respectively.

The general approach in this library is to begin with an unnormalized distribution, then construct the potential functions using `helpers.construct_potential`.
While not a particularly useful application, in the following example, we use Hamiltonian Monte Carlo to approximate a multivariate Gaussian distribution.

First, we begin with the required imports.
We additionally import jit from jax to compile the target distribution.

    from HMC_unbiased.helpers import construct_potential
    from HMC_unbiased.main import unbiased_HMC_chain
    from jax import jit
    import jax.random as random

Next, define a target distribution from which we will attempt to estimate the mean.
In this case, it will be an offset Gaussian distribution.

    @jit
    def distribution(x):
        squared_distance = -jnp.sum(jnp.square(x - 1))
        normalized_distance = squared_distance / 2
        density = jnp.exp(normalized_distance)/ ((2*jnp.pi)**(x.shape[0]/2))
        return density
    
    potential, potential_grad = construct_potential(distribution)
    
We then define the hyperparameters for the Markov chains.

    step_size = 0.05    # Leap-frog integrator step size
    num_steps = 10      # Number of leap-frog steps per HMC step
    gamma = 0.1         # Probability of random walk step
    std = 0.2           # Standard Deviation of random walk step

Choose some initialization for the Markov chains.

    key = random.PRNGkey(1234)
    key, subkey1, subkey2 = random.split(key, 3)
    Q1 = random.gaussian(subkey1, 5)
    Q2 = random.gaussian(subkey2, 5)

Finally, we can construct our sample chains.

    Q1, Q2, chain_length = unbiased_HMC_chain(Q1, Q2
                                              potential, potential_grad
                                              step_size, num_steps,
                                              gamma,
                                              std, marginal,
                                              key)
    
    Q1 = Q1[:chain_length, :]
    Q2 = Q2[:chain_length, :]

