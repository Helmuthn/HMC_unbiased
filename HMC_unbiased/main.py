import jax
from jaxtyping import Float, Array, Key
from typing import Callable

from .helpers import sample_gaussian_max_coupling, HMC_step

from functools import partial

@partial(jax.jit, static_argnums=(2,3,8))
def unbiased_HMC_step(Q1: Float[Array, " dim"], 
                      Q2: Float[Array, " dim"], 
                      potential: Callable[[Float[Array, " dim"]], Float],
                      potential_grad: Callable[[Float[Array, " dim"]], 
                                                Float[Array, " dim"]], 
                      step_size: Float, 
                      num_steps: int,
                      gamma: Float,
                      std: Float,
                      marginal: Callable[[Float[Array, " dim"]], Float],
                      key: Key
            ) -> tuple[Float[Array, " dim"], Float[Array, " dim"]] :
    """Completes a step of unbiased Hamiltonian Monte Carlo.

    Completes a step of the unbiased HMC process by randomly sleecting
    between HMC and a maximally coupled random walk Metropolis-Hastings process.
    
    Args:
        Q1: Current state of the first Markov chain
        Q2: Current state of the second Markov chain
        potential: Potential function
        potential_grad: Gradient of the potential function
        step_size: Stepsize for leap-frog integration in HMC
        num_steps: Number of steps for leap-frog integration in HMC
        gamma: Probability of selecting random walk 
        std: Standard deviation for random walk
        marginal: Target marginal distribution
        key: Jax pseudorandom number key
    
    Returns:
        The result of an unbiased step of Hamiltonian Monte Carlo    
    """

    key1, key2 = jax.random.split(key)
    random_walk_choice = jax.random.bernoulli(key1, gamma)
    args = (Q1, 
            Q2, 
            step_size, 
            num_steps, 
            std, 
            key2)

    def random_walk(args):
        Q1 = args[0]
        Q2 = args[1]
        std = args[4]
        key = args[5]
        return coupled_randomwalk_step(Q1, Q2, std, marginal, key)

    def hmc(args):
        Q1 = args[0]
        Q2 = args[1]
        step_size = args[2]
        num_steps = args[3]
        key = args[5]
        return coupled_HMC_step(Q1, Q2, 
                                potential, potential_grad, 
                                step_size, num_steps, 
                                key)

    return jax.lax.cond(random_walk_choice, random_walk, hmc, args)


@partial(jax.jit, static_argnums=(2,3))
def coupled_HMC_step(Q1: Float[Array, " dim"], 
                     Q2: Float[Array, " dim"], 
                     potential: Callable[[Float[Array, " dim"]], Float],
                     potential_grad: Callable[[Float[Array, " dim"]], 
                                    Float[Array, " dim"]], 
                     step_size: Float, 
                     num_steps: int,
                     key: Key
            ) -> tuple[Float[Array, " dim"], Float[Array, " dim"]] :
    """ Completes a coupled step of Hamiltonian Monte Carlo.

    Given the current states of two Hamiltonian Monte Carlo Markov chains,
    compute a pair of new states of the two chains using the same momentum
    and uniform random variable for acceptance.

    Args:
        Q1: Position in the first chain
        Q2: Position in the second chain
        potential: Potential function
        potential_grad: Gradient of the potential function
        step_size: Step size for leap-frog integration
        num_steps: Number of steps for leap-frog integration
        key: Jax pseudorandom number key
    
    Returns:
        Updated values for `Q1` and `Q2`.
    
    Notes:
        See Algorithm 1 in https://arxiv.org/abs/1709.00404 for details.
    """
    key1, key2 = jax.random.split(key, 2)
    momentum = jax.random.normal(key1, Q1.shape)
    U = jax.random.uniform(key2)

    Q1_out = HMC_step(momentum, Q1, U, 
                      potential, potential_grad, 
                      step_size, num_steps)

    Q2_out = HMC_step(momentum, Q2, U, 
                      potential, potential_grad, 
                      step_size, num_steps)
    
    return Q1_out, Q2_out


@partial(jax.jit, static_argnums=(3,))
def coupled_randomwalk_step(Q1: Float[Array, " dim"], 
                            Q2: Float[Array, " dim"], 
                            std: Float, 
                            marginal: Callable[[Float[Array, " dim"]], Float], 
                            key: Key
            ) -> tuple[Float[Array, " dim"], Float[Array, " dim"]] :
    """Compute a step of a coupled Metropolis-Hastings random walk.
    
    Computes a step of an unbiased random walk Metropolis-Hastings MCMC 
    algorithm. The proposal distribution comes from the maximal coupling
    of a pair of Gaussian random variables, guaranteeing a finite meeting
    time with high probability.
    
    Args:
        Q1: State of the first Markov chain
        Q2: State of the second Markov chain
        std: Standard deviation of the proposal Gaussians.
        marginal: Target marginal distribution
        key: Jax pseudorandom number key
    
    Returns:
        The updated pair of states for the Markov chains 
    """

    key1, key2 = jax.random.split(key, 2)
    Q1_proposed, Q2_proposed = sample_gaussian_max_coupling(Q1, Q2, std, key1)

    # Note, these values may be over 1, but it does not impact the decision
    Q1_acceptance_probability = marginal(Q1_proposed)/marginal(Q1)
    Q2_acceptance_probability = marginal(Q2_proposed)/marginal(Q2)

    U = jax.random.uniform(key2)

    Q1_out = jax.lax.select(U < Q1_acceptance_probability,
                            Q1_proposed, Q1)

    Q2_out = jax.lax.select(U < Q2_acceptance_probability,
                            Q2_proposed, Q2)

    return (Q1_out, Q2_out)
