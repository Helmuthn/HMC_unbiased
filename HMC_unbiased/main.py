import jax
from jaxtyping import Float, Array, Key
from typing import Callable

from helpers import sample_gaussian_max_coupling, HMC_step


def coupled_HMC_step(Q1: Float[Array, " dim"], 
                     Q2: Float[Array, " dim"], 
                     potential: Callable[[Float[Array, " dim"]], Float],
                     potential_grad: Callable[[Float[Array, " dim"]], 
                                    Float[Array, " dim"]], 
                     stepsize: Float, 
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
        stepsize: Stepsize for leapfrog integration
        key: Jax pseudorandom number key
    
    Returns:
        Updated values for `Q1` and `Q2`.
    
    Notes:
        See Algorithm 1 in https://arxiv.org/abs/1709.00404 for details.
    """
    key1, key2 = jax.random.split(key, 2)
    momentum = jax.random.normal(key1, Q1.shape)
    U = jax.random.uniform(key2)

    Q1_out = HMC_step(momentum, Q1, U, potential, potential_grad, stepsize)
    Q2_out = HMC_step(momentum, Q2, U, potential, potential_grad, stepsize)
    
    return Q1_out, Q2_out



def coupled_randomwalk_step(x: Float[Array, " dim"], 
                            y: Float[Array, " dim"], 
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
        x: State of the first Markov chain
        y: State of the second Markov chain
        std: Standard deviation of the proposal Gaussians.
        marginal: Target marginal distribution
        key: Jax pseudorandom number key
    
    Returns:
        The updated pair of states for the Markov chains 
    """

    key1, key2 = jax.random.split(key, 2)
    x_proposed, y_proposed = sample_gaussian_max_coupling(x, y, std, key1)

    x_acceptance_probability = min(1, marginal(x_proposed)/marginal(x))
    y_acceptance_probability = min(1, marginal(y_proposed)/marginal(y))

    U = jax.random.uniform(key2)

    if U < x_acceptance_probability:
        x_out = x_proposed
    else:
        x_out = x

    if U < y_acceptance_probability:
        y_out = y_proposed
    else:
        y_out = y

    return x_out, y_out