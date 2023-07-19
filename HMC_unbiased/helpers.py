from jaxtyping import Float, Array, Key
from typing import Callable
import jax.numpy as jnp

import jax

def construct_potential(marginal: Callable[[Float[Array, " dim"]], Float]
            ) -> tuple[Callable[[Float[Array, " dim"]], Float],
                       Callable[[Float[Array, " dim"]], Float[Array, " dim"]]]:
    """Convert an unnormalized marginal distribution into a potential function.
    
    Given an unnormalized marginal distribution, return a potential function
    and the gradient of the potential function.

    Args:
        marginal: Marginal distribution

    Returns:
        A tuple `(potential, potential_grad)` of jit compiled functions 
        representing the potential (log) of the distributiona and the
        gradient of the potential.

    Warning:
        Adds a small value to the marginal before computing the log in order to
        stablize the computation. This results in incorrect derivatives for
        regions with low probability.
    """
    potential = lambda x: jnp.log(marginal(x) + 1e-32)
    potential_grad = jax.grad(potential)
    return jax.jit(potential), jax.jit(potential_grad)


def isotropic_gaussian_pdf(x: Float[Array, " dim"], 
                           mu: Float[Array, " dim"],
                           std: Float) -> Float:
    """Compute the pdf of an isotropic Gaussian.
    
    Args:
        x: Evaluation point for the distribution
        mu: Mean of the distribution
        std: Standard deviation for each axis
    
    Returns:
        The evaluation of the multivariate Gaussian pdf at a point
    """
    squared_distance = -jnp.sum(jnp.square(x - mu))
    normalized_distance = squared_distance / (2 * std**2)
    density = jnp.exp(normalized_distance)/ (std**x.shape[0] * (2*jnp.pi)**(x.shape[0]/2))
    return density


def leapfrog_step(p: Float[Array, " dim"], 
                  q: Float[Array, " dim"], 
                  potential_grad: Callable[[Float[Array, " dim"]], 
                                    Float[Array, " dim"]], 
                  stepsize: Float
                ) -> tuple[Float[Array, " dim"], Float[Array, " dim"]]:
    """Completes a step of leapfrog integation.
    
    Args:
        p: Momentum of the system
        q: Position of the system
        potential_grad: Gradient of the potential function
        stepsize: Integrator Step Size
     
    Returns:
        `(p_out, q_out)` where `p_out` and `q_out` are the 
        updated momentum and position.
    """
    p_mid = p - stepsize/2 * potential_grad(q)
    q_out = q + stepsize * p_mid
    p_out = p_mid + stepsize/2 * potential_grad(q_out)

    return (p_out, q_out)


def hamiltonian(p: Float[Array, " dim"], 
                q: Float[Array, " dim"], 
                potential: Callable[[Float[Array, " dim"]], Float]
            ) -> Float:
    """Computes the Hamiltonian given the potential."""
    return potential(q) + jnp.sum(jnp.square(p))/2


def HMC_acceptance(p: Float[Array, " dim"], 
                   q: Float[Array, " dim"], 
                   p_proposed: Float[Array, " dim"], 
                   q_proposed: Float[Array, " dim"], 
                   potential: Callable[[Float[Array, " dim"]], Float]
                ) -> Float:
    """ Compute the acceptance probability of the HMC proposal.
    
    Args:
        p: Original momentum 
        q: Original position
        p_proposed: Proposed momentum
        q_proposed: Proposed position
        potential: Potential function
    
    Returns:
        The probability of acceptance in Hamiltonian Monte Carlo.
    """
    old_energy = hamiltonian(p, q, potential)
    new_energy = hamiltonian(p_proposed, q_proposed, potential)

    return min(1, jnp.exp(old_energy - new_energy))


def sample_gaussian_max_coupling(x: Float[Array, " dim"], 
                                 y: Float[Array, " dim"], 
                                 std: Float, 
                                 key: Key
    )-> tuple[Float[Array, " dim"], Float[Array, " dim"]] :
    """Sample from a max coupling of a pair of Gaussians
    
    Given two centers and a shared standard deviation, generate
    a sample from the maximally coupled distribution with marginal
    distributions of the two gaussians.
    
    Args:
        x: Center of the first Gaussian
        y: Center of the second Gaussian
        std: Standard deviation of the distributions
        key: Jax pseudorandom number key
    
    Returns:
        A realization of a pair of Gaussian random variables 
        with maximal coupling.
    """
    subkey, key = jax.random.split(key)
    x_proposed = x + std*jax.random.normal(subkey, x.shape)
    x_prob = isotropic_gaussian_pdf(x_proposed, x, std)
    y_prob = isotropic_gaussian_pdf(x_proposed, y, std)

    subkey, key = jax.random.split(key)
    W = x_prob * jax.random.uniform(subkey)

    if W < y_prob:
        return x_proposed, x_proposed
    else:
        W = -1
        while W <= x_prob:
            subkey, key = jax.random.split(key)
            y_proposed = y + std * jax.random.normal(subkey, y.shape)
            x_prob = isotropic_gaussian_pdf(y_proposed, x, std)
            y_prob = isotropic_gaussian_pdf(y_proposed, y, std)
            subkey, key = jax.random.split(key)
            W = y_prob * jax.random.uniform(subkey)

        return x_proposed, y_proposed


def HMC_step(p: Float[Array, " dim"], 
             q: Float[Array, " dim"], 
             U: Float, 
             potential: Callable[[Float[Array, " dim"]], Float], 
             potential_grad: Callable[[Float[Array, " dim"]], 
                                       Float[Array, " dim"]],  
             step_size: Float,
             num_steps: int
        ) -> Float[Array, " dim"]:
    """Given the random variables, compute a step of Hamiltonian Monte Carlo.
    
    Args:
        p: Current momentum
        q: Current position
        U: Uniform random variable to determine acceptance
        potential: Potential function for target distribution
        potential_grad: Gradient of the potential function
        step_size: Step size for the leap-frog integration
        num_steps: Number of steps for the leap-frog integration
    
    Returns:
        A new state variable from a single step of Hamiltonian Monte Carlo.
    """
    p_proposed, q_proposed = p, q
    for i in range(num_steps):
        p_proposed, q_proposed = leapfrog_step(p_proposed, 
                                               q_proposed, 
                                               potential_grad, 
                                               step_size)

    if U < HMC_acceptance(p, q, p_proposed, q_proposed, potential):
        return q_proposed
    else:
        return q