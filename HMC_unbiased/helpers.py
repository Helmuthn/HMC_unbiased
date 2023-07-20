from jaxtyping import Float, Array, Key
from typing import Callable
import jax.numpy as jnp

import jax
from functools import partial

def construct_potential(marginal: Callable[[Float[Array, " dim"]], Float]):
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
        Magnitude of the elements of the gradient is clipped to 10 to prevent
            issues with numerical instabilities.
    """
    potential = lambda x: -1 * jnp.log(marginal(x))
    potential_grad = jax.grad(potential)
    potential_grad_clipped = lambda x: jnp.clip(potential_grad(x), -10, 10)
    return jax.jit(potential), jax.jit(potential_grad_clipped)


@jax.jit
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


@partial(jax.jit, static_argnums=(2,))
def leapfrog_step(p: Float[Array, " dim"], 
                  q: Float[Array, " dim"], 
                  potential_grad: Callable[[Float[Array, " dim"]], 
                                            Float[Array, " dim"]], 
                  step_size: Float
                ) -> tuple[Float[Array, " dim"], Float[Array, " dim"]]:
    """Completes a step of leapfrog integation.
    
    Args:
        p: Momentum of the system
        q: Position of the system
        potential_grad: Gradient of the potential function
        step_size: Integrator Step Size
     
    Returns:
        A tuple `(p_out, q_out)`, where `p_out` and `q_out` are the 
            updated momentum and position, respectively.
    """
    p_mid = p - step_size/2 * potential_grad(q)
    q_out = q + step_size * p_mid
    p_out = p_mid + step_size/2 * potential_grad(q_out)

    return (p_out, q_out)


@partial(jax.jit, static_argnums=(2,))
def hamiltonian(p: Float[Array, " dim"], 
                q: Float[Array, " dim"], 
                potential: Callable[[Float[Array, " dim"]], Float]
            ) -> Float:
    """Computes the Hamiltonian given the potential."""
    return potential(q) + jnp.sum(jnp.square(p))/2


@partial(jax.jit, static_argnums=(4,))
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

    return jax.numpy.clip(jnp.exp(old_energy - new_energy), a_max=1)


@jax.jit
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

    def first_path(args):
        x_proposed, x_prob, y_prob, key = args
        return x_proposed, x_proposed
    
    def second_path(args):
        x_proposed, x_prob, y_prob, key = args
        W = -1

        def condition(args):
            y_proposed, x_prob, y_prob, W, key = args
            return W <= x_prob

        def inner(args):
            y_proposed, x_prob, y_prob, W, key = args

            subkey, key = jax.random.split(key)
            y_proposed = y + std * jax.random.normal(subkey, y.shape)
            x_prob = isotropic_gaussian_pdf(y_proposed, x, std)
            y_prob = isotropic_gaussian_pdf(y_proposed, y, std)
            subkey, key = jax.random.split(key)
            W = y_prob * jax.random.uniform(subkey)
            return (y_proposed, x_prob, y_prob, W, key)

        inner_args = (x_proposed, x_prob, y_prob, W, key)
        y_proposed, x_prob, y_prob, W, key = jax.lax.while_loop(
            condition, inner, inner_args
        )

        return x_proposed, y_proposed

    args = (x_proposed, x_prob, y_prob, key)
    out = jax.lax.cond(W < y_prob, first_path, second_path, args)
    return out




@partial(jax.jit, static_argnums=(3,4))
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
    def inner(i, args):
        p, q = args
        return leapfrog_step(p, 
                             q, 
                             potential_grad, 
                             step_size)
    p_proposed, q_proposed = jax.lax.fori_loop(0, num_steps, inner, (p, q))

    accept = U < HMC_acceptance(p, q, p_proposed, q_proposed, potential)

    return jax.lax.select(accept, q_proposed, q)