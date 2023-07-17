from HMC_unbiased.helpers import leapfrog_step, isotropic_gaussian_pdf, hamiltonian
from HMC_unbiased.helpers import HMC_acceptance


import jax.numpy as jnp
from pytest import approx

class Test_isotropic_gaussian_pdf:
    def test_standard_center(self):
        x = jnp.ones(1)
        mu = jnp.ones(1)
        std = 1

        test_value = isotropic_gaussian_pdf(x, mu, std)
        assert test_value == approx(1./jnp.sqrt(2*jnp.pi))


class Test_leapfrog_step:
    def test_flat(self):
        potential_grad = lambda x: 0
        p_in = jnp.ones(2)
        q_in = 2 * jnp.ones(2)
        stepsize = 2

        p_true = p_in
        q_true = stepsize * p_in + q_in

        p_test, q_test = leapfrog_step(p_in, q_in, potential_grad, stepsize)

        assert p_test == approx(p_true)
        assert q_test == approx(q_true)

    def test_quadratic(self):
        potential_grad = lambda x: x
        p_in = jnp.ones(2)
        q_in = 2 * jnp.ones(2)
        stepsize = 2

        q_true = q_in + stepsize * p_in - stepsize**2/2 * q_in
        p_true = p_in - stepsize/2 * (q_in + q_true) 

        p_test, q_test = leapfrog_step(p_in, q_in, potential_grad, stepsize)

        assert p_test == approx(p_true)
        assert q_test == approx(q_true)

class Test_hamiltonian:
    def test_quadratic(self):
        potential = lambda x: jnp.sum(jnp.square(x))

        p_in = jnp.ones(2)
        q_in = 2*jnp.ones(2)

        truth = 9
        test = hamiltonian(p_in, q_in, potential)

        assert truth == approx(test)

class Test_HMC_acceptance:
    def test_flat(self):
        potential = lambda x: 0
        p_in = jnp.ones(2)
        q_in = jnp.ones(2)
        p_proposed_in = 2 * jnp.ones(2)
        q_proposed_in = jnp.ones(2)

        truth = jnp.exp(-3)
        test = HMC_acceptance(p_in, q_in, 
                              p_proposed_in, q_proposed_in, 
                              potential)

        assert test == approx(truth)
        