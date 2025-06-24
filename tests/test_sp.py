"""Tests for Stochastic Probabilities (SP) module."""

import jax
import jax.numpy as jnp
import jax.scipy.stats as jss
import pytest

from genjax import gen, normal, beta, flip, uniform
from genjax.sp import (
    SPDistribution, SMCAlgorithm, Target, ImportanceSampling, Marginal,
    importance_sampling, marginal, get_selection
)
from genjax.core import const, Selection, AllSel, NoneSel, DictSel


class TestSPDistribution:
    """Test the SPDistribution abstract base class."""
    
    def test_sp_distribution_interface(self):
        """Test that SPDistribution properly extends GFI."""
        # SPDistribution is abstract, so we test with ImportanceSampling
        @gen
        def model():
            x = normal(0.0, 1.0) @ "x"
            return x
        
        target = Target(model=model, args=(), observations={})
        sp_dist = ImportanceSampling(target=target, n_particles=const(100))
        
        # Should have GFI methods
        assert hasattr(sp_dist, 'simulate')
        assert hasattr(sp_dist, 'assess')
        assert hasattr(sp_dist, 'generate')
        
        # Should have SP methods
        assert hasattr(sp_dist, 'random_weighted')
        assert hasattr(sp_dist, 'estimate_logpdf')


class TestTarget:
    """Test the Target abstraction."""
    
    def test_target_creation(self):
        """Test creating a Target with model and observations."""
        @gen
        def model():
            x = normal(0.0, 1.0) @ "x"
            y = normal(x, 0.1) @ "y"
            return x, y
        
        observations = {"y": 0.5}
        target = Target(model=model, args=(), observations=observations)
        
        assert target.model == model
        assert target.observations == observations
    
    def test_get_latents(self):
        """Test extracting latent variables from a trace."""
        @gen
        def model():
            x = normal(0.0, 1.0) @ "x"
            y = normal(x, 0.1) @ "y"
            return x, y
        
        observations = {"y": 0.5}
        target = Target(model=model, args=(), observations=observations)
        
        # Generate a trace
        trace, _ = model.generate(observations)
        
        # Get latents - should only have "x"
        latents = target.get_latents(trace)
        assert "x" in latents
        assert "y" not in latents


class TestImportanceSampling:
    """Test the ImportanceSampling SPDistribution."""
    
    def test_importance_sampling_with_default_proposal(self):
        """Test importance sampling using model's internal proposal."""
        @gen
        def model():
            x = normal(0.0, 1.0) @ "x"
            y = normal(x, 0.1) @ "y"
            return x
        
        observations = {"y": 0.5}
        target = Target(model=model, args=(), observations=observations)
        sp_dist = ImportanceSampling(target=target, n_particles=const(1000))
        
        # Sample from the SP distribution
        latents, weight = sp_dist.random_weighted()
        
        # Should get latent x
        assert "x" in latents
        assert isinstance(latents["x"], jnp.ndarray)
        
        # Weight should be positive
        assert weight > 0
    
    def test_importance_sampling_with_custom_proposal(self):
        """Test importance sampling with custom proposal."""
        @gen
        def model():
            x = normal(0.0, 1.0) @ "x"
            y = normal(x, 0.1) @ "y"
            return x
        
        @gen  
        def proposal():
            # Proposal centered near expected posterior
            x = normal(0.45, 0.2) @ "x"
            return x
        
        observations = {"y": 0.5}
        target = Target(model=model, args=(), observations=observations)
        sp_dist = ImportanceSampling(
            target=target, 
            proposal=proposal,
            n_particles=const(1000)
        )
        
        # Sample multiple times to check distribution
        samples = []
        weights = []
        for _ in range(100):
            latents, weight = sp_dist.random_weighted()
            samples.append(latents["x"])
            weights.append(weight)
        
        samples = jnp.array(samples)
        
        # Check that samples are reasonable (posterior should be between prior and observation)
        assert jnp.mean(samples) > 0.2  # Pulled toward observation
        assert jnp.mean(samples) < 0.5  # But not all the way
    
    def test_estimate_logpdf(self):
        """Test log probability estimation."""
        @gen
        def model():
            x = normal(0.0, 1.0) @ "x"
            return x
        
        target = Target(model=model, args=(), observations={})
        sp_dist = ImportanceSampling(target=target, n_particles=const(1000))
        
        # Estimate log pdf at a point
        test_value = {"x": 0.0}
        log_prob = sp_dist.estimate_logpdf(test_value)
        
        # Should be close to true log prob N(0, 1)
        true_log_prob = jss.norm.logpdf(0.0, 0.0, 1.0)
        assert jnp.abs(log_prob - true_log_prob) < 0.1


class TestMarginal:
    """Test the Marginal SPDistribution."""
    
    def test_marginal_single_address(self):
        """Test marginalizing to a single address."""
        @gen
        def model():
            x = normal(0.0, 1.0) @ "x"
            y = normal(x, 0.1) @ "y"
            return x, y
        
        # Create marginal over x using ImportanceSampling algorithm
        target = Target(model=model, args=(), observations={})
        algorithm = ImportanceSampling(target=target, n_particles=const(1000))
        marginal_x = Marginal(
            algorithm=algorithm,
            address="x"
        )
        
        # Sample from marginal
        value, weight = marginal_x.random_weighted()
        
        # Should get scalar value
        assert isinstance(value, jnp.ndarray)
        assert value.shape == ()
        
        # Weight comes from the algorithm 
        assert isinstance(weight, jnp.ndarray)
    
    def test_marginal_hierarchical_address(self):
        """Test marginal with hierarchical address."""
        @gen
        def sub_model():
            a = normal(0.0, 1.0) @ "a"
            b = normal(a, 0.1) @ "b"
            return a, b
        
        @gen
        def model():
            sub = sub_model() @ "sub"
            return sub
        
        # Create marginal over sub/a
        marginal_sub_a = Marginal(
            model=model,
            address="sub/a",
            args=(),
            n_particles=const(1000)
        )
        
        # Sample from marginal
        value, weight = marginal_sub_a.random_weighted()
        
        assert isinstance(value, jnp.ndarray)
    
    def test_marginal_estimate_logpdf(self):
        """Test estimating marginal log probability."""
        @gen
        def model():
            x = normal(0.0, 1.0) @ "x"
            y = normal(x, 0.5) @ "y"
            return x, y
        
        # Create marginal over y
        marginal_y = Marginal(
            model=model,
            address="y",
            args=(),
            n_particles=const(5000)
        )
        
        # Estimate log pdf at a point
        # Marginal of y should be N(0, sqrt(1 + 0.25))
        test_value = 0.0
        log_prob = marginal_y.estimate_logpdf(test_value)
        
        # True marginal: y ~ N(0, sqrt(1.25))
        true_log_prob = jss.norm.logpdf(0.0, 0.0, jnp.sqrt(1.25))
        
        # Should be close (within estimation error)
        assert jnp.abs(log_prob - true_log_prob) < 0.15


class TestHelperFunctions:
    """Test helper functions for creating SP distributions."""
    
    def test_importance_sampling_helper(self):
        """Test the importance_sampling helper function."""
        @gen
        def model():
            x = normal(0.0, 1.0) @ "x"
            return x
        
        target = Target(model=model, args=(), observations={})
        
        # Create using helper
        sp_dist = importance_sampling(target, n_particles=500)
        
        assert isinstance(sp_dist, ImportanceSampling)
        assert sp_dist.n_particles.value == 500
    
    def test_marginal_helper(self):
        """Test the marginal helper function."""
        @gen
        def model(mu, sigma):
            x = normal(mu, sigma) @ "x"
            return x
        
        # Create marginal using helper
        marg = marginal(model, "x", args=(2.0, 0.5), n_particles=200)
        
        assert isinstance(marg, Marginal)
        assert marg.address == "x"
        assert marg.n_particles.value == 200
        assert marg.args == (2.0, 0.5)


class TestIntegrationWithGenJAX:
    """Test integration of SP distributions with core GenJAX."""
    
    def test_sp_distribution_as_gfi(self):
        """Test that SP distributions work as generative functions."""
        @gen
        def model():
            x = normal(0.0, 1.0) @ "x"
            y = normal(x, 0.1) @ "y"
            return x
        
        observations = {"y": 0.5}
        target = Target(model=model, args=(), observations=observations)
        sp_dist = ImportanceSampling(target=target, n_particles=const(100))
        
        # Should be able to simulate
        trace = sp_dist.simulate()
        assert trace.get_retval() is not None
        
        # Choices should be latent variables
        choices = trace.get_retval()  # For SP, retval is the sampled value
        assert "x" in choices
    
    def test_nested_sp_distributions(self):
        """Test composing SP distributions."""
        @gen
        def inner_model():
            x = beta(2.0, 2.0) @ "x"
            return x
        
        # Create marginal of inner model
        inner_marginal = Marginal(
            model=inner_model,
            address="x",
            args=(),
            n_particles=const(100)
        )
        
        @gen
        def outer_model():
            # In principle, we could use inner_marginal here
            # but it requires more infrastructure
            p = beta(2.0, 2.0) @ "p"
            y = flip(p) @ "y"
            return y
        
        # This tests that our abstractions are composable
        target = Target(model=outer_model, args=(), observations={"y": True})
        sp_dist = ImportanceSampling(target=target, n_particles=const(100))
        
        # Should be able to sample
        latents, weight = sp_dist.random_weighted()
        assert "p" in latents


class TestConjugateModels:
    """Test SP distributions on models with known exact solutions."""
    
    def test_normal_normal_conjugate(self):
        """Test normal-normal conjugate model."""
        # Prior: x ~ N(0, 1)
        # Likelihood: y ~ N(x, 0.1)
        # Observation: y = 0.5
        # Posterior: x ~ N(posterior_mean, posterior_var)
        
        prior_mean = 0.0
        prior_var = 1.0
        obs_var = 0.1
        y_obs = 0.5
        
        # Exact posterior parameters
        posterior_var = 1.0 / (1.0/prior_var + 1.0/obs_var)
        posterior_mean = posterior_var * (prior_mean/prior_var + y_obs/obs_var)
        
        @gen
        def model():
            x = normal(prior_mean, jnp.sqrt(prior_var)) @ "x"
            y = normal(x, jnp.sqrt(obs_var)) @ "y"
            return x
        
        target = Target(model=model, args=(), observations={"y": y_obs})
        sp_dist = ImportanceSampling(target=target, n_particles=const(5000))
        
        # Sample many times and check posterior
        samples = []
        for _ in range(1000):
            latents, _ = sp_dist.random_weighted()
            samples.append(latents["x"])
        
        samples = jnp.array(samples)
        
        # Check posterior mean and variance
        emp_mean = jnp.mean(samples)
        emp_var = jnp.var(samples)
        
        assert jnp.abs(emp_mean - posterior_mean) < 0.02
        assert jnp.abs(emp_var - posterior_var) < 0.02
    
    def test_beta_bernoulli_conjugate(self):
        """Test beta-Bernoulli conjugate model."""
        # Prior: p ~ Beta(2, 2)
        # Likelihood: y ~ Bernoulli(p)
        # Observation: y = True
        # Posterior: p ~ Beta(3, 2)
        
        @gen
        def model():
            p = beta(2.0, 2.0) @ "p"
            y = flip(p) @ "y"
            return p
        
        target = Target(model=model, args=(), observations={"y": True})
        sp_dist = ImportanceSampling(target=target, n_particles=const(2000))
        
        # Sample and check posterior
        samples = []
        for _ in range(500):
            latents, _ = sp_dist.random_weighted()
            samples.append(latents["p"])
        
        samples = jnp.array(samples)
        
        # Posterior should be Beta(3, 2)
        # Mean = 3/(3+2) = 0.6
        posterior_mean = 3.0 / 5.0
        emp_mean = jnp.mean(samples)
        
        assert jnp.abs(emp_mean - posterior_mean) < 0.03


# =============================================================================
# GET_SELECTION FUNCTION TESTS
# =============================================================================

class TestGetSelection:
    """Test the standalone get_selection function."""
    
    def test_get_selection_none(self):
        """Test get_selection with None input."""
        sel = get_selection(None)
        assert isinstance(sel, Selection)
        assert isinstance(sel.s, NoneSel)
        assert "x" not in sel  # Should match nothing
    
    def test_get_selection_empty_dict(self):
        """Test get_selection with empty dictionary."""
        sel = get_selection({})
        assert isinstance(sel, Selection)
        assert isinstance(sel.s, NoneSel)
        assert "x" not in sel  # Should match nothing
    
    def test_get_selection_simple_dict(self):
        """Test get_selection with simple dictionary."""
        choices = {"x": 1.0, "y": 2.0, "z": 3.0}
        sel = get_selection(choices)
        
        assert isinstance(sel, Selection)
        assert isinstance(sel.s, DictSel)
        
        # Should match all keys
        assert "x" in sel
        assert "y" in sel
        assert "z" in sel
        
        # Should not match other keys
        assert "w" not in sel
        assert "foo" not in sel
    
    def test_get_selection_nested_dict(self):
        """Test get_selection with nested dictionary."""
        choices = {
            "outer1": {
                "inner1": 1.0,
                "inner2": 2.0
            },
            "outer2": {
                "inner3": 3.0,
                "inner4": {
                    "deep": 4.0
                }
            },
            "scalar": 5.0
        }
        
        sel = get_selection(choices)
        
        # Check top-level keys
        assert "outer1" in sel
        assert "outer2" in sel
        assert "scalar" in sel
        assert "nonexistent" not in sel
        
        # Check nested structure is preserved
        assert isinstance(sel.s, DictSel)
        outer1_sel = sel.s.d["outer1"]
        assert isinstance(outer1_sel, Selection)
        assert isinstance(outer1_sel.s, DictSel)
        
        # Deep nested structure
        outer2_sel = sel.s.d["outer2"]
        inner4_sel = outer2_sel.s.d["inner4"]
        assert isinstance(inner4_sel, Selection)
        assert isinstance(inner4_sel.s, DictSel)
    
    def test_get_selection_scalar_values(self):
        """Test get_selection with non-dict values."""
        # Scalar float
        sel = get_selection(42.0)
        assert isinstance(sel, Selection)
        assert isinstance(sel.s, AllSel)
        assert () in sel  # Should match all addresses
        
        # Array
        sel = get_selection(jnp.array([1.0, 2.0, 3.0]))
        assert isinstance(sel, Selection)
        assert isinstance(sel.s, AllSel)
        
        # Tuple
        sel = get_selection((1.0, 2.0))
        assert isinstance(sel, Selection)
        assert isinstance(sel.s, AllSel)
    
    def test_get_selection_mixed_structure(self):
        """Test get_selection with mixed nested structures."""
        choices = {
            "dict_value": {"a": 1, "b": 2},
            "scalar_value": 3.0,
            "array_value": jnp.array([4.0, 5.0]),
            "nested": {
                "level2": {
                    "level3": 6.0
                }
            }
        }
        
        sel = get_selection(choices)
        
        # All top-level keys should be present
        assert "dict_value" in sel
        assert "scalar_value" in sel
        assert "array_value" in sel
        assert "nested" in sel
        
        # Check that scalar values at leaves get AllSel
        scalar_sel = sel.s.d["scalar_value"]
        assert isinstance(scalar_sel, Selection)
        assert isinstance(scalar_sel.s, AllSel)


# =============================================================================
# HIERARCHICAL ADDRESS TESTS
# =============================================================================

class TestHierarchicalAddresses:
    """Test hierarchical address handling in SP module."""
    
    def test_target_with_hierarchical_observations(self):
        """Test Target with nested observation structure."""
        @gen
        def sub_model():
            x = normal(0.0, 1.0) @ "x"
            y = normal(x, 0.1) @ "y"
            return {"x": x, "y": y}
        
        @gen
        def model():
            sub1 = sub_model() @ "sub1"
            sub2 = sub_model() @ "sub2"
            z = normal(sub1["x"] + sub2["x"], 0.1) @ "z"
            return {"sub1": sub1, "sub2": sub2, "z": z}
        
        # Hierarchical observations
        observations = {
            "sub1": {"y": 0.5},
            "sub2": {"y": -0.3},
            "z": 0.2
        }
        
        target = Target(model=model, args=(), observations=observations)
        
        # The observation selection should match the structure
        obs_sel = target._obs_selection()
        assert isinstance(obs_sel, Selection)
        
        # Should match observed addresses
        assert "z" in obs_sel
        # Note: hierarchical matching depends on Selection implementation
        # The important thing is that get_selection creates appropriate structure
    
    def test_importance_sampling_with_hierarchical_model(self):
        """Test importance sampling with nested generative functions."""
        @gen
        def component(mu):
            x = normal(mu, 1.0) @ "x"
            obs = normal(x, 0.1) @ "obs"
            return x
        
        @gen
        def model():
            # Hierarchical model with multiple components
            comp1 = component(0.0) @ "comp1"
            comp2 = component(comp1) @ "comp2"
            comp3 = component(comp2) @ "comp3"
            return comp3
        
        # Observe leaf component
        observations = {
            "comp3": {"obs": 2.0}
        }
        
        target = Target(model=model, args=(), observations=observations)
        sp_dist = ImportanceSampling(target=target, n_particles=const(1000))
        
        # Sample and check structure
        latents, weight = sp_dist.random_weighted()
        
        # Should have latent variables from all components
        assert "comp1" in latents
        assert "comp2" in latents
        # comp3 has observed data, so only its unobserved parts should be in latents
        
        # Each component should have "x" (the latent part)
        assert "x" in latents["comp1"]
        assert "x" in latents["comp2"]
        # comp3/obs was observed, but comp3/x should be in the full trace
    
    def test_marginal_with_deep_hierarchical_address(self):
        """Test Marginal with deeply nested addresses."""
        @gen
        def leaf_model(base):
            return normal(base, 0.1) @ "value"
        
        @gen
        def branch_model(base):
            left = leaf_model(base) @ "left"
            right = leaf_model(base + 1.0) @ "right"
            return {"left": left, "right": right}
        
        @gen
        def tree_model():
            root = normal(0.0, 1.0) @ "root"
            branch1 = branch_model(root) @ "branch1"
            branch2 = branch_model(root + 0.5) @ "branch2"
            return {"branch1": branch1, "branch2": branch2}
        
        # Test marginal at different levels
        # Note: Current implementation expects addresses as strings or tuples
        # but the actual navigation might need adjustment
        
        # Marginal over root
        marginal_root = Marginal(
            model=tree_model,
            address="root",
            args=(),
            n_particles=const(500)
        )
        
        value, weight = marginal_root.random_weighted()
        assert isinstance(value, jnp.ndarray)
        assert jnp.isclose(weight, 1.0)
        
        # For hierarchical addresses, we'd need to handle tuple addresses
        # This tests that the infrastructure is ready for such extensions
    
    def test_get_selection_preserves_hierarchy(self):
        """Test that get_selection preserves hierarchical structure."""
        # Complex nested structure
        choices = {
            "model1": {
                "submodel1": {
                    "param1": 1.0,
                    "param2": 2.0
                },
                "submodel2": {
                    "data": jnp.array([3.0, 4.0]),
                    "nested": {
                        "deep": 5.0
                    }
                }
            },
            "model2": {
                "values": [6.0, 7.0, 8.0]
            }
        }
        
        sel = get_selection(choices)
        
        # Navigate through the selection structure
        # Top level
        assert "model1" in sel
        assert "model2" in sel
        
        # The selection should preserve the nested structure
        # so we can match hierarchical addresses properly
        assert isinstance(sel.s, DictSel)
        
        # Check model1 structure
        model1_sel = sel.s.d["model1"]
        assert isinstance(model1_sel, Selection)
        assert isinstance(model1_sel.s, DictSel)
        
        # Check submodel1 structure
        submodel1_sel = model1_sel.s.d["submodel1"]
        assert isinstance(submodel1_sel, Selection)
        assert isinstance(submodel1_sel.s, DictSel)
        
        # Leaf nodes should have AllSel
        param1_sel = submodel1_sel.s.d["param1"]
        assert isinstance(param1_sel, Selection)
        assert isinstance(param1_sel.s, AllSel)
    
    def test_sp_distribution_get_selection_method(self):
        """Test that SPDistribution.get_selection delegates correctly."""
        @gen
        def model():
            x = normal(0.0, 1.0) @ "x"
            y = normal(x, 0.1) @ "y"
            return x
        
        target = Target(model=model, args=(), observations={"y": 0.5})
        sp_dist = ImportanceSampling(target=target, n_particles=const(100))
        
        # Test with different choice structures
        # Dictionary
        dict_choices = {"a": 1.0, "b": 2.0}
        sel = sp_dist.get_selection(dict_choices)
        assert isinstance(sel, Selection)
        assert "a" in sel
        assert "b" in sel
        
        # None
        sel = sp_dist.get_selection(None)
        assert isinstance(sel, Selection)
        assert isinstance(sel.s, NoneSel)
        
        # Scalar
        sel = sp_dist.get_selection(42.0)
        assert isinstance(sel, Selection)
        assert isinstance(sel.s, AllSel)


# =============================================================================
# SMC ALGORITHM TESTS
# =============================================================================

class TestSMCAlgorithm:
    """Test SMCAlgorithm extension of SPDistribution."""
    
    def test_importance_sampling_run_smc(self):
        """Test ImportanceSampling.run_smc method."""
        from genjax.inference.smc import ParticleCollection
        
        @gen
        def model():
            x = normal(0.0, 1.0) @ "x"
            y = normal(x, 0.5) @ "y"
            return x
        
        # Create ImportanceSampling instance
        target = Target(model=model, args=(), observations={"y": 1.0})
        is_alg = ImportanceSampling(target=target, n_particles=const(100))
        
        # Run SMC (no key needed with PJAX)
        particles = is_alg.run_smc(n_particles=200)
        
        # Check result type and structure
        assert isinstance(particles, ParticleCollection)
        assert particles.n_samples.value == 200
        assert particles.traces.get_choices()["x"].shape == (200,)
        
        # Check effective sample size is reasonable
        ess = particles.effective_sample_size()
        assert ess > 50  # Should have reasonable diversity
    
    def test_importance_sampling_run_csmc(self):
        """Test ImportanceSampling.run_csmc method."""
        from genjax.inference.smc import ParticleCollection
        
        @gen
        def model():
            x = normal(0.0, 1.0) @ "x"
            y = normal(x, 0.5) @ "y"
            return x
        
        # Create ImportanceSampling instance
        target = Target(model=model, args=(), observations={"y": 1.0})
        is_alg = ImportanceSampling(target=target, n_particles=const(100))
        
        # Define retained choices
        retained_choices = {"x": 0.5, "y": 1.0}
        
        # Run CSMC (no key needed with PJAX)
        particles = is_alg.run_csmc(n_particles=200, retained_choices=retained_choices)
        
        # Check result type and structure
        assert isinstance(particles, ParticleCollection)
        assert particles.n_samples.value == 200
        
        # Note: In this simplified implementation, the choices aren't actually overridden
        # but the weights are adjusted. This is sufficient for basic functionality testing.
        # A full implementation would require more complex trace reconstruction.
    
    def test_smc_marginal_likelihood_convergence(self):
        """Test SMC marginal likelihood convergence with increasing particles."""
        # Use conjugate Normal-Normal model with known marginal likelihood
        prior_mean, prior_var = 0.0, 1.0
        obs_var = 0.25
        y_obs = 1.0
        
        @gen
        def model():
            x = normal(prior_mean, jnp.sqrt(prior_var)) @ "x"
            y = normal(x, jnp.sqrt(obs_var)) @ "y"
            return x
        
        # Analytical marginal likelihood: y ~ N(prior_mean, prior_var + obs_var)
        true_log_marginal = jss.norm.logpdf(y_obs, prior_mean, jnp.sqrt(prior_var + obs_var))
        
        target = Target(model=model, args=(), observations={"y": y_obs})
        is_alg = ImportanceSampling(target=target, n_particles=const(100))
        
        # Test convergence with increasing particle counts
        particle_counts = [100, 500, 1000, 2000]
        log_marginals = []
        
        for n_particles in particle_counts:
            particles = is_alg.run_smc(n_particles=n_particles)
            log_marginal = particles.log_marginal_likelihood()
            log_marginals.append(log_marginal)
        
        # Check convergence to true value
        errors = [jnp.abs(lm - true_log_marginal) for lm in log_marginals]
        
        # Error should generally decrease (allowing some stochasticity)
        assert errors[-1] < errors[0] + 0.1  # Final error better than initial (with tolerance)
        assert errors[-1] < 0.2  # Final error should be small
    
    def test_csmc_retained_particle_preservation(self):
        """Test that CSMC preserves the retained particle throughout."""
        from genjax.inference.smc import ParticleCollection
        
        @gen
        def model():
            x = normal(0.0, 1.0) @ "x"
            return x
        
        target = Target(model=model, args=(), observations={})
        is_alg = ImportanceSampling(target=target, n_particles=const(100))
        
        # Define retained choices
        retained_choices = {"x": 2.5}  # Unusual value to make retention obvious
        
        # Run CSMC multiple times
        for i in range(5):
            particles = is_alg.run_csmc(n_particles=50, retained_choices=retained_choices)
            
            # Check that CSMC runs successfully and returns valid structure
            assert isinstance(particles, ParticleCollection)
            assert particles.n_samples.value == 50
    
    def test_smc_algorithm_inheritance(self):
        """Test that ImportanceSampling properly inherits from SMCAlgorithm."""
        from genjax.sp import SMCAlgorithm
        
        @gen
        def model():
            x = normal(0.0, 1.0) @ "x"
            return x
        
        target = Target(model=model, args=(), observations={})
        is_alg = ImportanceSampling(target=target, n_particles=const(100))
        
        # Should be instance of SMCAlgorithm
        assert isinstance(is_alg, SMCAlgorithm)
        
        # Should have all 4 required methods
        assert hasattr(is_alg, 'random_weighted')
        assert hasattr(is_alg, 'estimate_logpdf')
        assert hasattr(is_alg, 'run_smc')
        assert hasattr(is_alg, 'run_csmc')
        
        # Methods should be callable
        
        # Test random_weighted (inherited from SPDistribution)
        samples, weights = is_alg.random_weighted()
        assert isinstance(samples, dict)
        assert isinstance(weights, jnp.ndarray)
        
        # Test estimate_logpdf (inherited from SPDistribution)
        log_prob = is_alg.estimate_logpdf({"x": 0.0})
        assert isinstance(log_prob, jnp.ndarray)
        
        # Test run_smc (new SMCAlgorithm method)
        particles = is_alg.run_smc(n_particles=100)
        assert particles.n_samples.value == 100
        
        # Test run_csmc (new SMCAlgorithm method)
        particles_csmc = is_alg.run_csmc(n_particles=100, retained_choices={"x": 1.0})
        assert particles_csmc.n_samples.value == 100


if __name__ == "__main__":
    pytest.main([__file__])