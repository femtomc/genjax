"""
Test cases for GenJAX cross-module integration.

These tests validate how different GenJAX modules work together:
- ADEV + Inference algorithms (MCMC, SMC, VI)
- Complex hierarchical models
- End-to-end workflows
- Multi-level model composition

Note: GP + Inference integration tests have been disabled (GP module removed)
"""

import jax.numpy as jnp
import jax.random as jrand

from genjax import gen
from genjax.core import sel
from genjax.pjax import seed
from genjax.distributions import normal, exponential, beta, flip
from genjax.adev import expectation
from genjax import (
    normal_reparam,
    normal_reinforce,
    flip_enum,
)

# GP module removed - GP integration tests disabled
from genjax.inference.mcmc import mh


# =============================================================================
# ADEV + INFERENCE INTEGRATION TESTS
# =============================================================================


class TestADEVInferenceIntegration:
    """Test ADEV gradient estimators with inference algorithms."""

    def test_adev_mcmc_integration(self):
        """Test ADEV estimators work with MCMC algorithms."""

        @gen
        def target_model():
            x = normal_reparam(0.0, 1.0) @ "x"
            y = normal_reparam(x, 0.5) @ "y"
            return x + y

        @gen
        def proposal_model(trace):
            # MCMC proposal using ADEV
            x_old = trace.get_choices()["x"]
            x_new = normal_reparam(x_old, 0.1) @ "x"  # Random walk proposal
            y = normal_reparam(x_new, 0.5) @ "y"
            return x_new + y

        # Initial trace
        key = jrand.key(42)
        initial_trace = seed(target_model.simulate)(key)

        # MCMC step with ADEV
        def mcmc_step(trace, key):
            return mh(trace, sel("x"))

        # Run a few MCMC steps (avoiding scan for now due to PJAX compilation issues)
        current_trace = initial_trace
        for i in range(3):  # Just a few steps to test integration
            current_trace = mcmc_step(current_trace, jrand.split(key, i + 1)[0])

        final_trace = current_trace

        # Should produce valid traces
        assert hasattr(final_trace, "get_choices")
        assert "x" in final_trace.get_choices()
        assert "y" in final_trace.get_choices()

        # Should produce different results than initial (basic sanity check)
        assert (
            final_trace.get_choices()["x"] != initial_trace.get_choices()["x"]
            or final_trace.get_choices()["y"] != initial_trace.get_choices()["y"]
        )

    def test_adev_variational_inference(self):
        """Test ADEV estimators in variational inference."""

        @gen
        def target_model():
            x = normal(0.0, 1.0) @ "x"
            y = normal(x, 0.5) @ "y"
            # Observed data
            normal(y, 0.1) @ "obs"

        @gen
        def variational_family(theta):
            # Variational family with ADEV
            mu, log_sigma = theta[:1], theta[1:]
            sigma = jnp.exp(log_sigma)
            x = normal_reparam(mu[0], sigma[0]) @ "x"
            y = normal_reparam(x, sigma[0]) @ "y"

        @expectation
        def elbo(data, theta):
            tr = variational_family.simulate(theta)
            q_score = tr.get_score()
            p_score, _ = target_model.assess({**data, **tr.get_choices()})
            return p_score + q_score

        # Test ELBO computation
        data = {"obs": 1.0}
        theta = jnp.array([0.0, 0.0])  # Initial parameters

        # Should compute ELBO without errors
        elbo_value = elbo.estimate(data, theta)
        assert jnp.isfinite(elbo_value)

        # Should compute gradients
        grad_data, grad_theta = elbo.grad_estimate(data, theta)
        assert jnp.all(jnp.isfinite(grad_theta))

    def test_mixed_adev_estimators(self):
        """Test mixing different ADEV estimators in one model."""

        @gen
        def mixed_model(theta):
            # Mix of REPARAM, REINFORCE, and ENUM
            continuous = normal_reparam(theta[0], 1.0) @ "continuous"
            discrete = flip_enum(theta[1]) @ "discrete"
            hierarchical = normal_reinforce(continuous, 0.5) @ "hierarchical"
            return continuous + jnp.float32(discrete) + hierarchical

        @expectation
        def objective(theta):
            tr = mixed_model.simulate(theta)
            return jnp.sum(tr.get_retval()) ** 2

        theta = jnp.array([0.5, 0.7])

        # Should work with mixed estimators
        value = objective.estimate(theta)
        assert jnp.isfinite(value)

        grad = objective.grad_estimate(theta)
        assert jnp.all(jnp.isfinite(grad))
        assert grad.shape == theta.shape


# =============================================================================
# COMPLEX HIERARCHICAL MODEL TESTS
# =============================================================================


class TestComplexHierarchicalModels:
    """Test complex hierarchical probabilistic models."""

    def test_multilevel_regression(self):
        """Test multilevel regression with ADEV and MCMC."""

        # Synthetic grouped data
        n_groups = 3
        n_per_group = 10

        @gen
        def multilevel_model():
            # Population-level parameters
            mu_alpha = normal(0.0, 1.0) @ "mu_alpha"
            sigma_alpha = exponential(1.0) @ "sigma_alpha"
            mu_beta = normal(0.0, 1.0) @ "mu_beta"
            sigma_beta = exponential(1.0) @ "sigma_beta"
            sigma_y = exponential(1.0) @ "sigma_y"

            # Group-level parameters
            alphas = []
            betas = []
            for g in range(n_groups):
                alpha = normal_reparam(mu_alpha, sigma_alpha) @ f"alpha_{g}"
                beta = normal_reparam(mu_beta, sigma_beta) @ f"beta_{g}"
                alphas.append(alpha)
                betas.append(beta)

            # Observations
            for g in range(n_groups):
                for i in range(n_per_group):
                    x = i / n_per_group  # Predictor
                    mu = alphas[g] + betas[g] * x
                    normal(mu, sigma_y) @ f"y_{g}_{i}"

        # Should simulate hierarchical structure
        key = jrand.key(131415)
        trace = seed(multilevel_model.simulate)(key)
        choices = trace.get_choices()

        # Check population parameters
        assert "mu_alpha" in choices
        assert "sigma_alpha" in choices
        assert "mu_beta" in choices
        assert "sigma_beta" in choices

        # Check group parameters
        for g in range(n_groups):
            assert f"alpha_{g}" in choices
            assert f"beta_{g}" in choices

        # Check observations
        for g in range(n_groups):
            for i in range(n_per_group):
                assert f"y_{g}_{i}" in choices

    def test_mixture_model_with_selection(self):
        """Test mixture models with address selection."""

        @gen
        def mixture_component_1():
            return normal(0.0, 1.0) @ "value"

        @gen
        def mixture_component_2():
            return normal(5.0, 1.0) @ "value"

        @gen
        def mixture_model():
            # Mixture weights
            p = beta(2.0, 2.0) @ "mixture_weight"

            # Component selection
            component = flip(p) @ "component"

            # Conditional execution with same addresses
            if component:
                value = mixture_component_1() @ "comp"
            else:
                value = mixture_component_2() @ "comp"

            return value

        # Test simulation
        key = jrand.key(161718)
        trace = seed(mixture_model.simulate)(key)
        choices = trace.get_choices()

        assert "mixture_weight" in choices
        assert "component" in choices
        assert "comp" in choices
        assert "value" in choices["comp"]

        # Test regeneration with selection
        selection = sel("component")
        new_trace, weight, discarded = mixture_model.regenerate(trace, selection)

        # Should regenerate component selection
        assert "component" in discarded
        new_choices = new_trace.get_choices()
        assert "mixture_weight" in new_choices  # Should be preserved
        assert "comp" in new_choices  # Should be regenerated


# =============================================================================
# END-TO-END WORKFLOW TESTS
# =============================================================================


class TestEndToEndWorkflows:
    """Test complete probabilistic programming workflows."""

    def test_posterior_predictive_checks(self):
        """Test posterior predictive checking workflow."""

        # Simple regression model
        n_data = 20
        x_data = jnp.linspace(0, 1, n_data)
        true_slope = 2.0
        true_noise = 0.2
        key = jrand.key(282930)
        y_data = true_slope * x_data + true_noise * jrand.normal(key, (n_data,))

        @gen
        def regression_model(predict=False, x_pred=None):
            slope = normal(0.0, 2.0) @ "slope"
            noise = exponential(1.0) @ "noise"

            if predict and x_pred is not None:
                # Posterior predictive
                predictions = []
                for i, x in enumerate(x_pred):
                    mu = slope * x
                    pred = normal(mu, noise) @ f"pred_{i}"
                    predictions.append(pred)
                return jnp.array(predictions)
            else:
                # Fit to data
                for i, (x, y) in enumerate(zip(x_data, y_data)):
                    mu = slope * x
                    normal(mu, noise) @ f"obs_{i}"
                return slope

        # Posterior sampling via MCMC
        data_dict = {f"obs_{i}": y for i, y in enumerate(y_data)}

        # Initial trace
        key = jrand.key(313233)
        initial_trace = seed(regression_model.simulate)(key)
        choices = {
            **data_dict,
            **{
                k: v
                for k, v in initial_trace.get_choices().items()
                if k not in data_dict
            },
        }
        initial_trace = regression_model.generate(choices)[0]

        # MCMC proposal
        @gen
        def mcmc_proposal(trace):
            old_choices = trace.get_choices()
            slope = normal(old_choices["slope"], 0.1) @ "slope"
            noise = exponential(1.0) @ "noise"  # Resample from prior

            for i in range(n_data):
                x = x_data[i]
                mu = slope * x
                normal(mu, noise) @ f"obs_{i}"

        # Run MCMC
        def mcmc_step(trace, key):
            return mh(trace, sel("slope") | sel("noise"))

        # Few MCMC steps
        keys = jrand.split(key, 5)
        traces = [initial_trace]

        for k in keys:
            new_trace, accepted = mcmc_step(traces[-1], k)
            traces.append(new_trace)

        # Posterior predictive
        x_pred = jnp.linspace(0, 1.2, 10)  # Slightly extrapolate

        posterior_preds = []
        for trace in traces[-3:]:  # Use last few samples
            choices = trace.get_choices()
            pred_trace = seed(regression_model.simulate)(
                key, predict=True, x_pred=x_pred
            )
            # Condition on posterior parameters
            slope_val = choices["slope"]
            noise_val = choices["noise"]

            pred_choices = {**pred_trace.get_choices()}
            pred_choices["slope"] = slope_val
            pred_choices["noise"] = noise_val

            pred_trace = regression_model.generate(pred_choices)[0]
            posterior_preds.append(pred_trace.get_retval())

        # Should have posterior predictive samples
        assert len(posterior_preds) == 3
        for pred in posterior_preds:
            assert pred.shape == (10,)
            assert jnp.all(jnp.isfinite(pred))
