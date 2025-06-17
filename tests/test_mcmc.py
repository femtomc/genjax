"""
Test cases for GenJAX MCMC inference algorithms.

These tests validate MCMC implementations against analytically known posteriors,
following the same pattern as SMC tests which validate against exact log marginals.
Tests include Metropolis-Hastings implementations.
"""

import jax
import jax.numpy as jnp
import jax.random as jrand
import pytest

from genjax.core import gen, seed, sel, Const
from genjax.distributions import beta, flip, exponential
from genjax.mcmc import (
    MCMCResult,
    metropolis_hastings,
)


# ============================================================================
# MCMC-Specific Fixtures
# ============================================================================


@pytest.fixture
def mcmc_steps_small():
    """Small number of MCMC steps for fast tests."""
    return Const(100)


@pytest.fixture
def mcmc_steps_medium():
    """Medium number of MCMC steps for balanced speed/accuracy."""
    return Const(5000)


@pytest.fixture
def mcmc_steps_large():
    """Large number of MCMC steps for convergence tests."""
    return Const(50000)


@pytest.fixture
def mcmc_key(base_key):
    """MCMC-specific random key."""
    return base_key


@pytest.fixture
def beta_bernoulli_model():
    """Beta-Bernoulli conjugate model for exact posterior testing."""

    @gen
    def model():
        p = beta(2.0, 5.0) @ "p"
        obs = flip(p) @ "obs"
        return obs

    return model


@pytest.fixture
def gamma_exponential_model():
    """Simple exponential model for testing."""

    @gen
    def model():
        return exponential(2.0) @ "x"

    return model


@pytest.fixture
def mcmc_tolerance():
    """Tolerance for MCMC convergence tests."""
    return 0.3


# ============================================================================
# Helper Functions for MCMC Post-Processing
# ============================================================================


def apply_burn_in(result: MCMCResult, burn_in_frac: float = 0.2) -> MCMCResult:
    """
    Apply burn-in to MCMC results by discarding initial samples.

    Args:
        result: Original MCMC result
        burn_in_frac: Fraction of samples to discard as burn-in

    Returns:
        New MCMCResult with burn-in samples removed
    """
    burn_in_steps = int(result.n_steps * burn_in_frac)

    # Apply burn-in using tree_map to handle all trace structures
    post_burn_in_traces = jax.tree_util.tree_map(
        lambda x: x[burn_in_steps:] if hasattr(x, "shape") and len(x.shape) > 0 else x,
        result.traces,
    )

    return MCMCResult(
        traces=post_burn_in_traces,
        n_steps=result.n_steps - burn_in_steps,
        acceptance_rate=result.acceptance_rate,  # Keep original acceptance rate
    )


# ============================================================================
# Helper Functions for Exact Posteriors
# ============================================================================


def exact_beta_bernoulli_posterior_moments(
    obs_value: bool, alpha: float = 2.0, beta: float = 5.0
):
    """
    Compute exact posterior moments for Beta-Bernoulli model.

    Prior: p ~ Beta(alpha, beta)
    Likelihood: obs ~ Bernoulli(p)
    Posterior: p | obs ~ Beta(alpha + obs, beta + (1 - obs))
    """
    posterior_alpha = alpha + float(obs_value)
    posterior_beta = beta + (1.0 - float(obs_value))

    # Exact moments of Beta distribution
    mean = posterior_alpha / (posterior_alpha + posterior_beta)
    variance = (posterior_alpha * posterior_beta) / (
        (posterior_alpha + posterior_beta) ** 2 * (posterior_alpha + posterior_beta + 1)
    )

    return mean, variance, posterior_alpha, posterior_beta


def exact_normal_normal_posterior_moments(
    y_obs: float,
    prior_mean: float = 0.0,
    prior_var: float = 1.0,
    likelihood_var: float = 0.25,
):
    """
    Compute exact posterior moments for Normal-Normal conjugate model.

    Prior: mu ~ Normal(prior_mean, prior_var)
    Likelihood: y ~ Normal(mu, likelihood_var)
    Posterior: mu | y ~ Normal(posterior_mean, posterior_var)
    """
    prior_precision = 1.0 / prior_var
    likelihood_precision = 1.0 / likelihood_var

    posterior_precision = prior_precision + likelihood_precision
    posterior_variance = 1.0 / posterior_precision

    posterior_mean = posterior_variance * (
        prior_precision * prior_mean + likelihood_precision * y_obs
    )

    return posterior_mean, posterior_variance


# ============================================================================
# MCMC Data Structure Tests
# ============================================================================


@pytest.mark.mcmc
@pytest.mark.unit
@pytest.mark.fast
def test_mcmc_result_creation(simple_normal_model, mcmc_steps_small, mcmc_key, helpers):
    """Test MCMCResult creation and field access."""
    initial_trace = simple_normal_model.simulate((0.0, 1.0))
    selection = sel("x")

    # Apply seed transformation to eliminate PJAX primitives
    seeded_mh = seed(metropolis_hastings)
    result = seeded_mh(mcmc_key, initial_trace, selection, mcmc_steps_small)

    # Validate result structure
    assert isinstance(result, MCMCResult)
    assert result.n_steps == mcmc_steps_small.value
    assert 0.0 <= result.acceptance_rate <= 1.0
    assert result.traces is not None

    # Validate trace structure
    helpers.assert_valid_trace(result.traces)


# ============================================================================
# Beta-Bernoulli Posterior Tests
# ============================================================================


@pytest.mark.mcmc
@pytest.mark.integration
@pytest.mark.slow
def test_mh_beta_bernoulli_obs_true(
    beta_bernoulli_model, mcmc_steps_large, mcmc_key, mcmc_tolerance
):
    """Test MH on Beta-Bernoulli with obs=True."""
    # Create constrained trace
    constraints = {"obs": True}
    initial_trace, _ = beta_bernoulli_model.generate((), constraints)

    # Run MCMC
    selection = sel("p")

    # Apply seed transformation
    seeded_mh = seed(metropolis_hastings)
    raw_result = seeded_mh(mcmc_key, initial_trace, selection, mcmc_steps_large)

    # Apply burn-in post-processing
    result = apply_burn_in(raw_result, burn_in_frac=0.3)

    # Extract p samples
    p_samples = result.traces.get_choices()["p"]

    # Compute sample moments
    sample_mean = jnp.mean(p_samples)
    sample_variance = jnp.var(p_samples)

    # Exact posterior moments
    exact_mean, exact_variance, _, _ = exact_beta_bernoulli_posterior_moments(True)

    # Test moments are close to exact values
    mean_error = jnp.abs(sample_mean - exact_mean)
    var_error = jnp.abs(sample_variance - exact_variance)

    # Use practical tolerance for MCMC convergence testing
    practical_mean_tolerance = 0.01  # Relaxed but reasonable tolerance

    assert mean_error < practical_mean_tolerance, (
        f"Mean error {mean_error:.4f} > tolerance {practical_mean_tolerance:.4f}"
    )
    assert var_error < mcmc_tolerance, (
        f"Variance error {var_error:.4f} > {mcmc_tolerance}"
    )
    assert result.acceptance_rate > 0.1, (
        f"Low acceptance rate: {result.acceptance_rate:.3f}"
    )


@pytest.mark.mcmc
@pytest.mark.integration
@pytest.mark.slow
def test_mh_beta_bernoulli_obs_false(
    beta_bernoulli_model, mcmc_steps_large, mcmc_key, mcmc_tolerance
):
    """Test MH on Beta-Bernoulli with obs=False."""
    # Create constrained trace
    constraints = {"obs": False}
    initial_trace, _ = beta_bernoulli_model.generate((), constraints)

    # Run MCMC
    selection = sel("p")

    # Apply seed transformation
    seeded_mh = seed(metropolis_hastings)
    raw_result = seeded_mh(mcmc_key, initial_trace, selection, mcmc_steps_large)

    # Apply burn-in post-processing
    result = apply_burn_in(raw_result, burn_in_frac=0.3)

    # Extract p samples
    p_samples = result.traces.get_choices()["p"]

    # Compute sample moments
    sample_mean = jnp.mean(p_samples)
    sample_variance = jnp.var(p_samples)

    # Exact posterior moments
    exact_mean, exact_variance, _, _ = exact_beta_bernoulli_posterior_moments(False)

    # Test moments
    mean_error = jnp.abs(sample_mean - exact_mean)
    var_error = jnp.abs(sample_variance - exact_variance)

    practical_mean_tolerance = 0.01  # Practical tolerance for MCMC

    assert mean_error < practical_mean_tolerance, (
        f"Mean error {mean_error:.4f} > tolerance {practical_mean_tolerance:.4f}"
    )
    assert var_error < mcmc_tolerance, (
        f"Variance error {var_error:.4f} > {mcmc_tolerance}"
    )


# ============================================================================
# Hierarchical Normal Posterior Tests
# ============================================================================


@pytest.mark.mcmc
@pytest.mark.integration
@pytest.mark.slow
def test_mh_hierarchical_normal(
    hierarchical_normal_model, mcmc_steps_large, mcmc_key, mcmc_tolerance
):
    """Test MH on hierarchical normal model."""
    y_observed = 1.5

    # Create constrained trace
    constraints = {"y": y_observed}
    initial_trace, _ = hierarchical_normal_model.generate((0.0, 1.0, 0.5), constraints)

    # Run MCMC
    selection = sel("mu")

    # Apply seed transformation
    seeded_mh = seed(metropolis_hastings)
    raw_result = seeded_mh(mcmc_key, initial_trace, selection, mcmc_steps_large)

    # Apply burn-in post-processing
    result = apply_burn_in(raw_result, burn_in_frac=0.3)

    # Extract mu samples
    mu_samples = result.traces.get_choices()["mu"]

    # Compute sample moments
    sample_mean = jnp.mean(mu_samples)
    sample_variance = jnp.var(mu_samples)

    # Exact posterior moments
    exact_mean, exact_variance = exact_normal_normal_posterior_moments(y_observed)

    # Test moments
    mean_error = jnp.abs(sample_mean - exact_mean)
    var_error = jnp.abs(sample_variance - exact_variance)

    practical_mean_tolerance = 1.5  # Larger tolerance for hierarchical model

    assert mean_error < practical_mean_tolerance, (
        f"Mean error {mean_error:.4f} > tolerance {practical_mean_tolerance:.4f}"
    )
    assert var_error < mcmc_tolerance, (
        f"Variance error {var_error:.4f} > {mcmc_tolerance}"
    )
    assert result.acceptance_rate > 0.1, (
        f"Low acceptance rate: {result.acceptance_rate:.3f}"
    )


# ============================================================================
# Bivariate Normal Tests
# ============================================================================


@pytest.mark.mcmc
@pytest.mark.integration
@pytest.mark.slow
def test_mh_bivariate_normal_marginal(
    bivariate_normal_model, mcmc_steps_medium, mcmc_key, mcmc_tolerance
):
    """Test MH on bivariate normal, conditioning on y."""
    y_observed = 2.0

    # Create constrained trace
    constraints = {"y": y_observed}
    initial_trace, _ = bivariate_normal_model.generate((), constraints)

    # Run MCMC to sample x | y
    selection = sel("x")

    seeded_mh = seed(metropolis_hastings)
    raw_result = seeded_mh(mcmc_key, initial_trace, selection, mcmc_steps_medium)

    # Apply burn-in post-processing
    result = apply_burn_in(raw_result, burn_in_frac=0.3)

    # Extract x samples
    x_samples = result.traces.get_choices()["x"]

    # For this model: x ~ N(0, 1), y | x ~ N(0.5*x, 0.5^2)
    # Posterior: x | y ~ N(posterior_mean, posterior_var)
    # Using Bayesian linear regression formulas

    prior_var = 1.0
    likelihood_var = 0.25  # 0.5^2
    slope = 0.5

    posterior_var = 1.0 / (1.0 / prior_var + slope**2 / likelihood_var)
    posterior_mean = posterior_var * (slope * y_observed / likelihood_var)

    # Compute sample moments
    sample_mean = jnp.mean(x_samples)
    sample_variance = jnp.var(x_samples)

    # Test moments
    mean_error = jnp.abs(sample_mean - posterior_mean)
    var_error = jnp.abs(sample_variance - posterior_var)

    practical_mean_tolerance = 2.5  # Larger tolerance for bivariate model

    assert mean_error < practical_mean_tolerance, (
        f"Mean error {mean_error:.4f} > tolerance {practical_mean_tolerance:.4f}"
    )
    assert var_error < mcmc_tolerance, (
        f"Variance error {var_error:.4f} > {mcmc_tolerance}"
    )


# ============================================================================
# MCMC Diagnostics Tests
# ============================================================================


@pytest.mark.mcmc
@pytest.mark.unit
@pytest.mark.fast
def test_acceptance_rates(gamma_exponential_model, mcmc_steps_medium, mcmc_key):
    """Test that acceptance rates are reasonable."""
    initial_trace = gamma_exponential_model.simulate(())
    selection = sel("x")

    seeded_mh = seed(metropolis_hastings)
    result = seeded_mh(mcmc_key, initial_trace, selection, mcmc_steps_medium)

    # Acceptance rate should be reasonable (not too high or low)
    assert 0.05 < result.acceptance_rate < 0.95, (
        f"Acceptance rate {result.acceptance_rate:.3f} outside reasonable range"
    )

    # Check acceptance rate is computed correctly
    assert result.acceptance_rate >= 0.0 and result.acceptance_rate <= 1.0


@pytest.mark.mcmc
@pytest.mark.integration
@pytest.mark.fast
def test_chain_stationarity(
    simple_normal_model, mcmc_steps_medium, mcmc_key, convergence_tolerance
):
    """Test basic stationarity of MCMC chains."""
    initial_trace = simple_normal_model.simulate((0.0, 1.0))
    selection = sel("x")

    seeded_mh = seed(metropolis_hastings)
    result = seeded_mh(mcmc_key, initial_trace, selection, mcmc_steps_medium)

    samples = result.traces.get_choices()["x"]

    # Split chain in half and compare means (rough stationarity test)
    first_half = samples[: mcmc_steps_medium.value // 2]
    second_half = samples[mcmc_steps_medium.value // 2 :]

    mean_diff = jnp.abs(jnp.mean(first_half) - jnp.mean(second_half))

    # Difference in means should be small for stationary chain
    assert mean_diff < convergence_tolerance, (
        f"Large difference in half-chain means: {mean_diff:.3f}"
    )


# ============================================================================
# Distribution Moment Validation Tests
# ============================================================================


@pytest.mark.mcmc
@pytest.mark.integration
@pytest.mark.slow
def test_exponential_moments(
    gamma_exponential_model, mcmc_steps_large, mcmc_key, mcmc_tolerance
):
    """Test MCMC samples match exponential distribution moments."""
    rate = 2.0
    initial_trace = gamma_exponential_model.simulate(())
    selection = sel("x")

    # Apply seed transformation
    seeded_mh = seed(metropolis_hastings)
    raw_result = seeded_mh(mcmc_key, initial_trace, selection, mcmc_steps_large)

    # Apply burn-in post-processing
    result = apply_burn_in(raw_result, burn_in_frac=0.3)

    samples = result.traces.get_choices()["x"]

    # Exponential(rate) has mean = 1/rate, variance = 1/rate^2
    expected_mean = 1.0 / rate
    expected_var = 1.0 / (rate**2)

    sample_mean = jnp.mean(samples)
    sample_var = jnp.var(samples)

    mean_error = jnp.abs(sample_mean - expected_mean)
    var_error = jnp.abs(sample_var - expected_var)

    # Test moments with reasonable tolerances
    practical_mean_tolerance = 0.3  # Practical tolerance for exponential model

    assert mean_error < practical_mean_tolerance, (
        f"Mean error {mean_error:.4f} > tolerance {practical_mean_tolerance:.4f}"
    )
    assert var_error < mcmc_tolerance, (
        f"Variance error {var_error:.4f} > {mcmc_tolerance}"
    )


# ============================================================================
# Robustness Tests
# ============================================================================


@pytest.mark.mcmc
@pytest.mark.regression
@pytest.mark.fast
@pytest.mark.parametrize("seed_val", [42, 123, 456, 789])
def test_mcmc_deterministic_with_seed(simple_normal_model, mcmc_steps_small, seed_val):
    """Test that MCMC is deterministic given the same seed."""
    key = jrand.key(seed_val)
    initial_trace = simple_normal_model.simulate((0.0, 1.0))
    selection = sel("x")

    seeded_mh = seed(metropolis_hastings)

    # Run twice with same key
    result1 = seeded_mh(key, initial_trace, selection, mcmc_steps_small)
    result2 = seeded_mh(key, initial_trace, selection, mcmc_steps_small)

    # Results should be identical
    samples1 = result1.traces.get_choices()["x"]
    samples2 = result2.traces.get_choices()["x"]

    assert jnp.allclose(samples1, samples2), "MCMC not deterministic with same seed"
    assert jnp.allclose(result1.acceptance_rate, result2.acceptance_rate), (
        "Acceptance rates differ"
    )


@pytest.mark.mcmc
@pytest.mark.regression
@pytest.mark.fast
@pytest.mark.parametrize("n_steps_val", [10, 50, 100])
def test_mcmc_result_structure(simple_normal_model, base_key, n_steps_val, helpers):
    """Test MCMC result structure with different step counts."""
    initial_trace = simple_normal_model.simulate((0.0, 1.0))
    selection = sel("x")
    n_steps = Const(n_steps_val)

    seeded_mh = seed(metropolis_hastings)
    result = seeded_mh(base_key, initial_trace, selection, n_steps)

    # Check result structure
    assert result.n_steps == n_steps_val
    assert result.traces.get_choices()["x"].shape == (n_steps_val,)

    # Validate traces
    helpers.assert_valid_trace(result.traces)
