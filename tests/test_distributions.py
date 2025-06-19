"""
Test cases for GenJAX distributions.

These tests validate all probability distributions in the distributions module:
- Basic functionality (simulate, assess, log_prob)
- Consistency between GenJAX and TFP implementations
- Parameter validation and edge cases
- Integration with the generative function interface
"""

import jax
import jax.numpy as jnp
import jax.random as jrand
import pytest
import tensorflow_probability.substrates.jax as tfp

from genjax.core import gen
from genjax.distributions import (
    # Original distributions
    bernoulli,
    beta,
    categorical,
    flip,
    normal,
    uniform,
    exponential,
    poisson,
    multivariate_normal,
    dirichlet,
    geometric,
    # New high-priority distributions
    binomial,
    gamma,
    log_normal,
    student_t,
    laplace,
    half_normal,
    inverse_gamma,
    weibull,
    cauchy,
    chi2,
    multinomial,
    negative_binomial,
    zipf,
)

tfd = tfp.distributions


# =============================================================================
# TEST FIXTURES AND HELPERS
# =============================================================================


@pytest.fixture
def key():
    """Standard random key for reproducible tests."""
    return jrand.PRNGKey(42)


@pytest.fixture
def strict_tolerance():
    """Strict tolerance for exact comparisons."""
    return 1e-6


@pytest.fixture
def standard_tolerance():
    """Standard tolerance for numerical comparisons."""
    return 1e-4


def assert_distribution_consistency(
    dist_fn, tfp_dist_fn, params, samples, tolerance=1e-4
):
    """Test that GenJAX distribution matches TFP distribution."""
    # Test log probabilities - GenJAX assess takes (value, *params)
    for sample in samples:
        genjax_logprob, _ = dist_fn().assess(sample, *params)
        tfp_logprob = tfp_dist_fn(*params).log_prob(sample)

        assert jnp.allclose(genjax_logprob, tfp_logprob, atol=tolerance), (
            f"Log probabilities differ for sample {sample}: GenJAX={genjax_logprob}, TFP={tfp_logprob}"
        )

    # Test sampling - GenJAX simulate takes (*params)
    genjax_trace = dist_fn().simulate(*params)
    genjax_sample = genjax_trace.get_retval()

    # Test that samples have reasonable values (finite)
    assert jnp.isfinite(genjax_sample), f"Sample is not finite: {genjax_sample}"


# =============================================================================
# DISCRETE DISTRIBUTIONS TESTS
# =============================================================================


@pytest.mark.distributions
@pytest.mark.unit
@pytest.mark.fast
def test_bernoulli_consistency(key, standard_tolerance):
    """Test Bernoulli distribution consistency with TFP."""
    logits = 0.5
    samples = jnp.array([0.0, 1.0, 0.0, 1.0])

    assert_distribution_consistency(
        lambda x: bernoulli(logits=x),
        lambda x: tfd.Bernoulli(logits=x),
        (logits,),
        samples,
        standard_tolerance,
    )


@pytest.mark.distributions
@pytest.mark.unit
@pytest.mark.fast
def test_flip_consistency(key, standard_tolerance):
    """Test Flip distribution consistency with TFP."""
    p = 0.7
    samples = jnp.array([True, False, True, False])

    # Test that flip produces boolean samples
    trace = flip(p).simulate(key)
    assert trace.get_retval().dtype == jnp.bool_

    # Test log probability computation
    genjax_logprob, _ = flip(p).assess(samples, p)
    tfp_logprob = tfd.Bernoulli(probs=p, dtype=jnp.bool_).log_prob(samples)

    assert jnp.allclose(genjax_logprob, tfp_logprob, atol=standard_tolerance)


@pytest.mark.distributions
@pytest.mark.unit
@pytest.mark.fast
def test_binomial_consistency(key, standard_tolerance):
    """Test Binomial distribution consistency with TFP."""
    total_count = 10
    probs = 0.3
    samples = jnp.array([2, 5, 8, 3])

    assert_distribution_consistency(
        lambda n, p: binomial(total_count=n, probs=p),
        lambda n, p: tfd.Binomial(total_count=n, probs=p),
        (total_count, probs),
        samples,
        standard_tolerance,
    )


@pytest.mark.distributions
@pytest.mark.unit
@pytest.mark.fast
def test_categorical_consistency(key, standard_tolerance):
    """Test Categorical distribution consistency with TFP."""
    logits = jnp.array([0.1, 0.6, 0.3])
    samples = jnp.array([0, 1, 2, 1])

    assert_distribution_consistency(
        lambda x: categorical(logits=x),
        lambda x: tfd.Categorical(logits=x),
        (logits,),
        samples,
        standard_tolerance,
    )


@pytest.mark.distributions
@pytest.mark.unit
@pytest.mark.fast
def test_geometric_consistency(key, standard_tolerance):
    """Test Geometric distribution consistency with TFP."""
    probs = 0.2
    samples = jnp.array([1, 3, 5, 2])

    assert_distribution_consistency(
        lambda p: geometric(probs=p),
        lambda p: tfd.Geometric(probs=p),
        (probs,),
        samples,
        standard_tolerance,
    )


@pytest.mark.distributions
@pytest.mark.unit
@pytest.mark.fast
def test_poisson_consistency(key, standard_tolerance):
    """Test Poisson distribution consistency with TFP."""
    rate = 3.5
    samples = jnp.array([2, 4, 1, 6])

    assert_distribution_consistency(
        lambda r: poisson(rate=r),
        lambda r: tfd.Poisson(rate=r),
        (rate,),
        samples,
        standard_tolerance,
    )


@pytest.mark.distributions
@pytest.mark.unit
@pytest.mark.fast
def test_multinomial_consistency(key, standard_tolerance):
    """Test Multinomial distribution consistency with TFP."""
    total_count = 10
    probs = jnp.array([0.2, 0.5, 0.3])
    samples = jnp.array([[2, 5, 3], [1, 6, 3], [3, 4, 3]])

    assert_distribution_consistency(
        lambda n, p: multinomial(total_count=n, probs=p),
        lambda n, p: tfd.Multinomial(total_count=n, probs=p),
        (total_count, probs),
        samples,
        standard_tolerance,
    )


@pytest.mark.distributions
@pytest.mark.unit
@pytest.mark.fast
def test_negative_binomial_consistency(key, standard_tolerance):
    """Test Negative Binomial distribution consistency with TFP."""
    total_count = 5
    probs = 0.3
    samples = jnp.array([8, 12, 6, 15])

    assert_distribution_consistency(
        lambda n, p: negative_binomial(total_count=n, probs=p),
        lambda n, p: tfd.NegativeBinomial(total_count=n, probs=p),
        (total_count, probs),
        samples,
        standard_tolerance,
    )


@pytest.mark.distributions
@pytest.mark.unit
@pytest.mark.fast
def test_zipf_consistency(key, standard_tolerance):
    """Test Zipf distribution consistency with TFP."""
    power = 2.0
    samples = jnp.array([1, 2, 3, 1])

    assert_distribution_consistency(
        lambda p: zipf(power=p),
        lambda p: tfd.Zipf(power=p),
        (power,),
        samples,
        standard_tolerance,
    )


# =============================================================================
# CONTINUOUS DISTRIBUTIONS TESTS
# =============================================================================


@pytest.mark.distributions
@pytest.mark.unit
@pytest.mark.fast
def test_normal_consistency(key, standard_tolerance):
    """Test Normal distribution consistency with TFP."""
    loc = 1.0
    scale = 2.0
    samples = jnp.array([0.5, 1.5, -0.5, 3.0])

    assert_distribution_consistency(
        lambda: normal,
        lambda mu, sigma: tfd.Normal(loc=mu, scale=sigma),
        (loc, scale),
        samples,
        standard_tolerance,
    )


@pytest.mark.distributions
@pytest.mark.unit
@pytest.mark.fast
def test_beta_consistency(key, standard_tolerance):
    """Test Beta distribution consistency with TFP."""
    concentration1 = 2.0
    concentration0 = 3.0
    samples = jnp.array([0.2, 0.5, 0.8, 0.1])

    assert_distribution_consistency(
        lambda a, b: beta(concentration1=a, concentration0=b),
        lambda a, b: tfd.Beta(concentration1=a, concentration0=b),
        (concentration1, concentration0),
        samples,
        standard_tolerance,
    )


@pytest.mark.distributions
@pytest.mark.unit
@pytest.mark.fast
def test_uniform_consistency(key, standard_tolerance):
    """Test Uniform distribution consistency with TFP."""
    low = -1.0
    high = 3.0
    samples = jnp.array([0.0, 1.5, -0.5, 2.8])

    assert_distribution_consistency(
        lambda l, h: uniform(low=l, high=h),
        lambda l, h: tfd.Uniform(low=l, high=h),
        (low, high),
        samples,
        standard_tolerance,
    )


@pytest.mark.distributions
@pytest.mark.unit
@pytest.mark.fast
def test_exponential_consistency(key, standard_tolerance):
    """Test Exponential distribution consistency with TFP."""
    rate = 1.5
    samples = jnp.array([0.5, 1.0, 2.0, 0.1])

    assert_distribution_consistency(
        lambda r: exponential(rate=r),
        lambda r: tfd.Exponential(rate=r),
        (rate,),
        samples,
        standard_tolerance,
    )


@pytest.mark.distributions
@pytest.mark.unit
@pytest.mark.fast
def test_gamma_consistency(key, standard_tolerance):
    """Test Gamma distribution consistency with TFP."""
    concentration = 2.0
    rate = 1.5
    samples = jnp.array([0.5, 1.5, 3.0, 0.8])

    assert_distribution_consistency(
        lambda a, r: gamma(concentration=a, rate=r),
        lambda a, r: tfd.Gamma(concentration=a, rate=r),
        (concentration, rate),
        samples,
        standard_tolerance,
    )


@pytest.mark.distributions
@pytest.mark.unit
@pytest.mark.fast
def test_log_normal_consistency(key, standard_tolerance):
    """Test Log-Normal distribution consistency with TFP."""
    loc = 0.0
    scale = 1.0
    samples = jnp.array([0.5, 1.5, 3.0, 0.1])

    assert_distribution_consistency(
        lambda mu, sigma: log_normal(loc=mu, scale=sigma),
        lambda mu, sigma: tfd.LogNormal(loc=mu, scale=sigma),
        (loc, scale),
        samples,
        standard_tolerance,
    )


@pytest.mark.distributions
@pytest.mark.unit
@pytest.mark.fast
def test_student_t_consistency(key, standard_tolerance):
    """Test Student's t distribution consistency with TFP."""
    df = 3.0
    loc = 0.0
    scale = 1.0
    samples = jnp.array([-2.0, 0.0, 1.5, -0.5])

    assert_distribution_consistency(
        lambda d, l, s: student_t(df=d, loc=l, scale=s),
        lambda d, l, s: tfd.StudentT(df=d, loc=l, scale=s),
        (df, loc, scale),
        samples,
        standard_tolerance,
    )


@pytest.mark.distributions
@pytest.mark.unit
@pytest.mark.fast
def test_laplace_consistency(key, standard_tolerance):
    """Test Laplace distribution consistency with TFP."""
    loc = 1.0
    scale = 0.5
    samples = jnp.array([0.5, 1.0, 1.5, 2.0])

    assert_distribution_consistency(
        lambda l, s: laplace(loc=l, scale=s),
        lambda l, s: tfd.Laplace(loc=l, scale=s),
        (loc, scale),
        samples,
        standard_tolerance,
    )


@pytest.mark.distributions
@pytest.mark.unit
@pytest.mark.fast
def test_half_normal_consistency(key, standard_tolerance):
    """Test Half-Normal distribution consistency with TFP."""
    scale = 1.5
    samples = jnp.array([0.5, 1.0, 2.0, 0.1])

    assert_distribution_consistency(
        lambda s: half_normal(scale=s),
        lambda s: tfd.HalfNormal(scale=s),
        (scale,),
        samples,
        standard_tolerance,
    )


@pytest.mark.distributions
@pytest.mark.unit
@pytest.mark.fast
def test_inverse_gamma_consistency(key, standard_tolerance):
    """Test Inverse Gamma distribution consistency with TFP."""
    concentration = 3.0
    rate = 2.0
    samples = jnp.array([0.5, 1.0, 1.5, 2.0])

    assert_distribution_consistency(
        lambda a, r: inverse_gamma(concentration=a, rate=r),
        lambda a, r: tfd.InverseGamma(concentration=a, rate=r),
        (concentration, rate),
        samples,
        standard_tolerance,
    )


@pytest.mark.distributions
@pytest.mark.unit
@pytest.mark.fast
def test_weibull_consistency(key, standard_tolerance):
    """Test Weibull distribution consistency with TFP."""
    concentration = 2.0
    scale = 1.5
    samples = jnp.array([0.5, 1.0, 1.5, 2.0])

    assert_distribution_consistency(
        lambda c, s: weibull(concentration=c, scale=s),
        lambda c, s: tfd.Weibull(concentration=c, scale=s),
        (concentration, scale),
        samples,
        standard_tolerance,
    )


@pytest.mark.distributions
@pytest.mark.unit
@pytest.mark.fast
def test_cauchy_consistency(key, standard_tolerance):
    """Test Cauchy distribution consistency with TFP."""
    loc = 0.0
    scale = 1.0
    samples = jnp.array([-2.0, 0.0, 1.5, -0.5])

    assert_distribution_consistency(
        lambda l, s: cauchy(loc=l, scale=s),
        lambda l, s: tfd.Cauchy(loc=l, scale=s),
        (loc, scale),
        samples,
        standard_tolerance,
    )


@pytest.mark.distributions
@pytest.mark.unit
@pytest.mark.fast
def test_chi2_consistency(key, standard_tolerance):
    """Test Chi-squared distribution consistency with TFP."""
    df = 4.0
    samples = jnp.array([1.0, 3.0, 6.0, 9.0])

    assert_distribution_consistency(
        lambda d: chi2(df=d),
        lambda d: tfd.Chi2(df=d),
        (df,),
        samples,
        standard_tolerance,
    )


# =============================================================================
# MULTIVARIATE DISTRIBUTIONS TESTS
# =============================================================================


@pytest.mark.distributions
@pytest.mark.unit
@pytest.mark.fast
def test_multivariate_normal_consistency(key, standard_tolerance):
    """Test Multivariate Normal distribution consistency with TFP."""
    loc = jnp.array([0.0, 1.0])
    cov = jnp.array([[1.0, 0.5], [0.5, 2.0]])
    samples = jnp.array([[0.5, 1.5], [-0.5, 0.8], [1.0, 2.0]])

    assert_distribution_consistency(
        lambda mu, sigma: multivariate_normal(loc=mu, covariance_matrix=sigma),
        lambda mu, sigma: tfd.MultivariateNormalFullCovariance(
            loc=mu, covariance_matrix=sigma
        ),
        (loc, cov),
        samples,
        standard_tolerance,
    )


@pytest.mark.distributions
@pytest.mark.unit
@pytest.mark.fast
def test_dirichlet_consistency(key, standard_tolerance):
    """Test Dirichlet distribution consistency with TFP."""
    concentration = jnp.array([1.0, 2.0, 3.0])
    samples = jnp.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3], [0.4, 0.2, 0.4]])

    assert_distribution_consistency(
        lambda c: dirichlet(concentration=c),
        lambda c: tfd.Dirichlet(concentration=c),
        (concentration,),
        samples,
        standard_tolerance,
    )


# =============================================================================
# INTEGRATION WITH GENERATIVE FUNCTIONS TESTS
# =============================================================================


@pytest.mark.distributions
@pytest.mark.unit
@pytest.mark.fast
def test_distributions_in_generative_functions(key, standard_tolerance):
    """Test that all distributions work correctly in @gen functions."""

    @gen
    def test_model():
        # Test some key distributions in generative context
        x1 = normal(0.0, 1.0) @ "normal"
        x2 = gamma(2.0, 1.0) @ "gamma"
        x3 = binomial(10, 0.5) @ "binomial"
        x4 = beta(2.0, 3.0) @ "beta"
        return x1 + x2, x3, x4

    # Test simulate
    trace = test_model.simulate(key)
    assert trace.get_retval() is not None

    # Test assess
    choices = trace.get_choices()
    log_prob, retval = test_model.assess(choices)
    assert jnp.isfinite(log_prob)

    # Test that choices are accessible
    assert "normal" in choices
    assert "gamma" in choices
    assert "binomial" in choices
    assert "beta" in choices


# =============================================================================
# PARAMETER VALIDATION TESTS
# =============================================================================


@pytest.mark.distributions
@pytest.mark.unit
@pytest.mark.fast
def test_parameter_validation():
    """Test that invalid parameters raise appropriate errors."""

    with pytest.raises((ValueError, Exception)):
        # Negative scale should fail
        normal(0.0, -1.0).simulate(jrand.PRNGKey(0))

    with pytest.raises((ValueError, Exception)):
        # Invalid probability should fail
        binomial(10, 1.5).simulate(jrand.PRNGKey(0))

    with pytest.raises((ValueError, Exception)):
        # Non-positive concentration should fail
        gamma(-1.0, 1.0).simulate(jrand.PRNGKey(0))


# =============================================================================
# PERFORMANCE AND COMPILATION TESTS
# =============================================================================


@pytest.mark.distributions
@pytest.mark.unit
@pytest.mark.fast
def test_distributions_jit_compilation(key):
    """Test that distributions work correctly under JIT compilation."""

    @jax.jit
    def jitted_sampling(key):
        # Test various distributions under JIT
        k1, k2, k3, k4 = jrand.split(key, 4)

        s1 = normal(0.0, 1.0).simulate(k1).get_retval()
        s2 = gamma(2.0, 1.0).simulate(k2).get_retval()
        s3 = binomial(10, 0.5).simulate(k3).get_retval()
        s4 = beta(2.0, 3.0).simulate(k4).get_retval()

        return s1, s2, s3, s4

    # Should compile and run without errors
    result = jitted_sampling(key)
    assert len(result) == 4
    assert all(jnp.isfinite(x) for x in result if jnp.isscalar(x))


@pytest.mark.distributions
@pytest.mark.integration
@pytest.mark.slow
def test_distributions_vectorization(key):
    """Test that distributions work correctly with vectorization."""

    @gen
    def vector_model(n):
        return normal(0.0, 1.0).vmap(in_axes=(None, None))(n) @ "samples"

    # Test vectorized sampling
    trace = vector_model.simulate(key, 100)
    samples = trace.get_retval()
    assert samples.shape == (100,)

    # Test vectorized assessment
    choices = {"samples": samples}
    log_prob, _ = vector_model.assess(choices, 100)
    assert jnp.isfinite(log_prob)
