"""
Test cases for GenJAX stdlib inference algorithms.

These tests compare approximate inference algorithms against exact inference
on discrete HMMs to validate correctness and accuracy.
"""

import jax.numpy as jnp

from genjax.stdlib import (
    importance_sampling,
    sequential_importance_sampling,
    estimate_marginal_likelihood,
)
from genjax.core import gen

from discrete_hmm import (
    discrete_hmm_model,
    forward_filter,
    sample_hmm_dataset,
)


def create_simple_hmm_params():
    """Create simple HMM parameters for testing."""
    # 2 states, 2 observations
    initial_probs = jnp.array([0.6, 0.4])
    transition_matrix = jnp.array([[0.7, 0.3], [0.4, 0.6]])
    emission_matrix = jnp.array(
        [
            [0.8, 0.2],  # state 0 -> obs 0 likely, obs 1 unlikely
            [0.3, 0.7],  # state 1 -> obs 0 unlikely, obs 1 likely
        ]
    )
    return initial_probs, transition_matrix, emission_matrix


def create_complex_hmm_params():
    """Create more complex HMM parameters for testing."""
    # 3 states, 4 observations
    initial_probs = jnp.array([0.5, 0.3, 0.2])
    transition_matrix = jnp.array([[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.1, 0.4, 0.5]])
    emission_matrix = jnp.array(
        [[0.7, 0.2, 0.05, 0.05], [0.1, 0.6, 0.2, 0.1], [0.05, 0.1, 0.3, 0.55]]
    )
    return initial_probs, transition_matrix, emission_matrix


@gen
def hmm_proposal(initial_probs, transition_matrix, emission_matrix, T):
    """
    Simple proposal for HMM that samples from the prior.
    """
    return discrete_hmm_model(initial_probs, transition_matrix, emission_matrix, T)


class TestImportanceSampling:
    """Test importance sampling against exact inference."""

    def test_simple_hmm_marginal_likelihood(self):
        """Test importance sampling marginal likelihood estimation on simple HMM."""
        initial_probs, transition_matrix, emission_matrix = create_simple_hmm_params()
        T = 5
        n_samples = 1000

        # Generate test data
        true_states, observations = sample_hmm_dataset(
            initial_probs, transition_matrix, emission_matrix, T
        )

        # Compute exact log marginal likelihood using forward filter
        _, exact_log_marginal = forward_filter(
            observations, initial_probs, transition_matrix, emission_matrix
        )

        # Create constraints from observations
        constraints = {f"obs_{t}": observations[t] for t in range(T)}

        # Estimate using importance sampling
        result = importance_sampling(
            discrete_hmm_model,
            hmm_proposal,
            (initial_probs, transition_matrix, emission_matrix, T),
            (initial_probs, transition_matrix, emission_matrix, T),
            n_samples,
            constraints,
        )

        estimated_log_marginal = result.log_marginal_likelihood

        # Check that estimate is close to exact value
        # Allow for Monte Carlo error with generous tolerance
        tolerance = 0.5  # log space tolerance
        assert jnp.abs(estimated_log_marginal - exact_log_marginal) < tolerance

        # Check that effective sample size is reasonable
        assert result.effective_sample_size > n_samples * 0.1

    def test_complex_hmm_marginal_likelihood(self):
        """Test importance sampling on more complex HMM."""
        initial_probs, transition_matrix, emission_matrix = create_complex_hmm_params()
        T = 8
        n_samples = 2000

        # Generate test data
        true_states, observations = sample_hmm_dataset(
            initial_probs, transition_matrix, emission_matrix, T
        )

        # Compute exact log marginal likelihood
        _, exact_log_marginal = forward_filter(
            observations, initial_probs, transition_matrix, emission_matrix
        )

        # Create constraints from observations
        constraints = {f"obs_{t}": observations[t] for t in range(T)}

        # Estimate using importance sampling
        result = importance_sampling(
            discrete_hmm_model,
            hmm_proposal,
            (initial_probs, transition_matrix, emission_matrix, T),
            (initial_probs, transition_matrix, emission_matrix, T),
            n_samples,
            constraints,
        )

        estimated_log_marginal = result.log_marginal_likelihood

        # Check accuracy
        tolerance = 0.7  # Slightly more generous for complex case
        assert jnp.abs(estimated_log_marginal - exact_log_marginal) < tolerance

    def test_marginal_likelihood_convergence(self):
        """Test that marginal likelihood estimates converge with more samples."""
        initial_probs, transition_matrix, emission_matrix = create_simple_hmm_params()
        T = 4

        # Generate test data
        true_states, observations = sample_hmm_dataset(
            initial_probs, transition_matrix, emission_matrix, T
        )

        # Compute exact log marginal likelihood
        _, exact_log_marginal = forward_filter(
            observations, initial_probs, transition_matrix, emission_matrix
        )

        # Create constraints
        constraints = {f"obs_{t}": observations[t] for t in range(T)}

        # Test with increasing sample sizes
        sample_sizes = [100, 500, 1000, 2000]
        errors = []

        for n_samples in sample_sizes:
            result = importance_sampling(
                discrete_hmm_model,
                hmm_proposal,
                (initial_probs, transition_matrix, emission_matrix, T),
                (initial_probs, transition_matrix, emission_matrix, T),
                n_samples,
                constraints,
            )

            error = jnp.abs(result.log_marginal_likelihood - exact_log_marginal)
            errors.append(error)

        # Errors should generally decrease (allow some Monte Carlo variation)
        # Check that largest sample size has smaller error than smallest
        assert errors[-1] < errors[0] * 1.5  # Allow some tolerance for randomness


class TestSequentialImportanceSampling:
    """Test sequential importance sampling against exact inference."""

    def test_sequential_marginal_likelihood(self):
        """Test sequential importance sampling marginal likelihood estimation."""
        initial_probs, transition_matrix, emission_matrix = create_simple_hmm_params()
        T = 6
        n_particles = 500

        # Generate test data
        true_states, observations = sample_hmm_dataset(
            initial_probs, transition_matrix, emission_matrix, T
        )

        # Compute exact log marginal likelihood
        _, exact_log_marginal = forward_filter(
            observations, initial_probs, transition_matrix, emission_matrix
        )

        # Convert observations to vectorized format for sequential sampling
        vectorized_observations = jnp.array(
            [{f"obs_{t}": observations[t]} for t in range(T)]
        )

        # Run sequential importance sampling
        result = sequential_importance_sampling(
            discrete_hmm_model,
            hmm_proposal,
            (initial_probs, transition_matrix, emission_matrix, T),
            (initial_probs, transition_matrix, emission_matrix, T),
            vectorized_observations,
            n_particles,
            resample_threshold=0.5,
        )

        estimated_log_marginal = result.log_marginal_likelihood

        # Check accuracy (sequential methods may have higher variance)
        tolerance = 1.0
        assert jnp.abs(estimated_log_marginal - exact_log_marginal) < tolerance

    def test_sequential_with_resampling(self):
        """Test sequential importance sampling with different resampling thresholds."""
        initial_probs, transition_matrix, emission_matrix = create_simple_hmm_params()
        T = 5
        n_particles = 300

        # Generate test data
        true_states, observations = sample_hmm_dataset(
            initial_probs, transition_matrix, emission_matrix, T
        )

        # Compute exact log marginal likelihood
        _, exact_log_marginal = forward_filter(
            observations, initial_probs, transition_matrix, emission_matrix
        )

        # Convert observations to vectorized format
        vectorized_observations = jnp.array(
            [{f"obs_{t}": observations[t]} for t in range(T)]
        )

        # Test different resampling thresholds
        thresholds = [0.3, 0.7, 1.0]  # 1.0 means always resample

        for threshold in thresholds:
            result = sequential_importance_sampling(
                discrete_hmm_model,
                hmm_proposal,
                (initial_probs, transition_matrix, emission_matrix, T),
                (initial_probs, transition_matrix, emission_matrix, T),
                vectorized_observations,
                n_particles,
                resample_threshold=threshold,
            )

            # All should give reasonable estimates
            tolerance = 1.2
            error = jnp.abs(result.log_marginal_likelihood - exact_log_marginal)
            assert error < tolerance, (
                f"Failed for threshold {threshold}, error: {error}"
            )


class TestMarginalLikelihoodEstimation:
    """Test the convenience function for marginal likelihood estimation."""

    def test_estimate_marginal_likelihood_function(self):
        """Test the estimate_marginal_likelihood convenience function."""
        initial_probs, transition_matrix, emission_matrix = create_simple_hmm_params()
        T = 4
        n_samples = 1000

        # Generate test data
        true_states, observations = sample_hmm_dataset(
            initial_probs, transition_matrix, emission_matrix, T
        )

        # Compute exact log marginal likelihood
        _, exact_log_marginal = forward_filter(
            observations, initial_probs, transition_matrix, emission_matrix
        )

        # Create constraints
        constraints = {f"obs_{t}": observations[t] for t in range(T)}

        # Estimate using convenience function
        estimated_log_marginal, standard_error = estimate_marginal_likelihood(
            discrete_hmm_model,
            hmm_proposal,
            (initial_probs, transition_matrix, emission_matrix, T),
            (initial_probs, transition_matrix, emission_matrix, T),
            constraints,
            n_samples,
        )

        # Check accuracy
        tolerance = 0.6
        assert jnp.abs(estimated_log_marginal - exact_log_marginal) < tolerance

        # Check that standard error is reasonable (positive and not too large)
        assert standard_error > 0
        assert standard_error < 2.0  # Should be reasonably precise


class TestRobustness:
    """Test robustness of inference algorithms."""

    def test_small_datasets(self):
        """Test behavior on very small datasets."""
        initial_probs, transition_matrix, emission_matrix = create_simple_hmm_params()
        T = 2  # Very short sequence
        n_samples = 100

        # Generate test data
        true_states, observations = sample_hmm_dataset(
            initial_probs, transition_matrix, emission_matrix, T
        )

        # Compute exact log marginal likelihood
        _, exact_log_marginal = forward_filter(
            observations, initial_probs, transition_matrix, emission_matrix
        )

        # Create constraints
        constraints = {f"obs_{t}": observations[t] for t in range(T)}

        # Should not crash and should give reasonable results
        result = importance_sampling(
            discrete_hmm_model,
            hmm_proposal,
            (initial_probs, transition_matrix, emission_matrix, T),
            (initial_probs, transition_matrix, emission_matrix, T),
            n_samples,
            constraints,
        )

        # More generous tolerance for small datasets
        tolerance = 1.5
        assert jnp.abs(result.log_marginal_likelihood - exact_log_marginal) < tolerance

    def test_deterministic_observations(self):
        """Test with highly deterministic observation model."""
        # Create HMM with very deterministic emissions
        initial_probs = jnp.array([0.5, 0.5])
        transition_matrix = jnp.array([[0.8, 0.2], [0.2, 0.8]])
        # Very deterministic emissions
        emission_matrix = jnp.array([[0.95, 0.05], [0.05, 0.95]])

        T = 4
        n_samples = 500

        # Generate test data
        true_states, observations = sample_hmm_dataset(
            initial_probs, transition_matrix, emission_matrix, T
        )

        # Compute exact log marginal likelihood
        _, exact_log_marginal = forward_filter(
            observations, initial_probs, transition_matrix, emission_matrix
        )

        # Create constraints
        constraints = {f"obs_{t}": observations[t] for t in range(T)}

        # Should handle deterministic case
        result = importance_sampling(
            discrete_hmm_model,
            hmm_proposal,
            (initial_probs, transition_matrix, emission_matrix, T),
            (initial_probs, transition_matrix, emission_matrix, T),
            n_samples,
            constraints,
        )

        tolerance = 0.8
        assert jnp.abs(result.log_marginal_likelihood - exact_log_marginal) < tolerance
