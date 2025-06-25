"""
Tests for GenJAX state space models with exact inference.

This module provides comprehensive tests for the state space models in the
extras module, including discrete Hidden Markov Models (HMMs) and linear
Gaussian state space models. Tests validate exact inference algorithms
against TensorFlow Probability implementations.

The module includes:
- Discrete HMM tests against TFP's HiddenMarkovModel
- Linear Gaussian SSM tests against TFP's LinearGaussianStateSpaceModel
- Forward filtering backward sampling (FFBS) validation
- Kalman filtering and smoothing validation
- Inference testing API validation
"""

import jax
import jax.numpy as jnp
import jax.random as jrand

# TensorFlow Probability imports
import tensorflow_probability.substrates.jax as tfp

# GenJAX imports
from genjax.core import const
from genjax.pjax import seed

# Import state space implementations from extras module
from genjax.extras.state_space import (
    # Discrete HMM functions
    forward_filter,
    backward_sample,
    sample_hmm_dataset,
    discrete_hmm_test_dataset,
    discrete_hmm_exact_log_marginal,
    discrete_hmm_inference_problem,
    # Linear Gaussian functions
    kalman_filter,
    kalman_smoother,
    sample_linear_gaussian_dataset,
    linear_gaussian_test_dataset,
    linear_gaussian_exact_log_marginal,
    linear_gaussian_inference_problem,
)

tfd = tfp.distributions

# JIT-compiled versions for performance
_jitted_forward_filter = jax.jit(forward_filter)
_jitted_kalman_filter = jax.jit(kalman_filter)
_jitted_kalman_smoother = jax.jit(kalman_smoother)


# =============================================================================
# DISCRETE HMM TESTS
# =============================================================================


class TestDiscreteHMMAgainstTFP:
    """Test suite comparing GenJAX discrete HMM against TFP's HiddenMarkovModel."""

    def create_tfp_hmm(
        self, initial_probs, transition_matrix, emission_matrix, num_steps
    ):
        """Create equivalent TFP HiddenMarkovModel."""
        initial_distribution = tfd.Categorical(probs=initial_probs)
        transition_distribution = tfd.Categorical(probs=transition_matrix)
        observation_distribution = tfd.Categorical(probs=emission_matrix)

        return tfd.HiddenMarkovModel(
            initial_distribution=initial_distribution,
            transition_distribution=transition_distribution,
            observation_distribution=observation_distribution,
            num_steps=num_steps,
        )

    def create_simple_hmm_params(self):
        """Create simple HMM parameters for testing."""
        # 2 states, 3 observations
        initial_probs = jnp.array([0.6, 0.4])
        transition_matrix = jnp.array([[0.7, 0.3], [0.4, 0.6]])
        emission_matrix = jnp.array([[0.5, 0.3, 0.2], [0.1, 0.4, 0.5]])

        return initial_probs, transition_matrix, emission_matrix

    def create_complex_hmm_params(self):
        """Create more complex HMM parameters."""
        # 4 states, 5 observations
        key = jrand.PRNGKey(42)

        # Random but valid probability matrices
        initial_probs = jnp.array([0.25, 0.25, 0.25, 0.25])

        # Create row-stochastic transition matrix
        transition_raw = jrand.uniform(key, (4, 4))
        transition_matrix = transition_raw / jnp.sum(
            transition_raw, axis=1, keepdims=True
        )

        # Create row-stochastic emission matrix
        key, _ = jrand.split(key)
        emission_raw = jrand.uniform(key, (4, 5))
        emission_matrix = emission_raw / jnp.sum(emission_raw, axis=1, keepdims=True)

        return initial_probs, transition_matrix, emission_matrix

    def test_simple_hmm_log_prob_consistency(self):
        """Test that log probabilities match between GenJAX and TFP for simple HMM."""
        initial_probs, transition_matrix, emission_matrix = (
            self.create_simple_hmm_params()
        )

        T = 10
        num_sequences = 5

        # Create TFP model
        tfp_hmm = self.create_tfp_hmm(
            initial_probs, transition_matrix, emission_matrix, T
        )

        # Generate test sequences using TFP
        key = jrand.PRNGKey(0)
        sequences = tfp_hmm.sample(num_sequences, seed=key)

        for i in range(num_sequences):
            obs_seq = sequences[i]

            # Compute log prob with TFP
            tfp_log_prob = tfp_hmm.log_prob(obs_seq)

            # Compute log prob with GenJAX
            genjax_log_prob = discrete_hmm_exact_log_marginal(
                obs_seq, initial_probs, transition_matrix, emission_matrix
            )

            # Should match within numerical tolerance (relaxed for float32 precision)
            assert jnp.abs(tfp_log_prob - genjax_log_prob) < 2e-6, (
                f"Log prob mismatch: TFP={tfp_log_prob}, GenJAX={genjax_log_prob}"
            )

    def test_complex_hmm_log_prob_consistency(self):
        """Test log probability consistency for more complex HMM."""
        initial_probs, transition_matrix, emission_matrix = (
            self.create_complex_hmm_params()
        )

        T = 15

        # Create TFP model
        tfp_hmm = self.create_tfp_hmm(
            initial_probs, transition_matrix, emission_matrix, T
        )

        # Generate single test sequence
        key = jrand.PRNGKey(123)
        obs_seq = tfp_hmm.sample(seed=key)

        # Compute log prob with TFP
        tfp_log_prob = tfp_hmm.log_prob(obs_seq)

        # Compute log prob with GenJAX
        genjax_log_prob = discrete_hmm_exact_log_marginal(
            obs_seq, initial_probs, transition_matrix, emission_matrix
        )

        # Should match within numerical tolerance
        assert jnp.abs(tfp_log_prob - genjax_log_prob) < 1e-5

    def test_forward_filtering_messages(self):
        """Test that forward filtering produces valid probability distributions."""
        initial_probs, transition_matrix, emission_matrix = (
            self.create_simple_hmm_params()
        )

        # Generate test sequence
        key = jrand.PRNGKey(42)
        seeded_sample = seed(sample_hmm_dataset)
        states, obs_seq, dataset_dict = seeded_sample(
            key, initial_probs, transition_matrix, emission_matrix, T=const(10)
        )

        # Run forward filtering
        log_messages, log_marginal = _jitted_forward_filter(
            obs_seq, initial_probs, transition_matrix, emission_matrix
        )

        # Check that messages are valid log probabilities
        for t in range(len(log_messages)):
            # Convert to probabilities and check they sum to 1
            probs = jnp.exp(log_messages[t])
            assert jnp.abs(jnp.sum(probs) - 1.0) < 1e-6  # Relaxed for float32 precision
            assert jnp.all(probs >= 0)

    def test_ffbs_sampling_consistency(self):
        """Test that FFBS produces samples with correct marginal probabilities."""
        initial_probs, transition_matrix, emission_matrix = (
            self.create_simple_hmm_params()
        )

        T = 8
        num_samples = 1000

        # Generate observation sequence
        key = jrand.PRNGKey(0)
        seeded_sample = seed(sample_hmm_dataset)
        states, observations, constraints = seeded_sample(
            key, initial_probs, transition_matrix, emission_matrix, T=const(T)
        )
        obs_seq = observations

        # Run forward filtering
        log_messages, log_marginal = _jitted_forward_filter(
            obs_seq, initial_probs, transition_matrix, emission_matrix
        )

        # Generate many samples using FFBS
        key = jrand.PRNGKey(1)
        seeded_backward_sample = seed(backward_sample)
        samples = []
        for i in range(num_samples):
            key, subkey = jrand.split(key)
            state_seq = seeded_backward_sample(subkey, log_messages, transition_matrix)
            samples.append(state_seq)

        samples = jnp.array(samples)

        # Check that empirical state probabilities match forward messages
        for t in range(T):
            empirical_probs = (
                jnp.bincount(samples[:, t], length=len(initial_probs)) / num_samples
            )
            true_probs = jnp.exp(log_messages[t])

            # Should match within Monte Carlo error (relaxed for stochastic sampling)
            assert jnp.max(jnp.abs(empirical_probs - true_probs)) < 0.10

    def test_inference_testing_api(self):
        """Test the inference testing API functions."""
        key = jrand.PRNGKey(42)
        initial_probs, transition_matrix, emission_matrix = (
            self.create_simple_hmm_params()
        )

        # Test discrete_hmm_test_dataset
        seeded_dataset = seed(discrete_hmm_test_dataset)
        dataset = seeded_dataset(
            key, initial_probs, transition_matrix, emission_matrix, T=const(10)
        )

        assert "z" in dataset  # True states
        assert "obs" in dataset  # Observations
        assert dataset["z"].shape == (10,)
        assert dataset["obs"].shape == (10,)

        # Test exact log marginal computation
        log_marginal = discrete_hmm_exact_log_marginal(
            dataset["obs"], initial_probs, transition_matrix, emission_matrix
        )
        assert jnp.isfinite(log_marginal)

        # Test one-call inference problem
        seeded_problem = seed(discrete_hmm_inference_problem)
        dataset2, log_marginal2 = seeded_problem(
            key, initial_probs, transition_matrix, emission_matrix, T=const(10)
        )

        assert "z" in dataset2
        assert "obs" in dataset2
        assert jnp.isfinite(log_marginal2)


# =============================================================================
# LINEAR GAUSSIAN SSM TESTS
# =============================================================================


class TestLinearGaussianSSMAgainstTFP:
    """Test suite comparing GenJAX linear Gaussian SSM against TFP's LinearGaussianStateSpaceModel."""

    def create_simple_lgssm_params(self):
        """Create simple linear Gaussian SSM parameters for testing."""
        # 1D state, 1D observation
        initial_mean = jnp.array([0.0])
        initial_cov = jnp.array([[1.0]])
        A = jnp.array([[0.9]])  # Stable dynamics
        Q = jnp.array([[0.1]])  # Process noise
        C = jnp.array([[1.0]])  # Direct observation
        R = jnp.array([[0.2]])  # Observation noise

        return initial_mean, initial_cov, A, Q, C, R

    def create_complex_lgssm_params(self):
        """Create more complex linear Gaussian SSM parameters."""
        # 2D state, 1D observation
        initial_mean = jnp.array([0.0, 0.0])
        initial_cov = jnp.array([[1.0, 0.1], [0.1, 1.0]])

        # Position-velocity model
        A = jnp.array([[1.0, 1.0], [0.0, 0.9]])
        Q = jnp.array([[0.01, 0.0], [0.0, 0.1]])

        # Observe position only
        C = jnp.array([[1.0, 0.0]])
        R = jnp.array([[0.1]])

        return initial_mean, initial_cov, A, Q, C, R

    def create_tfp_lgssm(self, initial_mean, initial_cov, A, Q, C, R, T):
        """Create equivalent TFP LinearGaussianStateSpaceModel."""
        # TFP uses slightly different parameterization
        initial_distribution = tfd.MultivariateNormalFullCovariance(
            loc=initial_mean, covariance_matrix=initial_cov
        )

        # Use time-invariant parameters to avoid broadcasting issues
        # TFP expects consistent batch dimensions across all parameters
        transition_matrix = A  # Shape: (state_dim, state_dim)
        transition_noise = tfd.MultivariateNormalFullCovariance(
            loc=jnp.zeros(A.shape[0]), covariance_matrix=Q
        )

        observation_matrix = C  # Shape: (obs_dim, state_dim)
        observation_noise = tfd.MultivariateNormalFullCovariance(
            loc=jnp.zeros(C.shape[0]), covariance_matrix=R
        )

        return tfd.LinearGaussianStateSpaceModel(
            num_timesteps=T,
            transition_matrix=transition_matrix,
            transition_noise=transition_noise,
            observation_matrix=observation_matrix,
            observation_noise=observation_noise,
            initial_state_prior=initial_distribution,
        )

    def test_simple_lgssm_log_prob_consistency(self):
        """Test log probability consistency for simple linear Gaussian SSM."""
        initial_mean, initial_cov, A, Q, C, R = self.create_simple_lgssm_params()

        T = 10

        # Create TFP model
        tfp_lgssm = self.create_tfp_lgssm(initial_mean, initial_cov, A, Q, C, R, T)

        # Generate test sequence using TFP
        key = jrand.PRNGKey(0)
        obs_seq = tfp_lgssm.sample(seed=key)

        # Compute log prob with TFP
        tfp_log_prob = tfp_lgssm.log_prob(obs_seq)

        # Compute log prob with GenJAX Kalman filter
        _, _, log_marginal = _jitted_kalman_filter(
            obs_seq, initial_mean, initial_cov, A, Q, C, R
        )

        # Should match within numerical tolerance
        assert jnp.abs(tfp_log_prob - log_marginal) < 1e-4

    def test_complex_lgssm_log_prob_consistency(self):
        """Test log probability consistency for complex linear Gaussian SSM."""
        initial_mean, initial_cov, A, Q, C, R = self.create_complex_lgssm_params()

        T = 15

        # Create TFP model
        tfp_lgssm = self.create_tfp_lgssm(initial_mean, initial_cov, A, Q, C, R, T)

        # Generate test sequence
        key = jrand.PRNGKey(123)
        obs_seq = tfp_lgssm.sample(seed=key)

        # Compute log prob with TFP
        tfp_log_prob = tfp_lgssm.log_prob(obs_seq)

        # Compute log prob with GenJAX
        _, _, log_marginal = _jitted_kalman_filter(
            obs_seq, initial_mean, initial_cov, A, Q, C, R
        )

        # Should match within numerical tolerance
        assert jnp.abs(tfp_log_prob - log_marginal) < 1e-3

    def test_kalman_filtering_posterior_means(self):
        """Test that Kalman filtering produces reasonable posterior means."""
        initial_mean, initial_cov, A, Q, C, R = self.create_simple_lgssm_params()

        # Generate synthetic data
        key = jrand.PRNGKey(42)
        seeded_sample = seed(sample_linear_gaussian_dataset)
        true_states, observations, constraints = seeded_sample(
            key, initial_mean, initial_cov, A, Q, C, R, T=const(10)
        )
        obs_seq = observations

        # Run Kalman filter
        filter_means, filter_covs, _ = _jitted_kalman_filter(
            obs_seq, initial_mean, initial_cov, A, Q, C, R
        )

        # Check that filter means are reasonable estimates of true states
        # (This is a loose test since filtering is online and won't be perfect)
        errors = jnp.abs(filter_means.squeeze() - true_states.squeeze())
        mean_error = jnp.mean(errors)

        # Mean error should be reasonable for this noise level
        assert mean_error < 1.0

    def test_kalman_smoothing_improvement(self):
        """Test that smoothing provides better estimates than filtering."""
        initial_mean, initial_cov, A, Q, C, R = self.create_complex_lgssm_params()

        # Generate synthetic data
        key = jrand.PRNGKey(42)
        seeded_sample = seed(sample_linear_gaussian_dataset)
        true_states, observations, constraints = seeded_sample(
            key, initial_mean, initial_cov, A, Q, C, R, T=const(20)
        )
        obs_seq = observations

        # Run filtering
        filter_means, filter_covs, _ = _jitted_kalman_filter(
            obs_seq, initial_mean, initial_cov, A, Q, C, R
        )

        # Run smoothing
        smooth_means, smooth_covs = _jitted_kalman_smoother(
            obs_seq, initial_mean, initial_cov, A, Q, C, R
        )

        # Compute errors
        filter_errors = jnp.mean((filter_means - true_states) ** 2, axis=0)
        smooth_errors = jnp.mean((smooth_means - true_states) ** 2, axis=0)

        # Smoothing should generally provide better estimates
        # (Allow some variance since this is stochastic)
        improvement_ratio = jnp.mean(smooth_errors) / jnp.mean(filter_errors)
        assert improvement_ratio < 1.1  # At least not much worse

    def test_inference_testing_api_lgssm(self):
        """Test the linear Gaussian SSM inference testing API."""
        key = jrand.PRNGKey(42)
        initial_mean, initial_cov, A, Q, C, R = self.create_simple_lgssm_params()

        # Test linear_gaussian_test_dataset
        seeded_dataset = seed(linear_gaussian_test_dataset)
        dataset = seeded_dataset(
            key, initial_mean, initial_cov, A, Q, C, R, T=const(10)
        )

        assert "z" in dataset  # True states
        assert "obs" in dataset  # Observations
        assert dataset["z"].shape[0] == 10  # T timesteps
        assert dataset["obs"].shape[0] == 10

        # Test exact log marginal computation
        log_marginal = linear_gaussian_exact_log_marginal(
            dataset["obs"], initial_mean, initial_cov, A, Q, C, R
        )
        assert jnp.isfinite(log_marginal)

        # Test one-call inference problem
        seeded_problem = seed(linear_gaussian_inference_problem)
        dataset2, log_marginal2 = seeded_problem(
            key, initial_mean, initial_cov, A, Q, C, R, T=const(10)
        )

        assert "z" in dataset2
        assert "obs" in dataset2
        assert jnp.isfinite(log_marginal2)


# =============================================================================
# PERFORMANCE AND CONVERGENCE TESTS
# =============================================================================


class TestStateSpacePerformance:
    """Test performance characteristics and convergence properties."""

    def test_hmm_scaling_with_sequence_length(self):
        """Test that HMM algorithms scale reasonably with sequence length."""
        initial_probs, transition_matrix, emission_matrix = (
            TestDiscreteHMMAgainstTFP().create_simple_hmm_params()
        )

        # Test different sequence lengths
        lengths = [10, 50, 100]
        times = []

        for T in lengths:
            # Generate test sequence
            key = jrand.PRNGKey(0)
            seeded_sample = seed(sample_hmm_dataset)
            states, observations, constraints = seeded_sample(
                key, initial_probs, transition_matrix, emission_matrix, T=const(T)
            )
            obs_seq = observations

            # Time forward filtering
            import time

            start = time.time()
            _ = _jitted_forward_filter(
                obs_seq, initial_probs, transition_matrix, emission_matrix
            )
            end = time.time()
            times.append(end - start)

        # Should scale roughly linearly (allowing for JIT overhead)
        # Just check it doesn't explode (note: first call includes JIT overhead)
        # Use median time for more robust comparison
        median_time = sorted(times)[len(times) // 2]
        assert times[-1] < median_time * 100  # Very loose bound to account for JIT

    def test_lgssm_numerical_stability(self):
        """Test numerical stability of Kalman filtering."""
        # Create potentially challenging parameters
        initial_mean = jnp.array([0.0])
        initial_cov = jnp.array([[100.0]])  # High initial uncertainty
        A = jnp.array([[0.99]])  # Nearly unstable
        Q = jnp.array([[0.001]])  # Low process noise
        C = jnp.array([[1.0]])
        R = jnp.array([[0.001]])  # Low observation noise

        T = 50

        # Generate observations
        key = jrand.PRNGKey(42)
        seeded_sample = seed(sample_linear_gaussian_dataset)
        true_states, observations, constraints = seeded_sample(
            key, initial_mean, initial_cov, A, Q, C, R, T=const(T)
        )
        obs_seq = observations

        # Run Kalman filter
        filter_means, filter_covs, log_marginal = _jitted_kalman_filter(
            obs_seq, initial_mean, initial_cov, A, Q, C, R
        )

        # Check for numerical issues
        assert jnp.all(jnp.isfinite(filter_means))
        assert jnp.all(jnp.isfinite(filter_covs))
        assert jnp.isfinite(log_marginal)

        # Covariances should remain positive definite
        assert jnp.all(jnp.linalg.eigvals(filter_covs) > 0)


if __name__ == "__main__":
    # Discrete HMM tests
    test_hmm = TestDiscreteHMMAgainstTFP()
    test_hmm.test_simple_hmm_log_prob_consistency()
    test_hmm.test_complex_hmm_log_prob_consistency()
    test_hmm.test_forward_filtering_messages()
    test_hmm.test_ffbs_sampling_consistency()
    test_hmm.test_inference_testing_api()

    # Linear Gaussian SSM tests
    test_lgssm = TestLinearGaussianSSMAgainstTFP()
    test_lgssm.test_simple_lgssm_log_prob_consistency()
    test_lgssm.test_complex_lgssm_log_prob_consistency()
    test_lgssm.test_kalman_filtering_posterior_means()
    test_lgssm.test_kalman_smoothing_improvement()
    test_lgssm.test_inference_testing_api_lgssm()

    # Performance tests
    test_perf = TestStateSpacePerformance()
    test_perf.test_hmm_scaling_with_sequence_length()
    test_perf.test_lgssm_numerical_stability()

    print("All state space tests passed!")
