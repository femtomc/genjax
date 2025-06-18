"""
Test linear Gaussian state space model implementation against TensorFlow Probability.

This module validates GenJAX's linear Gaussian SSM implementation by comparing
marginal log probabilities and filtering/smoothing results against TensorFlow
Probability's LinearGaussianStateSpaceModel distribution.
"""

import jax
import jax.numpy as jnp
import jax.random as jrand
import pytest

# TensorFlow Probability imports
import tensorflow_probability.substrates.jax as tfp

# GenJAX imports
from genjax.core import const
from genjax.pjax import seed

# Import linear Gaussian implementation from extras module
from genjax.extras.state_space import (
    kalman_filter,
    kalman_smoother,
    sample_linear_gaussian_dataset,
    linear_gaussian_test_dataset,
    linear_gaussian_exact_log_marginal,
    linear_gaussian_inference_problem,
)


tfd = tfp.distributions


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
        """Create more complex linear Gaussian SSM parameters for testing."""
        # 2D state (position, velocity), 1D observation (position only)
        initial_mean = jnp.array([0.0, 0.0])
        initial_cov = jnp.array([[1.0, 0.0], [0.0, 1.0]])

        # Position-velocity dynamics with dt=1
        A = jnp.array([[1.0, 1.0], [0.0, 0.9]])  # Position += velocity, velocity *= 0.9
        Q = jnp.array([[0.1, 0.0], [0.0, 0.05]])  # Process noise
        C = jnp.array([[1.0, 0.0]])  # Observe position only
        R = jnp.array([[0.2]])  # Observation noise

        return initial_mean, initial_cov, A, Q, C, R

    def create_tfp_lgssm(self, initial_mean, initial_cov, A, Q, C, R, num_timesteps):
        """Create equivalent TFP LinearGaussianStateSpaceModel."""
        # Create time-invariant linear Gaussian SSM using constant matrices
        return tfd.LinearGaussianStateSpaceModel(
            num_timesteps=num_timesteps,
            initial_state_prior=tfd.MultivariateNormalFullCovariance(
                loc=initial_mean, covariance_matrix=initial_cov
            ),
            # Use constant matrices (not functions)
            transition_matrix=A,
            transition_noise=tfd.MultivariateNormalFullCovariance(
                loc=jnp.zeros(A.shape[0]), covariance_matrix=Q
            ),
            observation_matrix=C,
            observation_noise=tfd.MultivariateNormalFullCovariance(
                loc=jnp.zeros(C.shape[0]), covariance_matrix=R
            ),
        )

    def test_marginal_log_prob_simple_case(self):
        """Test marginal log probability against TFP for simple case."""
        initial_mean, initial_cov, A, Q, C, R = self.create_simple_lgssm_params()

        # Test observations of different lengths
        key = jrand.key(42)
        test_sequences = [
            jrand.normal(key, (1, 1)),  # T=1
            jrand.normal(key, (2, 1)),  # T=2
            jrand.normal(key, (3, 1)),  # T=3
            jrand.normal(key, (5, 1)),  # T=5
        ]

        for obs_seq in test_sequences:
            T = len(obs_seq)

            # GenJAX Kalman filter
            _, _, genjax_log_marginal = kalman_filter(
                obs_seq, initial_mean, initial_cov, A, Q, C, R
            )

            # TFP LinearGaussianStateSpaceModel
            tfp_lgssm = self.create_tfp_lgssm(
                initial_mean, initial_cov, A, Q, C, R, num_timesteps=T
            )
            tfp_log_marginal = tfp_lgssm.log_prob(obs_seq)

            # Compare marginal log probabilities
            assert jnp.allclose(
                genjax_log_marginal,
                tfp_log_marginal,
                rtol=1e-6,
                atol=1e-6,
            ), f"Marginal log prob mismatch for sequence length {T}"

    def test_marginal_log_prob_complex_case(self):
        """Test marginal log probability against TFP for complex case."""
        initial_mean, initial_cov, A, Q, C, R = self.create_complex_lgssm_params()

        # Test with longer sequences
        key = jrand.key(123)
        test_sequences = [
            jrand.normal(key, (5, 1)),  # T=5
            jrand.normal(key, (10, 1)),  # T=10
            jrand.normal(key, (15, 1)),  # T=15
        ]

        for obs_seq in test_sequences:
            T = len(obs_seq)

            # GenJAX Kalman filter
            _, _, genjax_log_marginal = kalman_filter(
                obs_seq, initial_mean, initial_cov, A, Q, C, R
            )

            # TFP LinearGaussianStateSpaceModel
            tfp_lgssm = self.create_tfp_lgssm(
                initial_mean, initial_cov, A, Q, C, R, num_timesteps=T
            )
            tfp_log_marginal = tfp_lgssm.log_prob(obs_seq)

            # Compare marginal log probabilities
            assert jnp.allclose(
                genjax_log_marginal,
                tfp_log_marginal,
                rtol=1e-6,
                atol=1e-6,
            ), f"Marginal log prob mismatch for sequence length {T}"

    def test_sample_vs_tfp_statistics(self):
        """Test that sampling statistics roughly match between GenJAX and TFP."""
        initial_mean, initial_cov, A, Q, C, R = self.create_simple_lgssm_params()
        key = jrand.key(42)
        T = 10

        # Generate single sequences multiple times to compare statistics
        seeded_sample = seed(sample_linear_gaussian_dataset)

        # Generate multiple single sequences
        keys = jrand.split(key, 100)  # Generate 100 sequences
        genjax_samples = []
        for k in keys:
            states, observations, _ = seeded_sample(
                k, initial_mean, initial_cov, A, Q, C, R, const(T)
            )
            genjax_samples.append(observations)
        genjax_samples = jnp.stack(genjax_samples)

        # Sample from TFP model
        tfp_lgssm = self.create_tfp_lgssm(
            initial_mean, initial_cov, A, Q, C, R, num_timesteps=T
        )
        tfp_samples = tfp_lgssm.sample(100, seed=key)

        # Compare observation statistics (should be roughly similar)
        genjax_obs_mean = jnp.mean(genjax_samples)
        tfp_obs_mean = jnp.mean(tfp_samples)

        genjax_obs_std = jnp.std(genjax_samples)
        tfp_obs_std = jnp.std(tfp_samples)

        # Loose tolerance since this is statistical
        assert jnp.allclose(
            genjax_obs_mean,
            tfp_obs_mean,
            rtol=0.3,
            atol=0.2,  # 30% relative, 0.2 absolute tolerance
        ), "Observation means differ significantly between GenJAX and TFP"

        assert jnp.allclose(
            genjax_obs_std,
            tfp_obs_std,
            rtol=0.3,
            atol=0.2,  # 30% relative, 0.2 absolute tolerance
        ), "Observation standard deviations differ significantly between GenJAX and TFP"

    def test_sample_vs_analytical_statistics(self):
        """Test that sampling statistics match analytical expectations."""
        initial_mean, initial_cov, A, Q, C, R = self.create_simple_lgssm_params()
        key = jrand.key(42)
        T = 10

        # Generate multiple single sequences
        seeded_sample = seed(sample_linear_gaussian_dataset)
        keys = jrand.split(key, 100)
        observations_list = []
        for k in keys:
            states, observations, _ = seeded_sample(
                k, initial_mean, initial_cov, A, Q, C, R, const(T)
            )
            observations_list.append(observations)
        all_observations = jnp.stack(observations_list)

        # Check observation statistics
        obs_mean = jnp.mean(all_observations)  # Mean over all samples
        obs_std = jnp.std(all_observations)

        # For simple case: observations should be approximately zero-mean
        # (since initial state is zero-mean and dynamics are stable)
        assert jnp.allclose(
            obs_mean,
            0.0,
            rtol=0.3,  # Loose tolerance for statistical test
            atol=0.2,
        ), "Sample observation means deviate from expected values"

        # Standard deviation should be reasonable (not too small or too large)
        assert obs_std > 0.1, "Sample standard deviation too small"
        assert obs_std < 5.0, "Sample standard deviation too large"

    @pytest.mark.parametrize("T", [1, 2, 5, 10])
    def test_different_sequence_lengths(self, T):
        """Test marginal log probability for different sequence lengths."""
        initial_mean, initial_cov, A, Q, C, R = self.create_simple_lgssm_params()
        key = jrand.key(123)

        # Generate random observation sequence
        observations = jrand.normal(key, (T, 1))

        # GenJAX Kalman filter
        _, _, genjax_log_marginal = kalman_filter(
            observations, initial_mean, initial_cov, A, Q, C, R
        )

        # TFP LinearGaussianStateSpaceModel
        tfp_lgssm = self.create_tfp_lgssm(
            initial_mean, initial_cov, A, Q, C, R, num_timesteps=T
        )
        tfp_log_marginal = tfp_lgssm.log_prob(observations)

        # Compare
        assert jnp.allclose(
            genjax_log_marginal,
            tfp_log_marginal,
            rtol=1e-6,
            atol=1e-6,
        ), f"Marginal log prob mismatch for T={T}"

    def test_edge_case_single_timestep(self):
        """Test edge case of single time step (T=1)."""
        initial_mean, initial_cov, A, Q, C, R = self.create_simple_lgssm_params()
        key = jrand.key(42)

        observations = jrand.normal(key, (1, 1))

        # GenJAX Kalman filter
        _, _, genjax_log_marginal = kalman_filter(
            observations, initial_mean, initial_cov, A, Q, C, R
        )

        # Manual calculation for T=1: just the marginal likelihood of observation
        innovation = observations[0] - C @ initial_mean
        innovation_cov = C @ initial_cov @ C.T + R
        manual_log_marginal = jax.scipy.stats.multivariate_normal.logpdf(
            innovation, jnp.zeros_like(innovation), innovation_cov
        )

        # TFP comparison
        tfp_lgssm = self.create_tfp_lgssm(
            initial_mean, initial_cov, A, Q, C, R, num_timesteps=1
        )
        tfp_log_marginal = tfp_lgssm.log_prob(observations)

        # All should match
        assert jnp.allclose(
            genjax_log_marginal,
            manual_log_marginal,
            rtol=1e-15,
            atol=1e-15,
        ), "GenJAX doesn't match manual calculation for T=1"

        assert jnp.allclose(
            genjax_log_marginal,
            tfp_log_marginal,
            rtol=1e-6,
            atol=1e-6,
        ), "GenJAX doesn't match TFP for T=1"

    def test_smoother_consistency(self):
        """Test that Kalman smoother produces valid results."""
        initial_mean, initial_cov, A, Q, C, R = self.create_complex_lgssm_params()
        key = jrand.key(42)
        observations = jrand.normal(key, (10, 1))

        # Run Kalman smoother
        smoothed_means, smoothed_covs = kalman_smoother(
            observations, initial_mean, initial_cov, A, Q, C, R
        )

        # Run Kalman filter for comparison
        filtered_means, filtered_covs, _ = kalman_filter(
            observations, initial_mean, initial_cov, A, Q, C, R
        )

        # Smoothed estimates should generally have lower or equal uncertainty
        # (covariances should be smaller or equal)
        for t in range(len(observations)):
            # Check that smoothed covariances are PSD
            eigenvals = jnp.linalg.eigvals(smoothed_covs[t])
            assert jnp.all(eigenvals >= 0), f"Smoothed covariance not PSD at time {t}"

            # For the final time step, filtered and smoothed should be identical
            if t == len(observations) - 1:
                assert jnp.allclose(
                    smoothed_means[t],
                    filtered_means[t],
                    rtol=1e-10,
                    atol=1e-10,
                ), "Final smoothed mean doesn't match filtered mean"

                assert jnp.allclose(
                    smoothed_covs[t],
                    filtered_covs[t],
                    rtol=1e-10,
                    atol=1e-10,
                ), "Final smoothed covariance doesn't match filtered covariance"

    def test_numerical_stability(self):
        """Test numerical stability with extreme parameter values."""
        # Create parameters with very small process noise (near-deterministic)
        initial_mean = jnp.array([0.0])
        initial_cov = jnp.array([[1e-6]])  # Very small initial uncertainty
        A = jnp.array([[0.999]])  # Nearly unit dynamics
        Q = jnp.array([[1e-8]])  # Very small process noise
        C = jnp.array([[1.0]])
        R = jnp.array([[1e-6]])  # Very small observation noise

        key = jrand.key(42)
        # Long sequence that could cause numerical issues
        observations = jrand.normal(key, (50, 1)) * 1e-3  # Small observations

        # Should not produce NaN or inf
        _, _, genjax_log_marginal = kalman_filter(
            observations, initial_mean, initial_cov, A, Q, C, R
        )

        assert jnp.isfinite(genjax_log_marginal), (
            "Kalman filter produced non-finite result"
        )

        # Compare with TFP
        tfp_lgssm = self.create_tfp_lgssm(
            initial_mean, initial_cov, A, Q, C, R, num_timesteps=len(observations)
        )
        tfp_log_marginal = tfp_lgssm.log_prob(observations)

        assert jnp.allclose(
            genjax_log_marginal,
            tfp_log_marginal,
            rtol=1e-4,  # Looser tolerance for numerical edge cases
            atol=1e-6,
        ), "Numerical stability test failed"

    def test_kernel_function_consistency(self):
        """Test that the kernel generative function produces consistent results."""
        initial_mean, initial_cov, A, Q, C, R = self.create_simple_lgssm_params()
        key = jrand.key(42)
        T = 5

        # Generate dataset using kernel function
        states, observations, _ = seed(sample_linear_gaussian_dataset)(
            key, initial_mean, initial_cov, A, Q, C, R, const(T)
        )

        # Compute log marginal likelihood using generated observations
        _, _, log_marginal = kalman_filter(
            observations.reshape(-1, 1), initial_mean, initial_cov, A, Q, C, R
        )

        # Log marginal should be finite
        assert jnp.isfinite(log_marginal), (
            "Kernel-generated data produces non-finite log marginal"
        )

        # Test unified API functions
        seeded_dataset_fn = seed(linear_gaussian_test_dataset)
        dataset = seeded_dataset_fn(
            jrand.key(123), initial_mean, initial_cov, A, Q, C, R, const(T)
        )

        # Verify dataset structure
        assert "z" in dataset, "Dataset missing latent states"
        assert "obs" in dataset, "Dataset missing observations"
        assert dataset["z"].shape == (T, 1), (
            f"States have wrong shape: {dataset['z'].shape}"
        )
        assert dataset["obs"].shape == (T, 1), (
            f"Observations have wrong shape: {dataset['obs'].shape}"
        )

        # Test exact log marginal computation
        exact_log_marginal = linear_gaussian_exact_log_marginal(
            dataset["obs"], initial_mean, initial_cov, A, Q, C, R
        )

        assert jnp.isfinite(exact_log_marginal), "Exact log marginal is not finite"

    def test_inference_problem_generation(self):
        """Test that inference problem generation works correctly."""
        initial_mean, initial_cov, A, Q, C, R = self.create_complex_lgssm_params()
        T = 8

        # Generate inference problem
        seeded_problem_fn = seed(linear_gaussian_inference_problem)
        dataset, exact_log_marginal = seeded_problem_fn(
            jrand.key(456), initial_mean, initial_cov, A, Q, C, R, const(T)
        )

        # Verify structure
        assert isinstance(dataset, dict), "Dataset should be a dictionary"
        assert "z" in dataset and "obs" in dataset, "Dataset missing required keys"
        assert jnp.isfinite(exact_log_marginal), "Exact log marginal should be finite"

        # Verify dimensions
        assert dataset["z"].shape == (T, 2), (
            f"States have wrong shape: {dataset['z'].shape}"
        )
        assert dataset["obs"].shape == (T, 1), (
            f"Observations have wrong shape: {dataset['obs'].shape}"
        )

        # Test that we can reproduce the exact log marginal
        reproduced_log_marginal = linear_gaussian_exact_log_marginal(
            dataset["obs"], initial_mean, initial_cov, A, Q, C, R
        )

        assert jnp.allclose(
            exact_log_marginal,
            reproduced_log_marginal,
            rtol=1e-10,
            atol=1e-10,
        ), "Cannot reproduce exact log marginal from dataset"


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def simple_lgssm_params():
    """Simple 1D linear Gaussian SSM parameters."""
    initial_mean = jnp.array([0.0])
    initial_cov = jnp.array([[1.0]])
    A = jnp.array([[0.9]])
    Q = jnp.array([[0.1]])
    C = jnp.array([[1.0]])
    R = jnp.array([[0.2]])

    return {
        "initial_mean": initial_mean,
        "initial_cov": initial_cov,
        "A": A,
        "Q": Q,
        "C": C,
        "R": R,
    }


@pytest.fixture
def complex_lgssm_params():
    """Complex 2D state, 1D observation linear Gaussian SSM parameters."""
    initial_mean = jnp.array([0.0, 0.0])
    initial_cov = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    A = jnp.array([[1.0, 1.0], [0.0, 0.9]])
    Q = jnp.array([[0.1, 0.0], [0.0, 0.05]])
    C = jnp.array([[1.0, 0.0]])
    R = jnp.array([[0.2]])

    return {
        "initial_mean": initial_mean,
        "initial_cov": initial_cov,
        "A": A,
        "Q": Q,
        "C": C,
        "R": R,
    }
