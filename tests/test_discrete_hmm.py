"""
Test discrete HMM implementation against TensorFlow Probability.

This module validates GenJAX's discrete HMM implementation by comparing
marginal log probabilities and sampling behavior against TensorFlow
Probability's HiddenMarkovModel distribution.
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

# Import HMM implementation from extras module
from genjax.extras.state_space import (
    forward_filter,
    forward_filtering_backward_sampling,
    compute_sequence_log_prob,
    sample_hmm_dataset,
)

tfd = tfp.distributions

# JIT-compiled versions of HMM functions for performance
_jitted_forward_filter = jax.jit(forward_filter)
_jitted_compute_sequence_log_prob = jax.jit(compute_sequence_log_prob)


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
            num_steps=num_steps,  # Will be set when calling log_prob
        )

    def test_marginal_log_prob_simple_case(self, simple_hmm_params, jitted_hmm_ops):
        """Test marginal log probability against TFP for simple case."""
        initial_probs = simple_hmm_params["initial_probs"]
        transition_matrix = simple_hmm_params["transition_matrix"]
        emission_matrix = simple_hmm_params["emission_matrix"]

        # Test observations of different lengths
        test_sequences = [
            jnp.array([0]),  # T=1
            jnp.array([0, 1]),  # T=2
            jnp.array([0, 1, 0]),  # T=3
            jnp.array([1, 0, 1, 0]),  # T=4
        ]

        # Use JIT-compiled forward filter from fixture
        forward_filter_jit = jitted_hmm_ops["forward_filter"]

        for obs_seq in test_sequences:
            T = len(obs_seq)

            # GenJAX forward filter (JIT-compiled)
            _, genjax_log_marginal = forward_filter_jit(
                obs_seq, initial_probs, transition_matrix, emission_matrix
            )

            # TFP HiddenMarkovModel
            tfp_hmm = self.create_tfp_hmm(
                initial_probs,
                transition_matrix,
                emission_matrix,
                num_steps=T,
            )
            tfp_log_marginal = tfp_hmm.log_prob(obs_seq)

            # Compare marginal log probabilities
            assert jnp.allclose(
                genjax_log_marginal,
                tfp_log_marginal,
                rtol=1e-6,
                atol=1e-6,
            ), f"Marginal log prob mismatch for sequence {obs_seq}"

    def test_marginal_log_prob_complex_case(self, complex_hmm_params, jitted_hmm_ops):
        """Test marginal log probability against TFP for complex case."""
        initial_probs = complex_hmm_params["initial_probs"]
        transition_matrix = complex_hmm_params["transition_matrix"]
        emission_matrix = complex_hmm_params["emission_matrix"]

        # Test with longer sequences and different observations
        test_sequences = [
            jnp.array([0, 1, 2, 3]),  # T=4, all different obs
            jnp.array([0, 0, 1, 1, 2, 2]),  # T=6, repeated patterns
            jnp.array([3, 2, 1, 0, 1, 2, 3]),  # T=7, reverse pattern
        ]

        # Use JIT-compiled forward filter
        forward_filter_jit = jitted_hmm_ops["forward_filter"]

        for obs_seq in test_sequences:
            # GenJAX forward filter (JIT-compiled)
            _, genjax_log_marginal = forward_filter_jit(
                obs_seq, initial_probs, transition_matrix, emission_matrix
            )

            # TFP HiddenMarkovModel
            tfp_hmm = self.create_tfp_hmm(
                initial_probs,
                transition_matrix,
                emission_matrix,
                num_steps=len(obs_seq),
            )
            tfp_log_marginal = tfp_hmm.log_prob(obs_seq)

            # Compare marginal log probabilities
            assert jnp.allclose(
                genjax_log_marginal,
                tfp_log_marginal,
                rtol=1e-6,
                atol=1e-6,
            ), f"Marginal log prob mismatch for sequence {obs_seq}"

    def test_sequence_log_prob_computation(self, simple_hmm_params):
        """Test the compute_sequence_log_prob function against manual calculation."""
        initial_probs = simple_hmm_params["initial_probs"]
        transition_matrix = simple_hmm_params["transition_matrix"]
        emission_matrix = simple_hmm_params["emission_matrix"]

        # Simple test case: states=[0, 1], observations=[0, 1]
        states = jnp.array([0, 1])
        observations = jnp.array([0, 1])

        # Manual calculation
        manual_log_prob = (
            jnp.log(initial_probs[0])  # Initial state
            + jnp.log(emission_matrix[0, 0])  # Initial emission
            + jnp.log(transition_matrix[0, 1])  # Transition 0->1
            + jnp.log(emission_matrix[1, 1])  # Final emission
        )

        # Function calculation (use JIT-compiled version)
        computed_log_prob = _jitted_compute_sequence_log_prob(
            states, observations, initial_probs, transition_matrix, emission_matrix
        )

        assert jnp.allclose(
            computed_log_prob,
            manual_log_prob,
            rtol=1e-15,
            atol=1e-15,
        ), "Sequence log prob computation incorrect"

    def test_forward_filter_normalization(self, simple_hmm_params, jitted_hmm_ops):
        """Test that forward filter probabilities are properly normalized."""
        initial_probs = simple_hmm_params["initial_probs"]
        transition_matrix = simple_hmm_params["transition_matrix"]
        emission_matrix = simple_hmm_params["emission_matrix"]

        observations = jnp.array([0, 1, 0, 1])
        forward_filter_jit = jitted_hmm_ops["forward_filter"]
        alpha, _ = forward_filter_jit(
            observations, initial_probs, transition_matrix, emission_matrix
        )

        # Each time step should sum to 1 in probability space
        for t in range(len(observations)):
            prob_sum = jnp.sum(jnp.exp(alpha[t]))
            assert jnp.allclose(
                prob_sum,
                1.0,
                rtol=1e-6,
                atol=1e-6,
            ), f"Forward probabilities not normalized at time {t}"

    def test_sample_vs_tfp_statistics(self, simple_hmm_params):
        """Test that sampling statistics roughly match between GenJAX and TFP."""
        initial_probs = simple_hmm_params["initial_probs"]
        transition_matrix = simple_hmm_params["transition_matrix"]
        emission_matrix = simple_hmm_params["emission_matrix"]
        key = jrand.key(42)
        T = const(10)  # Use const for static value
        n_samples = const(1000)  # Use const for static value

        # JIT-compile the sampling function
        @jax.jit
        def sample_hmm_jit(key):
            seeded_sample = seed(sample_hmm_dataset)
            return seeded_sample(
                key,
                initial_probs,
                transition_matrix,
                emission_matrix,
                T,
                n_samples,
            )

        # Use JIT-compiled sampling
        _, genjax_samples, _ = sample_hmm_jit(key)

        # Sample from TFP model
        tfp_hmm = self.create_tfp_hmm(
            initial_probs, transition_matrix, emission_matrix, num_steps=10
        )
        tfp_samples = tfp_hmm.sample(1000, seed=key)

        # Compare observation frequencies (should be roughly similar)
        genjax_obs_freq = jnp.mean(genjax_samples == 0)  # Frequency of observation 0
        tfp_obs_freq = jnp.mean(tfp_samples == 0)

        # Loose tolerance since this is statistical
        assert jnp.allclose(
            genjax_obs_freq,
            tfp_obs_freq,
            rtol=0.1,
            atol=0.05,  # 10% relative, 5% absolute tolerance
        ), "Observation frequencies differ significantly between GenJAX and TFP"

    @pytest.mark.parametrize("T", [1, 2, 5, 10])
    def test_different_sequence_lengths(self, simple_hmm_params, T, jitted_hmm_ops):
        """Test marginal log probability for different sequence lengths."""
        initial_probs = simple_hmm_params["initial_probs"]
        transition_matrix = simple_hmm_params["transition_matrix"]
        emission_matrix = simple_hmm_params["emission_matrix"]
        key = jrand.key(123)

        # Generate random observation sequence
        observations = jrand.randint(key, shape=(T,), minval=0, maxval=2)

        # GenJAX forward filter (JIT-compiled)
        forward_filter_jit = jitted_hmm_ops["forward_filter"]
        _, genjax_log_marginal = forward_filter_jit(
            observations, initial_probs, transition_matrix, emission_matrix
        )

        # TFP HiddenMarkovModel
        tfp_hmm = self.create_tfp_hmm(
            initial_probs,
            transition_matrix,
            emission_matrix,
            num_steps=T,
        )
        tfp_log_marginal = tfp_hmm.log_prob(observations)

        # Compare
        assert jnp.allclose(
            genjax_log_marginal,
            tfp_log_marginal,
            rtol=1e-6,
            atol=1e-6,
        ), f"Marginal log prob mismatch for T={T}, obs={observations}"

    def test_edge_case_single_timestep(self, simple_hmm_params, jitted_hmm_ops):
        """Test edge case of single time step (T=1)."""
        initial_probs = simple_hmm_params["initial_probs"]
        transition_matrix = simple_hmm_params["transition_matrix"]
        emission_matrix = simple_hmm_params["emission_matrix"]

        forward_filter_jit = jitted_hmm_ops["forward_filter"]

        for obs in [0, 1]:
            observations = jnp.array([obs])

            # GenJAX forward filter (JIT-compiled)
            _, genjax_log_marginal = forward_filter_jit(
                observations, initial_probs, transition_matrix, emission_matrix
            )

            # Manual calculation for T=1
            manual_log_marginal = jax.scipy.special.logsumexp(
                jnp.log(initial_probs) + jnp.log(emission_matrix[:, obs])
            )

            # TFP comparison
            tfp_hmm = self.create_tfp_hmm(
                initial_probs,
                transition_matrix,
                emission_matrix,
                num_steps=1,
            )
            tfp_log_marginal = tfp_hmm.log_prob(observations)

            # All should match
            assert jnp.allclose(
                genjax_log_marginal,
                manual_log_marginal,
                rtol=1e-15,
                atol=1e-15,
            ), f"GenJAX doesn't match manual for T=1, obs={obs}"

            assert jnp.allclose(
                genjax_log_marginal,
                tfp_log_marginal,
                rtol=1e-6,
                atol=1e-6,
            ), f"GenJAX doesn't match TFP for T=1, obs={obs}"

    def test_ffbs_consistency(self, simple_hmm_params):
        """Test that FFBS produces valid samples with correct log probabilities."""
        initial_probs = simple_hmm_params["initial_probs"]
        transition_matrix = simple_hmm_params["transition_matrix"]
        emission_matrix = simple_hmm_params["emission_matrix"]
        key = jrand.key(42)

        # Generate test observations
        observations = jnp.array([0, 1, 0, 1])

        # JIT-compile the FFBS closure
        @jax.jit
        def ffbs_jit(key):
            def ffbs_closure():
                return forward_filtering_backward_sampling(
                    observations, initial_probs, transition_matrix, emission_matrix
                )

            return seed(ffbs_closure)(key)

        # Run FFBS with JIT
        hmm_trace = ffbs_jit(key)

        # Verify the log probability matches what we'd compute independently (JIT)
        computed_log_prob = _jitted_compute_sequence_log_prob(
            hmm_trace.states,
            observations,
            initial_probs,
            transition_matrix,
            emission_matrix,
        )

        assert jnp.allclose(
            hmm_trace.log_prob,
            computed_log_prob,
            rtol=1e-10,
            atol=1e-10,
        ), "FFBS log probability doesn't match independent computation"

        # Verify observations match
        assert jnp.array_equal(
            hmm_trace.observations,
            observations,
        ), "FFBS observations don't match input"

    def test_numerical_stability(self, jitted_hmm_ops):
        """Test numerical stability with extreme probability values."""
        # Create parameters with very small probabilities
        initial_probs = jnp.array([0.999, 0.001])
        transition_matrix = jnp.array([[0.999, 0.001], [0.001, 0.999]])
        emission_matrix = jnp.array([[0.999, 0.001], [0.001, 0.999]])

        # Long sequence that would cause underflow in probability space
        observations = jnp.array([0] * 20 + [1] * 20)

        # Should not produce NaN or inf (use JIT)
        forward_filter_jit = jitted_hmm_ops["forward_filter"]
        _, genjax_log_marginal = forward_filter_jit(
            observations, initial_probs, transition_matrix, emission_matrix
        )

        assert jnp.isfinite(genjax_log_marginal), (
            "Forward filter produced non-finite result"
        )

        # Compare with TFP
        tfp_hmm = self.create_tfp_hmm(
            initial_probs,
            transition_matrix,
            emission_matrix,
            num_steps=len(observations),
        )
        tfp_log_marginal = tfp_hmm.log_prob(observations)

        assert jnp.allclose(
            genjax_log_marginal,
            tfp_log_marginal,
            rtol=1e-6,
            atol=1e-6,  # Looser tolerance for numerical edge cases
        ), "Numerical stability test failed"
