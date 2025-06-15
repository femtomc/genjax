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
import numpy as np

# TensorFlow Probability imports
import tensorflow_probability.substrates.jax as tfp
tfd = tfp.distributions

# GenJAX imports
from genjax.core import get_choices
from genjax.distributions import categorical

# Import HMM implementation from the same directory
from .discrete_hmm import (
    discrete_hmm_model_factory,
    forward_filter,
    forward_filtering_backward_sampling,
    sample_hmm_dataset,
    compute_sequence_log_prob,
)


class TestDiscreteHMMAgainstTFP:
    """Test suite comparing GenJAX discrete HMM against TFP's HiddenMarkovModel."""

    @pytest.fixture
    def simple_hmm_params(self):
        """Simple 2-state, 2-observation HMM parameters for testing."""
        initial_probs = jnp.array([0.6, 0.4])
        transition_matrix = jnp.array([
            [0.7, 0.3],
            [0.4, 0.6]
        ])
        emission_matrix = jnp.array([
            [0.9, 0.1],  # State 0 emits obs 0 with prob 0.9
            [0.2, 0.8]   # State 1 emits obs 1 with prob 0.8
        ])
        return initial_probs, transition_matrix, emission_matrix

    @pytest.fixture
    def complex_hmm_params(self):
        """More complex 3-state, 4-observation HMM parameters."""
        initial_probs = jnp.array([0.5, 0.3, 0.2])
        transition_matrix = jnp.array([
            [0.5, 0.3, 0.2],
            [0.2, 0.6, 0.2],
            [0.1, 0.3, 0.6]
        ])
        emission_matrix = jnp.array([
            [0.4, 0.3, 0.2, 0.1],  # State 0
            [0.1, 0.4, 0.4, 0.1],  # State 1
            [0.1, 0.1, 0.2, 0.6]   # State 2
        ])
        return initial_probs, transition_matrix, emission_matrix

    def create_tfp_hmm(self, initial_probs, transition_matrix, emission_matrix):
        """Create equivalent TFP HiddenMarkovModel."""
        initial_distribution = tfd.Categorical(probs=initial_probs)
        transition_distribution = tfd.Categorical(probs=transition_matrix)
        observation_distribution = tfd.Categorical(probs=emission_matrix)
        
        return tfd.HiddenMarkovModel(
            initial_distribution=initial_distribution,
            transition_distribution=transition_distribution,
            observation_distribution=observation_distribution,
            num_steps=None  # Will be set when calling log_prob
        )

    def test_marginal_log_prob_simple_case(self, simple_hmm_params):
        """Test marginal log probability against TFP for simple case."""
        initial_probs, transition_matrix, emission_matrix = simple_hmm_params
        
        # Test observations of different lengths
        test_sequences = [
            jnp.array([0]),           # T=1
            jnp.array([0, 1]),        # T=2
            jnp.array([0, 1, 0]),     # T=3
            jnp.array([1, 0, 1, 0]),  # T=4
        ]
        
        for obs_seq in test_sequences:
            T = len(obs_seq)
            
            # GenJAX forward filter
            _, genjax_log_marginal = forward_filter(
                obs_seq, initial_probs, transition_matrix, emission_matrix
            )
            
            # TFP HiddenMarkovModel
            tfp_hmm = self.create_tfp_hmm(initial_probs, transition_matrix, emission_matrix)
            tfp_log_marginal = tfp_hmm.log_prob(obs_seq)
            
            # Compare marginal log probabilities
            np.testing.assert_allclose(
                genjax_log_marginal, tfp_log_marginal, 
                rtol=1e-10, atol=1e-10,
                err_msg=f"Marginal log prob mismatch for sequence {obs_seq}"
            )

    def test_marginal_log_prob_complex_case(self, complex_hmm_params):
        """Test marginal log probability against TFP for complex case."""
        initial_probs, transition_matrix, emission_matrix = complex_hmm_params
        
        # Test with longer sequences and different observations
        test_sequences = [
            jnp.array([0, 1, 2, 3]),       # T=4, all different obs
            jnp.array([0, 0, 1, 1, 2, 2]), # T=6, repeated patterns
            jnp.array([3, 2, 1, 0, 1, 2, 3]), # T=7, reverse pattern
        ]
        
        for obs_seq in test_sequences:
            # GenJAX forward filter
            _, genjax_log_marginal = forward_filter(
                obs_seq, initial_probs, transition_matrix, emission_matrix
            )
            
            # TFP HiddenMarkovModel
            tfp_hmm = self.create_tfp_hmm(initial_probs, transition_matrix, emission_matrix)
            tfp_log_marginal = tfp_hmm.log_prob(obs_seq)
            
            # Compare marginal log probabilities
            np.testing.assert_allclose(
                genjax_log_marginal, tfp_log_marginal, 
                rtol=1e-10, atol=1e-10,
                err_msg=f"Marginal log prob mismatch for sequence {obs_seq}"
            )

    def test_generative_model_log_prob_consistency(self, simple_hmm_params):
        """Test that GenJAX generative model assess matches forward filter."""
        initial_probs, transition_matrix, emission_matrix = simple_hmm_params
        T = 5
        
        # Sample from generative model
        discrete_hmm_model = discrete_hmm_model_factory(T)
        trace = discrete_hmm_model.simulate(
            (initial_probs, transition_matrix, emission_matrix)
        )
        
        choices = get_choices(trace)
        observations = trace.get_retval()
        
        # Test assess method consistency
        log_density, _ = discrete_hmm_model.assess(
            (initial_probs, transition_matrix, emission_matrix), choices
        )
        
        # Compare with forward filter marginal
        _, forward_log_marginal = forward_filter(
            observations, initial_probs, transition_matrix, emission_matrix
        )
        
        # The generative model density should match the marginal from forward filter
        np.testing.assert_allclose(
            log_density, forward_log_marginal,
            rtol=1e-10, atol=1e-10,
            err_msg="GenJAX assess density doesn't match forward filter marginal"
        )

    def test_sequence_log_prob_computation(self, simple_hmm_params):
        """Test the compute_sequence_log_prob function against manual calculation."""
        initial_probs, transition_matrix, emission_matrix = simple_hmm_params
        
        # Simple test case: states=[0, 1], observations=[0, 1]
        states = jnp.array([0, 1])
        observations = jnp.array([0, 1])
        
        # Manual calculation
        manual_log_prob = (
            jnp.log(initial_probs[0]) +                    # Initial state
            jnp.log(emission_matrix[0, 0]) +               # Initial emission
            jnp.log(transition_matrix[0, 1]) +             # Transition 0->1
            jnp.log(emission_matrix[1, 1])                 # Final emission
        )
        
        # Function calculation
        computed_log_prob = compute_sequence_log_prob(
            states, observations, initial_probs, transition_matrix, emission_matrix
        )
        
        np.testing.assert_allclose(
            computed_log_prob, manual_log_prob,
            rtol=1e-15, atol=1e-15,
            err_msg="Sequence log prob computation incorrect"
        )

    def test_forward_filter_normalization(self, simple_hmm_params):
        """Test that forward filter probabilities are properly normalized."""
        initial_probs, transition_matrix, emission_matrix = simple_hmm_params
        
        observations = jnp.array([0, 1, 0, 1])
        alpha, _ = forward_filter(
            observations, initial_probs, transition_matrix, emission_matrix
        )
        
        # Each time step should sum to 1 in probability space
        for t in range(len(observations)):
            prob_sum = jnp.sum(jnp.exp(alpha[t]))
            np.testing.assert_allclose(
                prob_sum, 1.0, rtol=1e-10, atol=1e-10,
                err_msg=f"Forward probabilities not normalized at time {t}"
            )

    def test_sample_vs_tfp_statistics(self, simple_hmm_params):
        """Test that sampling statistics roughly match between GenJAX and TFP."""
        initial_probs, transition_matrix, emission_matrix = simple_hmm_params
        key = jrand.key(42)
        T = 10
        n_samples = 1000
        
        # Sample from GenJAX model
        discrete_hmm_model = discrete_hmm_model_factory(T)
        
        genjax_samples = []
        for i in range(n_samples):
            trace = discrete_hmm_model.simulate(
                (initial_probs, transition_matrix, emission_matrix)
            )
            genjax_samples.append(trace.get_retval())
        genjax_samples = jnp.array(genjax_samples)
        
        # Sample from TFP model
        tfp_hmm = self.create_tfp_hmm(initial_probs, transition_matrix, emission_matrix)
        tfp_samples = tfp_hmm.sample(n_samples, seed=key, sample_shape=[T])
        
        # Compare observation frequencies (should be roughly similar)
        genjax_obs_freq = jnp.mean(genjax_samples == 0)  # Frequency of observation 0
        tfp_obs_freq = jnp.mean(tfp_samples == 0)
        
        # Loose tolerance since this is statistical
        np.testing.assert_allclose(
            genjax_obs_freq, tfp_obs_freq,
            rtol=0.1, atol=0.05,  # 10% relative, 5% absolute tolerance
            err_msg="Observation frequencies differ significantly between GenJAX and TFP"
        )

    @pytest.mark.parametrize("T", [1, 2, 5, 10])
    def test_different_sequence_lengths(self, simple_hmm_params, T):
        """Test marginal log probability for different sequence lengths."""
        initial_probs, transition_matrix, emission_matrix = simple_hmm_params
        key = jrand.key(123)
        
        # Generate random observation sequence
        observations = jrand.randint(key, shape=(T,), minval=0, maxval=2)
        
        # GenJAX forward filter
        _, genjax_log_marginal = forward_filter(
            observations, initial_probs, transition_matrix, emission_matrix
        )
        
        # TFP HiddenMarkovModel
        tfp_hmm = self.create_tfp_hmm(initial_probs, transition_matrix, emission_matrix)
        tfp_log_marginal = tfp_hmm.log_prob(observations)
        
        # Compare
        np.testing.assert_allclose(
            genjax_log_marginal, tfp_log_marginal,
            rtol=1e-10, atol=1e-10,
            err_msg=f"Marginal log prob mismatch for T={T}, obs={observations}"
        )

    def test_edge_case_single_timestep(self, simple_hmm_params):
        """Test edge case of single time step (T=1)."""
        initial_probs, transition_matrix, emission_matrix = simple_hmm_params
        
        for obs in [0, 1]:
            observations = jnp.array([obs])
            
            # GenJAX forward filter
            _, genjax_log_marginal = forward_filter(
                observations, initial_probs, transition_matrix, emission_matrix
            )
            
            # Manual calculation for T=1
            manual_log_marginal = jax.scipy.special.logsumexp(
                jnp.log(initial_probs) + jnp.log(emission_matrix[:, obs])
            )
            
            # TFP comparison
            tfp_hmm = self.create_tfp_hmm(initial_probs, transition_matrix, emission_matrix)
            tfp_log_marginal = tfp_hmm.log_prob(observations)
            
            # All should match
            np.testing.assert_allclose(
                genjax_log_marginal, manual_log_marginal,
                rtol=1e-15, atol=1e-15,
                err_msg=f"GenJAX doesn't match manual for T=1, obs={obs}"
            )
            
            np.testing.assert_allclose(
                genjax_log_marginal, tfp_log_marginal,
                rtol=1e-10, atol=1e-10,
                err_msg=f"GenJAX doesn't match TFP for T=1, obs={obs}"
            )

    def test_ffbs_consistency(self, simple_hmm_params):
        """Test that FFBS produces valid samples with correct log probabilities."""
        initial_probs, transition_matrix, emission_matrix = simple_hmm_params
        
        # Generate test observations
        observations = jnp.array([0, 1, 0, 1])
        
        # Run FFBS
        hmm_trace = forward_filtering_backward_sampling(
            observations, initial_probs, transition_matrix, emission_matrix
        )
        
        # Verify the log probability matches what we'd compute independently
        computed_log_prob = compute_sequence_log_prob(
            hmm_trace.states, observations, 
            initial_probs, transition_matrix, emission_matrix
        )
        
        np.testing.assert_allclose(
            hmm_trace.log_prob, computed_log_prob,
            rtol=1e-10, atol=1e-10,
            err_msg="FFBS log probability doesn't match independent computation"
        )
        
        # Verify observations match
        np.testing.assert_array_equal(
            hmm_trace.observations, observations,
            err_msg="FFBS observations don't match input"
        )

    def test_numerical_stability(self):
        """Test numerical stability with extreme probability values."""
        # Create parameters with very small probabilities
        initial_probs = jnp.array([0.999, 0.001])
        transition_matrix = jnp.array([
            [0.999, 0.001],
            [0.001, 0.999]
        ])
        emission_matrix = jnp.array([
            [0.999, 0.001],
            [0.001, 0.999]
        ])
        
        # Long sequence that would cause underflow in probability space
        observations = jnp.array([0] * 20 + [1] * 20)
        
        # Should not produce NaN or inf
        _, genjax_log_marginal = forward_filter(
            observations, initial_probs, transition_matrix, emission_matrix
        )
        
        assert jnp.isfinite(genjax_log_marginal), "Forward filter produced non-finite result"
        
        # Compare with TFP
        tfp_hmm = self.create_tfp_hmm(initial_probs, transition_matrix, emission_matrix)
        tfp_log_marginal = tfp_hmm.log_prob(observations)
        
        np.testing.assert_allclose(
            genjax_log_marginal, tfp_log_marginal,
            rtol=1e-8, atol=1e-8,  # Slightly looser tolerance for numerical edge cases
            err_msg="Numerical stability test failed"
        )


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])