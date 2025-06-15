#!/usr/bin/env python3
"""
Simple import validation script for the HMM tests.
This script checks if the test file can be imported without errors.
"""

try:
    # Test basic imports
    import jax
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax as tfp
    print("✓ Basic dependencies imported successfully")
    
    # Test GenJAX imports
    from genjax.core import get_choices
    from genjax.distributions import categorical
    print("✓ GenJAX components imported successfully")
    
    # Test HMM module imports
    from tests.discrete_hmm import (
        discrete_hmm_model_factory,
        forward_filter,
        forward_filtering_backward_sampling,
    )
    print("✓ HMM implementation imported successfully")
    
    # Test the test module imports
    from tests.test_discrete_hmm_tfp import TestDiscreteHMMAgainstTFP
    print("✓ Test suite imported successfully")
    
    # Quick functionality check
    test_class = TestDiscreteHMMAgainstTFP()
    print("✓ Test class instantiated successfully")
    
    print("\n🎉 All imports successful! The test suite should be ready to run.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("The test dependencies may not be properly installed.")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    print("There may be an issue with the test implementation.")