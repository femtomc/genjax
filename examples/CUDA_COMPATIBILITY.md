# CUDA Compatibility Report for GenJAX Examples

This document summarizes the CUDA compatibility status of all GenJAX examples as of 2025-06-21.

## Summary

Most examples run successfully on CUDA, though some require parameter adjustments or have specific issues that need addressing.

## Compatibility Status

### ✅ Fully Compatible (Works Out of the Box)

1. **faircoin** - Beta-Bernoulli framework comparison
   - Command: `pixi run -e cuda cuda-faircoin`
   - Status: Runs perfectly with no issues
   - Performance: Good speedup on GPU

2. **intuitive-physics** - Physics simulation inference
   - Command: `pixi run -e intuitive-psych-cuda python -m examples.intuitive_physics.main`
   - Status: Works well for all analysis types
   - Performance: Benefits from GPU acceleration

### ⚠️ Compatible with Caveats

3. **curvefit** - Curve fitting with multiple frameworks
   - Command: Must use `pixi run -e curvefit-cuda` (not generic cuda environment)
   - Issue: Requires NumPyro which is only in the curvefit-specific environment
   - Status: Works when using the correct environment

4. **localization** - Particle filter localization
   - Command: `pixi run -e localization-cuda python -m examples.localization.main`
   - Issue: Default parameters are computationally intensive
   - Solution: Use reduced parameters for testing:
     ```bash
     pixi run -e localization-cuda python -m examples.localization.main generate-data --n-steps 3 --n-particles 10 --timing-repeats 1
     ```
   - Status: Works well with adjusted parameters

5. **gol** (Game of Life) - Inference on Game of Life patterns
   - Command: `pixi run -e gol-cuda gol-quick`
   - Issues:
     - Generates many JAX FutureWarnings about int32->bool conversions
     - Runs very slowly even with reduced parameters
   - Status: Functional but needs optimization

6. **gen2d** - 2D generative models
   - Command: `pixi run -e gen2d-cuda python -m examples.gen2d.main`
   - Issue: Had relative import problems
   - Solution: Fixed by converting to absolute imports
   - Status: Now works correctly after fixes

7. **programmable-mcts** - Monte Carlo Tree Search
   - Command: `pixi run -e programmable-mcts-cuda python -m examples.programmable_mcts.main`
   - Issue: Times out with default parameters (computationally intensive)
   - Status: Likely works with reduced parameters but needs investigation

### ❌ Broken (Needs Code Fixes)

8. **state-space** - State space models
   - Command: `pixi run -e state-space-cuda state-space-quick`
   - Error: `TypeError: hmm_proposal() takes 5 positional arguments but 7 were given`
   - Issue: Argument mismatch in the proposal function
   - Status: Requires code fix to align proposal function signature

## Recommendations

1. **Immediate Actions**:
   - Fix the state-space example's proposal function signature
   - Add warning suppression for Game of Life's dtype warnings
   - Create quick-test commands for computationally intensive examples

2. **Documentation Updates**:
   - Add CUDA-specific instructions to each example's README
   - Document recommended parameters for GPU testing
   - Note environment-specific requirements (e.g., curvefit needs NumPyro)

3. **Performance Optimization**:
   - Profile and optimize Game of Life example
   - Add progress indicators for long-running examples
   - Consider adding GPU-specific optimizations where beneficial

## Testing Commands

For quick CUDA compatibility testing, use these commands:

```bash
# Fast examples that work well
pixi run -e cuda cuda-faircoin
pixi run -e intuitive-psych-cuda python -m examples.intuitive_physics.main --environment

# Examples needing specific environments or parameters
pixi run -e curvefit-cuda curvefit  # Needs curvefit environment for NumPyro
pixi run -e localization-cuda python -m examples.localization.main generate-data --n-steps 3 --n-particles 10
pixi run -e gen2d-cuda python -m examples.gen2d.main --n-frames 5 --n-particles 20 --data

# Slow but functional
pixi run -e gol-cuda gol-quick  # Generates warnings but works

# Currently broken
# pixi run -e state-space-cuda state-space-quick  # TypeError - needs fix
```
