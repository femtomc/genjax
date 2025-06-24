"""
Stochastic Probabilities (SP) for GenJAX

This module implements SPDistribution following the design of GenSP.jl,
enabling probabilistic programming with importance-weighted samples.
SP distributions produce weighted samples that enable unbiased estimation
of probabilities and expectations.

References:
    - "Probabilistic Programming with Stochastic Probabilities"
      Alexander K. Lew, Matin Ghavami, Martin Rinard, Vikash K. Mansinghka
    - GenSP.jl: https://github.com/probcomp/GenSP.jl
"""

import jax
import jax.numpy as jnp
import beartype.typing as btyping
from abc import ABC, abstractmethod

from genjax.core import (
    GFI, Trace, Pytree, X, R, Score, Density, 
    Const, const, Selection, sel
)
from genjax.pjax import wrap_sampler, wrap_logpdf
from genjax.distributions import categorical

# Type variables
T = btyping.TypeVar('T')  # Value type for SPDistribution
TypeVar = btyping.TypeVar
Generic = btyping.Generic
Tuple = btyping.Tuple
Any = btyping.Any
Optional = btyping.Optional


class SPDistribution(GFI[X, X], ABC):
    """
    Abstract base class for Stochastic Probability distributions.
    
    SPDistributions extend the GFI interface with importance-weighted
    sampling. Instead of implementing simulate/assess directly,
    subclasses implement random_weighted and estimate_logpdf.
    
    Note: SPDistribution is GFI[X, X] - its return value is the same as
    its choices (like Distribution in core.py).
    """
    
    @abstractmethod
    def random_weighted(self, *args, **kwargs) -> Tuple[X, jnp.ndarray]:
        """
        Sample a value and compute its importance weight.
        
        Returns:
            value: Sampled value of type X
            weight: Importance weight (not log weight)
        """
        pass
    
    @abstractmethod
    def estimate_logpdf(self, value: X, *args, **kwargs) -> jnp.ndarray:
        """
        Estimate the log probability density of a value.
        
        Args:
            value: Value to estimate density for
            
        Returns:
            Log probability density estimate
        """
        pass
    
    # Implement GFI methods using SP primitives
    def simulate(self, *args, **kwargs) -> Trace[X, X]:
        """Simulate by calling random_weighted."""
        value, weight = self.random_weighted(*args, **kwargs)
        
        # SP distributions have choices that equal their return value
        from genjax.core import Tr
        trace = Tr(
            _gen_fn=self,
            _args=(args, kwargs),
            _choices=value,  # Choices are the sampled value
            _retval=value,   # Return value is the same
            _score=-jnp.log(weight)  # Score is negative log weight
        )
        return trace
    
    def assess(self, x: X, *args, **kwargs) -> Tuple[Density, X]:
        """Assess using estimate_logpdf."""
        # For SP distributions, x is the value to assess
        log_density = self.estimate_logpdf(x, *args, **kwargs)
        return log_density, x
    
    def generate(self, x: Optional[X], *args, **kwargs) -> Tuple[Trace[X, X], Score]:
        """Generate - for SP distributions this is just simulate."""
        if x is None:
            trace = self.simulate(*args, **kwargs)
            return trace, jnp.array(0.0)
        else:
            # If constrained, assess and create trace
            log_density, retval = self.assess(x, *args, **kwargs)
            from genjax.core import Tr
            trace = Tr(
                _gen_fn=self,
                _args=(args, kwargs),
                _choices=x,
                _retval=retval,
                _score=-log_density
            )
            return trace, log_density
    
    def update(self, tr: Trace[X, X], x_: X, *args, **kwargs):
        """Update not supported for SP distributions."""
        raise NotImplementedError("SPDistribution does not support update")
    
    def regenerate(self, tr: Trace[X, X], s: Selection, *args, **kwargs):
        """Regenerate not supported for SP distributions."""
        raise NotImplementedError("SPDistribution does not support regenerate")
    
    def merge(self, x: X, x_: X, check: Optional[jnp.ndarray] = None):
        """Merge - SP distributions have no internal choices."""
        return {}, {}
    
    def filter(self, x: X, selection: Selection):
        """Filter - SP distributions have no internal choices."""
        return None, None
    
    def get_selection(self, x: X) -> Selection:
        """Get selection for SP distribution choices.
        
        For SPDistribution, we should determine the selection based on the
        actual structure of x. This is implementation-specific.
        """
        # Delegate to standalone function
        return get_selection(x)


class SMCAlgorithm(SPDistribution):
    """
    Abstract base class for SMC-based SP distributions.
    
    Extends SPDistribution with composable SMC functionality,
    bridging to the GenJAX inference.smc module.
    """
    
    @abstractmethod
    def run_smc(self, n_particles: int, **smc_kwargs):
        """
        Run standard Sequential Monte Carlo algorithm.
        
        Args:
            key: JAX random key
            n_particles: Number of particles to use
            **smc_kwargs: Additional arguments for SMC algorithm
            
        Returns:
            ParticleCollection from inference.smc module
        """
        pass
    
    @abstractmethod 
    def run_csmc(self, n_particles: int, retained_choices: X, **smc_kwargs):
        """
        Run Conditional Sequential Monte Carlo algorithm.
        
        Ensures one particle follows the retained trajectory while
        maintaining proper importance weighting.
        
        Args:
            key: JAX random key
            n_particles: Number of particles to use
            retained_choices: Choices for the retained particle trajectory
            **smc_kwargs: Additional arguments for SMC algorithm
            
        Returns:
            ParticleCollection where one particle matches retained_choices
        """
        pass


@Pytree.dataclass
class Target(Pytree):
    """
    Represents an unnormalized target distribution.
    
    Combines a generative function with arguments and observations
    to define a posterior distribution over latent variables.
    """
    model: GFI[X, R]
    args: Tuple[Any, ...]
    observations: X
    
    def get_latents(self, trace: Trace[X, R]) -> X:
        """Extract latent (unobserved) choices from a trace."""
        all_choices = trace.get_choices()
        # Filter out observed addresses
        latents, _ = self.model.filter(all_choices, ~self._obs_selection())
        return latents
    
    def _obs_selection(self) -> Selection:
        """Create selection for observed addresses."""
        # Use standalone get_selection function to build selection from observations
        return get_selection(self.observations)


@Pytree.dataclass
class ImportanceSampling(SMCAlgorithm, Pytree):
    """
    Importance sampling as an SPDistribution.
    
    Samples from a target distribution using a proposal distribution
    and importance weighting.
    """
    target: Target
    proposal: Optional[GFI[X, Any]] = None
    n_particles: Const[int] = const(100)
    
    def random_weighted(self, *args, **kwargs) -> Tuple[X, jnp.ndarray]:
        """Sample using importance sampling with vectorization."""
        from genjax.pjax import modular_vmap
        
        if self.proposal is None:
            # Use target's internal proposal
            def single_particle(_):
                return self.target.model.generate(
                    self.target.observations, 
                    *self.target.args
                )
            
            # Vectorize over n_particles
            vectorized_generate = modular_vmap(single_particle, in_axes=(0,))
            traces, log_weights = vectorized_generate(jnp.arange(self.n_particles.value))
        else:
            # Use custom proposal
            def single_particle_custom(_):
                # Sample from proposal
                proposal_trace = self.proposal.simulate(*self.target.args)
                proposal_choices = proposal_trace.get_choices()
                
                # Merge with observations
                merged_choices, _ = self.target.model.merge(
                    proposal_choices, 
                    self.target.observations
                )
                
                # Generate from target
                target_trace, target_weight = self.target.model.generate(
                    merged_choices, 
                    *self.target.args
                )
                
                # Compute importance weight
                proposal_score = proposal_trace.get_score()
                log_weight = target_weight + proposal_score
                
                return target_trace, log_weight
            
            # Vectorize
            vectorized_generate = modular_vmap(single_particle_custom, in_axes=(0,))
            traces, log_weights = vectorized_generate(jnp.arange(self.n_particles.value))
        
        # Sample particle according to weights
        log_probs = log_weights - jax.scipy.special.logsumexp(log_weights)
        idx = categorical.sample(log_probs)
        
        # Extract latent choices from selected particle
        # Use tree_map to index into vectorized trace structure
        selected_trace_choices = jax.tree.map(lambda x: x[idx], traces.get_choices())
        
        # Get latents - don't assume structure, use target's method
        selected_trace = traces.__class__(
            _gen_fn=traces._gen_fn,
            _args=traces._args,
            _choices=selected_trace_choices,
            _retval=jax.tree.map(lambda x: x[idx], traces._retval),
            _score=traces._score[idx]
        )
        latent_choices = self.target.get_latents(selected_trace)
        
        # Compute weight estimate
        weight = jnp.exp(jax.scipy.special.logsumexp(log_weights) - jnp.log(self.n_particles.value))
        
        return latent_choices, weight
    
    def estimate_logpdf(self, value: X, *args, **kwargs) -> jnp.ndarray:
        """Estimate log probability using importance sampling with vectorization."""
        from genjax.pjax import modular_vmap
        
        # Merge value with observations once
        merged_choices, _ = self.target.model.merge(value, self.target.observations)
        
        if self.proposal is None:
            # Assess directly on target
            def single_assess(_):
                log_density, _ = self.target.model.assess(
                    merged_choices,
                    *self.target.args
                )
                return log_density
            
            # Vectorize
            vectorized_assess = modular_vmap(single_assess, in_axes=(0,))
            log_weights = vectorized_assess(jnp.arange(self.n_particles.value))
        else:
            # Compute importance weights
            def single_importance(_):
                # Assess target
                target_log_density, _ = self.target.model.assess(
                    merged_choices,
                    *self.target.args
                )
                
                # Assess proposal  
                proposal_log_density, _ = self.proposal.assess(
                    value,
                    *self.target.args
                )
                
                return target_log_density - proposal_log_density
            
            # Vectorize
            vectorized_importance = modular_vmap(single_importance, in_axes=(0,))
            log_weights = vectorized_importance(jnp.arange(self.n_particles.value))
        
        # Estimate log probability
        log_prob_estimate = jax.scipy.special.logsumexp(log_weights) - jnp.log(self.n_particles.value)
        
        return log_prob_estimate
    
    def run_smc(self, n_particles: int, **smc_kwargs):
        """
        Run SMC algorithm using existing importance sampling implementation.
        
        Bridges to GenJAX inference.smc.init functionality.
        """
        from genjax.inference.smc import init
        from genjax.core import const
        
        return init(
            target_gf=self.target.model,
            target_args=self.target.args,
            n_samples=const(n_particles),
            constraints=self.target.observations,
            proposal_gf=self.proposal,
        )
    
    def run_csmc(self, n_particles: int, retained_choices: X, **smc_kwargs):
        """
        Run Conditional SMC algorithm with retained particle.
        
        Uses conditional SMC functionality from inference.smc module.
        """
        from genjax.inference.smc import init_csmc
        from genjax.core import const
        
        return init_csmc(
            target_gf=self.target.model,
            target_args=self.target.args,
            n_samples=const(n_particles),
            constraints=self.target.observations,
            retained_choices=retained_choices,
            proposal_gf=self.proposal,
        )


@Pytree.dataclass
class Marginal(SPDistribution, Pytree):
    """
    Marginal distribution over a specific address using an SMC algorithm.
    
    Following GenSP.jl design: parameterized by an algorithm that handles
    the actual inference, while Marginal specifies which address to marginalize.
    
    Returns the value at the specified address extracted from algorithm samples.
    """
    algorithm: SMCAlgorithm
    address: str | Tuple[str, ...]
    
    def random_weighted(self, *args, **kwargs) -> Tuple[X, jnp.ndarray]:
        """Sample from marginal distribution using the algorithm."""
        # Use the algorithm's own random_weighted method to get a sample
        full_sample, weight = self.algorithm.random_weighted(*args, **kwargs)
        
        # Extract the value at the specified address
        marginal_value = self._extract_address(full_sample, self.address)
        
        return marginal_value, weight
    
    def _extract_address(self, choices: X, address: str | Tuple[str, ...]) -> X:
        """Extract value at specific address from choices."""
        if isinstance(address, tuple):
            # Tuple address - navigate hierarchically
            values = choices
            for part in address:
                if isinstance(values, dict) and part in values:
                    values = values[part]
                else:
                    raise KeyError(f"Address {address} not found in choices")
            return values
        else:
            # String address - check if it contains "/" for hierarchical navigation
            if "/" in address:
                # Parse hierarchical string address
                parts = address.split("/")
                values = choices
                for part in parts:
                    if isinstance(values, dict) and part in values:
                        values = values[part]
                    else:
                        raise KeyError(f"Address {address} not found in choices")
                return values
            else:
                # Single-level string address
                if isinstance(choices, dict) and address in choices:
                    return choices[address]
                else:
                    raise KeyError(f"Address {address} not found in choices")
    
    def estimate_logpdf(self, value: X, *args, **kwargs) -> jnp.ndarray:
        """Estimate marginal log probability using the algorithm."""
        # Build constraint dictionary with value at the marginal address
        constraints = self._build_constraint(value, self.address)
        
        # Use the algorithm to estimate the log density
        # Note: This is a simplified approach. A full implementation would need
        # to properly handle the marginalization in the density estimation.
        return self.algorithm.estimate_logpdf(constraints, *args, **kwargs)
    
    def _build_constraint(self, value: X, address: str | Tuple[str, ...]) -> dict:
        """Build constraint dictionary with value at specified address."""
        if isinstance(address, tuple):
            # Build nested constraint dictionary for hierarchical address
            constraints = {}
            current = constraints
            for part in address[:-1]:
                current[part] = {}
                current = current[part]
            current[address[-1]] = value
            return constraints
        elif "/" in str(address):
            # Parse hierarchical string address
            parts = address.split("/")
            constraints = {}
            current = constraints
            for part in parts[:-1]:
                current[part] = {}
                current = current[part]
            current[parts[-1]] = value
            return constraints
        else:
            # Simple single-level constraint
            return {address: value}


# Standalone get_selection function

def get_selection(x: X) -> Selection:
    """Create a Selection object from a choice map.
    
    This function creates a Selection that matches all addresses present
    in the given choice map structure. It handles different types of
    choice maps used by various generative functions:
    
    - None: Returns NoneSel (matches no addresses)
    - dict: Returns selection matching all keys in the dictionary
    - other: Returns AllSel (matches all addresses)
    
    Args:
        x: Choice map structure (could be None, dict, or other types)
        
    Returns:
        Selection object matching addresses in the choice map
    """
    from genjax.core import Selection, AllSel, NoneSel, DictSel
    
    if x is None:
        # No choices - match nothing
        return Selection(NoneSel())
    elif isinstance(x, dict):
        # Dictionary of choices - create selection from keys
        if not x:
            # Empty dict - match nothing
            return Selection(NoneSel())
        else:
            # Create nested selection structure matching dict structure
            sel_dict = {}
            for key, value in x.items():
                if isinstance(value, dict):
                    # Recursively handle nested dicts
                    sel_dict[key] = get_selection(value)
                else:
                    # Leaf value - select this key
                    sel_dict[key] = Selection(AllSel())
            return Selection(DictSel(sel_dict))
    else:
        # Other types (e.g., raw values from Distribution) - match all
        return Selection(AllSel())


# Helper functions for creating SP distributions

def importance_sampling(
    target: Target,
    proposal: Optional[GFI[X, Any]] = None,
    n_particles: int = 100
) -> "ImportanceSampling":
    """
    Create an importance sampling SPDistribution.
    
    Args:
        target: Target distribution to sample from
        proposal: Optional custom proposal (uses target's internal if None)
        n_particles: Number of particles for importance sampling
        
    Returns:
        ImportanceSampling SPDistribution
    """
    return ImportanceSampling(
        target=target,
        proposal=proposal,
        n_particles=const(n_particles)
    )


def marginal(
    algorithm: SMCAlgorithm,
    address: str | Tuple[str, ...]
) -> "Marginal":
    """
    Create a marginal distribution over a specific address using an algorithm.
    
    Following GenSP.jl design: parameterized by an algorithm.
    
    Args:
        algorithm: SMC algorithm to use for inference
        address: Address to extract marginal for
        
    Returns:
        Marginal SPDistribution
    """
    return Marginal(
        algorithm=algorithm,
        address=address
    )