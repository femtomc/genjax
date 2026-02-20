"""Compatibility helpers for third-party API transitions.

These shims are kept minimal and can be removed once upstream dependencies
(namely TensorFlow Probability) support newer JAX APIs directly.
"""

from __future__ import annotations

import jax


def ensure_jax_tfp_compat() -> None:
    """Install small compatibility shims needed by current TFP on JAX >= 0.7.

    TFP 0.25 still references ``jax.interpreters.xla.pytype_aval_mappings``,
    which was removed in JAX 0.7 in favor of ``jax.core.pytype_aval_mappings``.
    """

    xla_interpreter = jax.interpreters.xla
    if not hasattr(xla_interpreter, "pytype_aval_mappings"):
        xla_interpreter.pytype_aval_mappings = jax.core.pytype_aval_mappings
