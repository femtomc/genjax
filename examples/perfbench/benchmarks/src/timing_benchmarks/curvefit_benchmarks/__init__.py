"""Benchmark implementations for different frameworks.

Submodules are imported lazily to avoid `runpy` warnings when individual modules
are executed via ``python -m timing_benchmarks.curvefit_benchmarks.<mod>``.
"""

_EXPORTS = {
    "genjax_polynomial_is_timing": "timing_benchmarks.curvefit_benchmarks.genjax",
    "genjax_polynomial_hmc_timing": "timing_benchmarks.curvefit_benchmarks.genjax",
    "numpyro_polynomial_is_timing": "timing_benchmarks.curvefit_benchmarks.numpyro",
    "handcoded_jax_polynomial_is_timing": "timing_benchmarks.curvefit_benchmarks.handcoded_jax",
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name):
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(module_name)
    try:
        value = getattr(module, name)
    except AttributeError as exc:
        raise AttributeError(f"module {module_name!r} does not define {name!r}") from exc
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)
