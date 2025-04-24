from beartype import BeartypeConf
from beartype.claw import beartype_this_package

conf = BeartypeConf(
    is_color=True,
    is_debug=False,
    is_pep484_tower=True,
    violation_type=TypeError,
)

beartype_this_package(conf=conf)

from .adev import (
    Dual,
    add_cost,
    categorical_enum_parallel,
    expectation,
    flip_enum,
    flip_enum_parallel,
    flip_mvd,
    flip_reinforce,
    geometric_reinforce,
    normal_reinforce,
    normal_reparam,
)
from .core import (
    GFI,
    Distribution,
    Fn,
    Pytree,
    Trace,
    Vmap,
    gen,
    get_choices,
    modular_vmap,
    seed,
    tfp_distribution,
    trace,
)
from .distributions import (
    bernoulli,
    beta,
    categorical,
    flip,
    normal,
)

__all__ = [
    "GFI",
    "Distribution",
    "Dual",
    "Fn",
    "Pytree",
    "Trace",
    "Vmap",
    "add_cost",
    "bernoulli",
    "beta",
    "categorical",
    "categorical_enum_parallel",
    "expectation",
    "flip",
    "flip_enum",
    "flip_enum_parallel",
    "flip_mvd",
    "flip_reinforce",
    "gen",
    "geometric_reinforce",
    "get_choices",
    "modular_vmap",
    "normal",
    "normal_reinforce",
    "normal_reparam",
    "seed",
    "tfp_distribution",
    "trace",
]
