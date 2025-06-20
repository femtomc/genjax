"""GenJAX Gaussian Process Module.

This module implements Gaussian processes as generative functions,
allowing them to be composed with other probabilistic programs
while using exact GP inference as the internal proposal.
"""

from .kernels import (
    Kernel,
    RBF,
    Matern12,
    Matern32,
    Matern52,
    Linear,
    Polynomial,
    White,
    Constant,
    Sum,
    Product,
)
from .mean import (
    MeanFunction,
    Zero,
    Constant as ConstantMean,
    Linear as LinearMean,
)
from .gp import GP

__all__ = [
    # Kernels
    "Kernel",
    "RBF",
    "Matern12",
    "Matern32",
    "Matern52",
    "Linear",
    "Polynomial",
    "White",
    "Constant",
    "Sum",
    "Product",
    # Mean functions
    "MeanFunction",
    "Zero",
    "ConstantMean",
    "LinearMean",
    # GP
    "GP",
]
