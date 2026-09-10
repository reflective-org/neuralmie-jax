"""JAX implementation of the NeuralMie aerosol optics emulator.

Derived from pnnl/NEURALMIE (BSD-2-Clause, Copyright 2024 Battelle Memorial
Institute). See NOTICE for attribution and the required disclaimer.
"""

from neuralmie_jax.neuralmie import (
    BulkOptics,
    MassOptics,
    MLPWeights,
    NeuralMieWeights,
    coreshell_bulk_optics,
    default_weights,
    is_rayleigh,
    rayleigh_bulk_optics,
    size_parameter,
    sphere_bulk_optics,
)

__version__ = "0.1.0"

__all__ = [
    "BulkOptics",
    "MLPWeights",
    "MassOptics",
    "NeuralMieWeights",
    "coreshell_bulk_optics",
    "default_weights",
    "is_rayleigh",
    "rayleigh_bulk_optics",
    "size_parameter",
    "sphere_bulk_optics",
]
