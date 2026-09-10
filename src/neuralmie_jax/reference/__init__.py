"""NumPy ground-truth reference: TAMie Mie scattering plus bulk integration.

Not on the JAX path and **not** part of the vendorable inference module. This
exists so the emulator's accuracy is a measured number rather than a claim.
"""

from neuralmie_jax.reference.bulk_optics import bulk_optics_reference, mass_efficiency
from neuralmie_jax.reference.tamie import HAS_NUMBA, coreshell, qqg, sphere

__all__ = ["HAS_NUMBA", "bulk_optics_reference", "coreshell", "mass_efficiency", "qqg", "sphere"]
