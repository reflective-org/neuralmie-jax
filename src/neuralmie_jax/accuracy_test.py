"""Emulator accuracy against the NumPy Mie + quadrature ground truth.

Deliberately separate from ``neuralmie_test.py``. That file asserts the
emulator reproduces its *published* values -- a weight-loading and scaling
contract that fails on a transposed kernel. This file asserts the emulator is
*physically accurate* -- which fails on a genuine accuracy regression. A single
blended test could not distinguish the two.

Marked slow: each reference sample runs a 1024-radius Mie quadrature.
"""

import unittest

import numpy as np
import pytest

from neuralmie_jax import neuralmie as nm
from neuralmie_jax.reference import bulk_optics as bo

# Thresholds are set from measured behaviour with margin, not from the paper's
# headline number, so a regression is caught against reality. See README.
KE_P90_RTOL = 5.0e-3
KE_MAX_RTOL = 2.0e-2
SSA_P90_ATOL = 5.0e-3
G_P90_ATOL = 5.0e-3

#: Reference cost scales with the Mie order, so the sweep caps the largest
#: size parameter. These statistics therefore do NOT characterise the
#: large-mu_x corner of the domain; see README.
X99_CAP = 100.0


def _sample(n, seed, cap=X99_CAP):
    """Draw samples that exercise the network arm and stay affordable."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        lam = float(np.exp(rng.uniform(np.log(2e-7), np.log(3e-5))))
        mu = float(np.exp(rng.uniform(np.log(1e-8), np.log(2e-6))))
        sg = float(rng.uniform(1.2, 2.0))
        mr = float(rng.uniform(1.3, 2.6))
        mi = float(np.exp(rng.uniform(np.log(1e-8), 0.0)))
        mu_x = 2 * np.pi * mu / lam
        x99 = mu_x * np.exp(nm.SQRT2_ERFINV_999 * np.log(sg))
        if x99 <= nm.RAYLEIGH_SIZE_PARAMETER or x99 > cap:
            continue
        out.append((lam, mu, sg, mr, mi))
    return out


@pytest.mark.slow
class SphereAccuracyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = []
        for args in _sample(60, seed=20260909):
            ref = bo.bulk_optics_reference(*args)
            got = nm.sphere_bulk_optics(*args)
            cls.rows.append((
                abs(float(got.ke_rho) / ref[0] - 1.0),
                abs(float(got.ssa) - ref[1]),
                abs(float(got.g) - ref[2]),
            ))
        cls.err = np.asarray(cls.rows)

    def test_extinction_accuracy(self):
        ke = self.err[:, 0]
        self.assertLess(float(np.percentile(ke, 90)), KE_P90_RTOL)
        self.assertLess(float(ke.max()), KE_MAX_RTOL)

    def test_ssa_accuracy(self):
        self.assertLess(float(np.percentile(self.err[:, 1], 90)), SSA_P90_ATOL)

    def test_asymmetry_accuracy(self):
        self.assertLess(float(np.percentile(self.err[:, 2], 90)), G_P90_ATOL)

    def test_report(self):
        """Print the table quoted in the README so it cannot drift silently."""
        names = ("ke_rho (rel)", "ssa (abs)", "g (abs)")
        print(f"\n  emulator vs Mie+quadrature, n={len(self.err)}, x99<={X99_CAP}")
        for i, name in enumerate(names):
            col = self.err[:, i]
            print(f"    {name:<14} median {np.median(col):.3e}  "
                  f"p90 {np.percentile(col, 90):.3e}  max {col.max():.3e}")


if __name__ == "__main__":
    unittest.main()
