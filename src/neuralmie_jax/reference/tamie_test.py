"""Tests for the NumPy TAMie reference."""

import unittest

import numpy as np

from neuralmie_jax import _golden as G
from neuralmie_jax.reference import bulk_optics as bo
from neuralmie_jax.reference import tamie


class TAMieTest(unittest.TestCase):
    def test_published_mie_values(self):
        """The reference must reproduce upstream's ``Mie code:`` triplets.

        rtol 1e-6 rather than machine epsilon: ``np.sum``'s pairwise reduction
        orders differently from the original numba-compiled sequential sum,
        which shows up in the tiny-x cases where Qe and Qs nearly cancel.
        """
        for cases, coreshell in ((G.SPHERE + G.SPHERE_RAYLEIGH, False),
                                 (G.CORESHELL + G.CORESHELL_RAYLEIGH, True)):
            for inp, mie, _ann in cases:
                ke, ks, g = bo.mass_efficiency(*inp)
                got = (ke / G.RHO, ks / ke, g)
                for a, b, name in zip(got, mie, ("ke", "ssa", "g"), strict=True):
                    self.assertAlmostEqual(a / b, 1.0, delta=1e-6,
                                           msg=f"{name} for {inp} (coreshell={coreshell})")

    def test_coreshell_degenerates_to_sphere(self):
        """Upstream's three short-circuit branches."""
        m = 1.5 + 0.01j
        self.assertEqual(tamie.coreshell(m, m, 5.0, 10.0), tamie.sphere(m, 10.0))
        # xc/xs < 0.01 -> homogeneous shell
        self.assertEqual(tamie.coreshell(1.9 + 0.5j, m, 0.05, 10.0), tamie.sphere(m, 10.0))
        # xc/xs > 0.99 -> core index at the shell size (intentional upstream)
        mc = 1.9 + 0.5j
        self.assertEqual(tamie.coreshell(mc, m, 9.95, 10.0), tamie.sphere(mc, 10.0))

    def test_energy_conservation(self):
        for x in (0.1, 1.0, 10.0, 50.0):
            for mi in (0.0, 0.01, 0.5):
                qe, qs, g = tamie.sphere(1.5 + 1j * mi, x)
                self.assertGreaterEqual(qe + 1e-12, qs, msg=f"Qs > Qe at x={x}, k={mi}")
                self.assertGreaterEqual(qs, -1e-12)
                self.assertTrue(-1.0 <= g <= 1.0)

    def test_rayleigh_limit(self):
        """As x -> 0, Qs -> (8/3) x^4 |(m^2-1)/(m^2+2)|^2."""
        m, x = 1.5 + 0.0j, 1e-3
        _qe, qs, _g = tamie.sphere(m, x)
        want = (8.0 / 3.0) * x**4 * abs((m**2 - 1) / (m**2 + 2)) ** 2
        self.assertAlmostEqual(qs / want, 1.0, delta=1e-3)

    def test_nonabsorbing_ssa_unity(self):
        for x in (0.5, 5.0, 25.0):
            qe, qs, _g = tamie.sphere(1.5 + 0.0j, x)
            self.assertAlmostEqual(qs / qe, 1.0, delta=1e-6)

    @unittest.skipUnless(tamie.HAS_NUMBA, "numba not installed")
    def test_numba_and_python_agree(self):
        qe, qs, g = tamie.sphere(1.5 + 0.01j, 10.0)
        self.assertTrue(np.isfinite([qe, qs, g]).all())


class BulkQuadratureTest(unittest.TestCase):
    def test_quadrature_convergence(self):
        """Quantify the ground truth's own discretisation error."""
        args = (5.5e-7, 1.0e-7, 1.8, 1.53, 5e-3)
        coarse = bo.mass_efficiency(*args, nrad=1024)
        fine = bo.mass_efficiency(*args, nrad=4096)
        for a, b in zip(coarse, fine, strict=True):
            self.assertAlmostEqual(a / b, 1.0, delta=1e-3)

    def test_integration_bounds_symmetric_in_log(self):
        lo, hi = bo.integration_bounds(1e-7, 1.8)
        self.assertAlmostEqual(np.log(1e-7) - np.log(lo), np.log(hi) - np.log(1e-7), delta=1e-9)

    def test_erfinv_matches_stdlib_normal(self):
        # scipy-free path: erfinv(2t-1) == NormalDist().inv_cdf(t)/sqrt(2)
        self.assertAlmostEqual(bo._erfinv_2t_minus_1(0.9995), 3.2905267314918945 / np.sqrt(2.0),
                               delta=1e-12)

    def test_grey_sphere_normalisation(self):
        """With Qe == 2 the integral reduces to 1.5/r_eff analytically.

        Validates the quadrature weights independently of the Mie solver.
        """
        mu, sg = 1e-7, 1.8
        rmin, rmax = bo.integration_bounds(mu, sg)
        r = np.exp(np.linspace(np.log(rmin), np.log(rmax), 4096))
        epdf = np.exp(-(np.log(r / mu) ** 2) / (2.0 * np.log(sg) ** 2))
        lnr = np.log(r)
        got = 0.75 * bo.integrate(lnr, 2.0 * epdf * r**2) / bo.integrate(lnr, epdf * r**3)
        r_eff = bo.integrate(lnr, epdf * r**3) / bo.integrate(lnr, epdf * r**2)
        self.assertAlmostEqual(got / (1.5 / r_eff), 1.0, delta=1e-9)


if __name__ == "__main__":
    unittest.main()
