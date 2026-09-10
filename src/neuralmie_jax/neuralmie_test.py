"""Tests for the NeuralMie JAX emulator."""

import itertools
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from neuralmie_jax import _golden as G
from neuralmie_jax import neuralmie as nm

# The FKB text weights carry %.7e = 8 significant digits, while a float32
# round-trip needs 9. So the committed weights may differ from the Keras
# originals by ~1 ulp and golden agreement floors out near 1e-6. Do not
# tighten this; a tighter bound would be asserting file-format noise.
GOLDEN_RTOL = 1.0e-5


def _sample_domain(n, seed=0, coreshell=False):
    """Draw ``n`` points from the training box, log-uniform where upstream is."""
    rng = np.random.default_rng(seed)
    lo, hi = nm.WAVELENGTH_RANGE
    out = [np.exp(rng.uniform(np.log(lo), np.log(hi), n))]
    lo, hi = nm.MU_RANGE
    out.append(np.exp(rng.uniform(np.log(lo), np.log(hi), n)))
    out.append(rng.uniform(*nm.SIGMA_G_RANGE, n))
    out.append(rng.uniform(*nm.M_R_RANGE, n))
    lo, hi = nm.M_I_RANGE
    out.append(np.exp(rng.uniform(np.log(lo), np.log(hi), n)))
    if coreshell:
        out.append(rng.uniform(*nm.M_R_RANGE, n))
        out.append(np.exp(rng.uniform(np.log(lo), np.log(hi), n)))
        out.append(rng.uniform(*nm.CORE_FRACTION_RANGE, n))
    return [jnp.asarray(a) for a in out]


class WeightsTest(unittest.TestCase):
    def test_architecture(self):
        w = nm.default_weights()
        self.assertEqual([x.kernel.shape for x in w.sphere.layers],
                         [(4, 69), (69, 69), (69, 69), (69, 69), (69, 3)])
        self.assertEqual([x.kernel.shape for x in w.coreshell.layers],
                         [(7, 112), (112, 112), (112, 112), (112, 112), (112, 3)])

    def test_parameter_counts(self):
        # A single scalar fingerprint of the whole architecture.
        for net, want in ((nm.default_weights().sphere, 15_045),
                          (nm.default_weights().coreshell, 39_203)):
            n = sum(x.kernel.size + x.bias.size for x in net.layers)
            self.assertEqual(n, want)

    def test_all_finite(self):
        for net in nm.default_weights():
            for layer in net.layers:
                self.assertTrue(bool(jnp.all(jnp.isfinite(layer.kernel))))
                self.assertTrue(bool(jnp.all(jnp.isfinite(layer.bias))))

    def test_memoised(self):
        self.assertIs(nm.default_weights(), nm.default_weights())

    def test_rejects_wrong_architecture(self):
        import importlib.resources
        data = importlib.resources.files("neuralmie_jax") / "data"
        with importlib.resources.as_file(data) as d:
            with self.assertRaises(ValueError):
                nm.load_mlp_weights(d / "sphere.npz", expect_layers=(4, 5, 3))


class ActivationTest(unittest.TestCase):
    def test_swish_matches_definition(self):
        # A non-swish activation produces no error but silently wrong output,
        # which upstream's README explicitly warns about.
        x = jnp.linspace(-50.0, 50.0, 1001)
        want = x / (1.0 + jnp.exp(-x))
        self.assertTrue(bool(jnp.allclose(nm.ACTIVATIONS["swish"](x), want, atol=1e-6)))

    def test_head_is_linear(self):
        self.assertTrue(bool(jnp.all(nm.ACTIVATIONS["linear"](jnp.arange(5.0)) == jnp.arange(5.0))))


class GoldenTest(unittest.TestCase):
    """The published reference values are the contract with upstream."""

    def _check(self, cases, fn, expect_rayleigh):
        for inp, _mie, ann in cases:
            out = fn(*inp)
            self.assertEqual(bool(out.rayleigh), expect_rayleigh, msg=f"branch for {inp}")
            self.assertAlmostEqual(float(out.ke_rho) / G.RHO, ann[0],
                                   delta=GOLDEN_RTOL * abs(ann[0]))
            self.assertAlmostEqual(float(out.ssa), ann[1], delta=GOLDEN_RTOL * abs(ann[1]))
            if ann[2] == 0:
                self.assertEqual(float(out.g), 0.0)
            else:
                self.assertAlmostEqual(float(out.g), ann[2], delta=GOLDEN_RTOL * abs(ann[2]))

    def test_sphere_network(self):
        self._check(G.SPHERE, nm.sphere_bulk_optics, expect_rayleigh=False)

    def test_sphere_rayleigh(self):
        self._check(G.SPHERE_RAYLEIGH, nm.sphere_bulk_optics, expect_rayleigh=True)

    def test_coreshell_network(self):
        self._check(G.CORESHELL, nm.coreshell_bulk_optics, expect_rayleigh=False)

    def test_coreshell_rayleigh(self):
        self._check(G.CORESHELL_RAYLEIGH, nm.coreshell_bulk_optics, expect_rayleigh=True)

    def test_agreement_with_mie_reference(self):
        """Loose check against the published Mie+quadrature truth.

        The emulator's claim is ~1%. The one deliberate exception is the last
        core-shell Rayleigh case: upstream's own demo shows 48.21 vs 60.27, a
        ~25% miss, because the core-shell Rayleigh limit volume-mixes at
        f=0.8. Asserting that tightly would be asserting a known defect.
        """
        for cases, fn, rtol in ((G.SPHERE, nm.sphere_bulk_optics, 2e-2),
                                (G.SPHERE_RAYLEIGH, nm.sphere_bulk_optics, 2e-2),
                                (G.CORESHELL, nm.coreshell_bulk_optics, 2e-2),
                                (G.CORESHELL_RAYLEIGH, nm.coreshell_bulk_optics, 0.3)):
            for inp, mie, _ann in cases:
                ke = float(fn(*inp).ke_rho) / G.RHO
                self.assertAlmostEqual(ke, mie[0], delta=rtol * abs(mie[0]),
                                       msg=f"ke for {inp}")


class DomainSafetyTest(unittest.TestCase):
    """Regression tests for the untrained-region overflow.

    Both networks are untrained below the Rayleigh switch, because upstream
    drops those rows (``keep = upper_bounds > 0.1``). Unclamped, the
    core-shell network's raw output reaches +2142 there, so ``exp`` overflows
    to inf across ~1.4% of the training box and poisons ``where`` gradients.
    """

    def test_outputs_finite_over_domain(self):
        for coreshell, fn in ((False, nm.sphere_bulk_optics), (True, nm.coreshell_bulk_optics)):
            args = _sample_domain(20_000, seed=11, coreshell=coreshell)
            out = fn(*args)
            self.assertTrue(bool(jnp.all(jnp.isfinite(out.ke_rho))))
            self.assertTrue(bool(jnp.all((out.ssa >= 0.0) & (out.ssa <= 1.0))))
            self.assertTrue(bool(jnp.all((out.g >= 0.0) & (out.g <= 1.0))))
            # The sample must actually exercise the Rayleigh arm.
            self.assertGreater(float(jnp.mean(out.rayleigh)), 0.05)

    def test_gradients_finite_over_domain(self):
        for coreshell, fn in ((False, nm.sphere_bulk_optics), (True, nm.coreshell_bulk_optics)):
            args = _sample_domain(4_000, seed=12, coreshell=coreshell)
            for i in range(len(args)):
                def loss(v, i=i, args=args, fn=fn):
                    a = list(args)
                    a[i] = v
                    return jnp.sum(fn(*a).ke_rho)

                grad = jax.grad(loss)(args[i])
                self.assertTrue(bool(jnp.all(jnp.isfinite(grad))),
                                msg=f"non-finite grad wrt arg {i} (coreshell={coreshell})")

    def test_clamp_is_identity_outside_rayleigh(self):
        """The mu_x clamp must not perturb any value the emulator returns."""
        args = _sample_domain(20_000, seed=13)
        mu_x = nm.size_parameter(args[0], args[1])
        mask = nm.is_rayleigh(mu_x, args[2])
        clamped = jnp.maximum(mu_x, nm._rayleigh_boundary_mu_x(args[2]))
        # Wherever the network's answer is actually used, the clamp is inert.
        self.assertEqual(float(jnp.max(jnp.abs((clamped - mu_x)[~mask]))), 0.0)


class BranchTest(unittest.TestCase):
    def test_switch_boundary(self):
        """Straddle the switch and confirm the arm flips.

        A longwave wavelength is required, not a visible one: at 550 nm the
        boundary sits at r_g = 1.27 nm, below the training domain's 5 nm
        floor, so domain clipping lifts every physical size above the switch.
        The Rayleigh arm is only reachable inside the valid mu range in the
        longwave -- which is also why no realistic aerosol mode ever takes it
        in the shortwave.
        """
        sigma_g, lam = 1.8, 1.0e-5
        r_boundary = nm._rayleigh_boundary_mu_x(sigma_g) * lam / (2.0 * jnp.pi)
        self.assertGreater(float(r_boundary), nm.MU_RANGE[0])
        below = nm.sphere_bulk_optics(lam, float(r_boundary) * 0.99, sigma_g, 1.5, 1e-3)
        above = nm.sphere_bulk_optics(lam, float(r_boundary) * 1.01, sigma_g, 1.5, 1e-3)
        self.assertTrue(bool(below.rayleigh))
        self.assertFalse(bool(above.rayleigh))
        self.assertEqual(float(below.g), 0.0)

    def test_rayleigh_g_exactly_zero(self):
        out = nm.sphere_bulk_optics(1.085e-04, 6.545e-08, 1.356, 2.467, 1.293e-03)
        self.assertTrue(bool(out.rayleigh))
        self.assertEqual(float(out.g), 0.0)


class CoreShellConsistencyTest(unittest.TestCase):
    def test_reduces_to_sphere_when_indices_match(self):
        """With core index == shell index, f is physically irrelevant."""
        lam, r_g, sg, mr, mi = 5.5e-7, 1.0e-7, 1.8, 1.53, 5e-3
        sphere = nm.sphere_bulk_optics(lam, r_g, sg, mr, mi)
        for f in (0.0, 0.2, 0.5, 0.8, 0.98):
            cs = nm.coreshell_bulk_optics(lam, r_g, sg, mr, mi, mr, mi, f)
            self.assertAlmostEqual(float(cs.ke_rho) / float(sphere.ke_rho), 1.0, delta=5e-3)
            self.assertAlmostEqual(float(cs.ssa), float(sphere.ssa), delta=5e-3)
            self.assertAlmostEqual(float(cs.g), float(sphere.g), delta=5e-3)

    def test_absorbing_core_reduces_ssa(self):
        """A black-carbon core in a transparent shell must darken the mode."""
        base = dict(wavelength=5.5e-7, r_g=1.0e-7, sigma_g=1.8, m_r=1.43, m_i=1e-8)
        ssa = [float(nm.coreshell_bulk_optics(**base, m_r_core=1.85, m_i_core=0.71,
                                              core_fraction=f).ssa)
               for f in (0.0, 0.2, 0.4, 0.6, 0.8)]
        self.assertTrue(all(a > b for a, b in itertools.pairwise(ssa)),
                        msg=f"ssa not monotone: {ssa}")


class TransformTest(unittest.TestCase):
    def test_jit_and_vmap_match_eager(self):
        args = _sample_domain(64, seed=3)
        eager = nm.sphere_bulk_optics(*args).ke_rho
        # rtol 1e-5, not 1e-6: XLA fusion reassociates the float32 matmuls, and
        # float32 eps is 1.2e-7, so the observed jit/eager spread is ~1e-6.
        self.assertTrue(bool(jnp.allclose(jax.jit(nm.sphere_bulk_optics)(*args).ke_rho,
                                          eager, rtol=1e-5)))
        self.assertTrue(bool(jnp.allclose(jax.vmap(nm.sphere_bulk_optics)(*args).ke_rho,
                                          eager, rtol=1e-5)))

    def test_broadcasting(self):
        out = nm.sphere_bulk_optics(jnp.full((6, 1), 5.5e-7), jnp.full((1, 5), 1e-7),
                                    1.8, 1.5, 1e-3)
        self.assertEqual(out.ke_rho.shape, (6, 5))

    def test_scalar_matches_batch(self):
        args = _sample_domain(8, seed=4)
        batch = nm.sphere_bulk_optics(*args).ke_rho
        scalar = nm.sphere_bulk_optics(*[float(a[0]) for a in args]).ke_rho
        self.assertEqual(scalar.shape, ())
        self.assertAlmostEqual(float(scalar) / float(batch[0]), 1.0, delta=1e-6)


class ScalingTest(unittest.TestCase):
    def test_dimensional_invariance(self):
        """Scaling lambda and r_g together must leave (ke_rho*lambda, ssa, g) fixed.

        Exact by construction: the network sees only the size parameter, and
        lambda re-enters solely through the 1/lambda output factor.
        """
        a = nm.sphere_bulk_optics(5.5e-7, 1.0e-7, 1.8, 1.5, 1e-3)
        b = nm.sphere_bulk_optics(5.5e-6, 1.0e-6, 1.8, 1.5, 1e-3)
        self.assertAlmostEqual(float(a.ke_rho) / float(b.ke_rho), 10.0, delta=1e-4)
        self.assertAlmostEqual(float(a.ssa), float(b.ssa), delta=1e-6)
        self.assertAlmostEqual(float(a.g), float(b.g), delta=1e-6)

    def test_per_mass_conversion(self):
        out = nm.sphere_bulk_optics(5.5e-7, 1.0e-7, 1.8, 1.5, 1e-3)
        mass = out.per_mass(1.8e3)
        want = float(out.ke_rho) / 1.8e3
        self.assertAlmostEqual(float(mass.ke) / want, 1.0, delta=1e-6)
        self.assertEqual(float(mass.ssa), float(out.ssa))

    def test_clipping_bounds_out_of_domain_inputs(self):
        at_edge = nm.sphere_bulk_optics(5.5e-7, nm.MU_RANGE[1], 1.8, 1.5, 1e-3)
        beyond = nm.sphere_bulk_optics(5.5e-7, nm.MU_RANGE[1] * 10.0, 1.8, 1.5, 1e-3)
        self.assertAlmostEqual(float(beyond.ssa), float(at_edge.ssa), delta=1e-6)
        unclipped = nm.sphere_bulk_optics(5.5e-7, nm.MU_RANGE[1] * 10.0, 1.8, 1.5, 1e-3,
                                          clip=False)
        self.assertNotAlmostEqual(float(unclipped.ssa), float(at_edge.ssa), delta=1e-9)

    def test_clip_gradient_passes_through(self):
        """Straight-through estimator: clipped inputs keep a finite sensitivity."""
        grad = jax.grad(lambda r: nm.sphere_bulk_optics(5.5e-7, r, 1.8, 1.5, 1e-3).ke_rho)(
            nm.MU_RANGE[1] * 10.0)
        self.assertTrue(bool(jnp.isfinite(grad)))
        self.assertNotEqual(float(grad), 0.0)


class VendorabilityTest(unittest.TestCase):
    def test_imports_are_self_contained(self):
        """The module must vendor into another tree by copying the file alone.

        Enforced mechanically so the property cannot rot: only the standard
        library, numpy and jax may be imported at module level.
        """
        import ast
        import pathlib

        src = pathlib.Path(nm.__file__).read_text()
        roots = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                roots.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
        allowed = {"__future__", "collections", "functools", "importlib", "math",
                   "typing", "numpy", "jax"}
        self.assertEqual(roots - allowed, set(), msg=f"unexpected imports: {roots - allowed}")


if __name__ == "__main__":
    unittest.main()
