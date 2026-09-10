# neuralmie-jax

A JAX implementation of the **NeuralMie** aerosol-optics emulator: two small
pretrained networks that predict the mode-integrated optical properties of a
lognormal aerosol population in a single forward pass, differentiably and under
`jit`/`vmap`.

**Inference only.** Training is out of scope — the published networks were fit
on 10^8 Mie samples. This repository ports the inference path, the NumPy Mie
ground truth needed to validate it, and nothing else.

Derived from [`pnnl/NEURALMIE`](https://github.com/pnnl/NEURALMIE) (BSD-2-Clause,
Copyright 2024 Battelle Memorial Institute). If you use this, cite the paper:

> Geiss, A. and Ma, P.-L.: *NeuralMie (v1.0): An Aerosol Optics Emulator*,
> Geoscientific Model Development, [doi:10.5194/gmd-2024-30](https://doi.org/10.5194/gmd-2024-30)

## Why

The upstream reference implementation runs through TensorFlow/Keras, and its
Mie code is numba-jitted NumPy. Neither is usable inside a differentiable GCM.
This port is `jit`/`grad`/`vmap`-safe with a runtime dependency floor of just
`jax` and `numpy` — **TensorFlow is not required at any point**, because the
weights are read from the Fortran-Keras-Bridge text format that upstream also
ships.

## Install

```bash
uv sync                       # or: pip install -e ".[dev]"
```

## Quickstart

```python
import jax
from neuralmie_jax import sphere_bulk_optics, coreshell_bulk_optics

# Homogeneous spheres: 550 nm, 100 nm number-median radius, sigma_g = 1.8,
# m = 1.53 + 0.005i
optics = sphere_bulk_optics(5.5e-7, 1.0e-7, 1.8, 1.53, 5e-3)
optics.ke_rho     # extinction cross-section per unit particle volume [1/m]
optics.ssa        # single-scattering albedo
optics.g          # scattering-weighted asymmetry parameter
optics.rayleigh   # True where the analytic small-particle limit was used

# Mass extinction coefficient [m^2/kg] for a given material density:
optics.per_mass(1.8e3).ke

# Black-carbon core in a sulfate coating; f is a RADIUS ratio.
coreshell_bulk_optics(5.5e-7, 1.0e-7, 1.8, 1.43, 1e-8,
                      m_r_core=1.85, m_i_core=0.71, core_fraction=0.4)

# Differentiable and batched, with no code changes:
jax.grad(lambda r: sphere_bulk_optics(5.5e-7, r, 1.8, 1.53, 5e-3).ke_rho)(1e-7)
jax.jit(jax.vmap(sphere_bulk_optics))(lam, r_g, sigma_g, m_r, m_i)
```

## Units and conventions

Every one of these is a real source of coupling bugs.

| argument | meaning | units | valid range |
|---|---|---|---|
| `wavelength` | wavelength | m | 2e-7 … 1e-3 |
| `r_g` | number-median **radius** — *not* a diameter | m | 5e-9 … 5e-5 |
| `sigma_g` | geometric standard deviation | – | 1.2 … 2.8 |
| `m_r` | real refractive index (shell, if coated) | – | 1.1 … 3.0 |
| `m_i` | imaginary index, **positive** convention | – | 1e-8 … 1.0 |
| `m_r_core`, `m_i_core` | core refractive index | – | as above |
| `core_fraction` | `r_core / r_total`, a **radius** ratio | – | 0 … 0.98 |

- `r_g` is a **radius**. MAM/E3SM's `dgnum` is a *diameter*; halve it.
- `core_fraction` is a **radius** ratio, so the volume fraction is its cube —
  `f = 0.2` is only 0.8% core by volume. Invert with `f = (V_core/V_tot)**(1/3)`.
- The primary output `ke_rho` is per unit **particle volume** (1/m), not per
  unit mass. `ke_rho = 0.75·∫Qe·pdf·r² dlnr / ∫pdf·r³ dlnr`. Density is not a
  network input; it only divides the output, hence `per_mass(rho)`.
- `g` is **scattering-weighted**, and comes from a sigmoid, so it lies in
  (0, 1) — this emulator cannot represent back-scattering.
- Inputs are clipped to the training box by default (`clip=False` to disable).
  `wavelength` is deliberately *not* clipped: the network sees it only through
  the size parameter, and it re-enters via the `1/λ` output factor.

## The Rayleigh branch

Below the switch — where 99.9% of particles have size parameter ≤ 0.1 — an
analytic Rayleigh expression replaces the network, and `g` is exactly 0. Both
arms are always evaluated and selected with `jnp.where`, so the function stays
traceable.

**The networks are untrained below the switch**, because upstream drops those
rows from the training set (`keep = upper_bounds > 0.1`). Unclamped, the
core-shell network's raw output reaches +2142 there, so `exp()` overflows to
`inf` across ~1.4% of the training box and poisons `where` gradients. This port
therefore clamps the size parameter *up to the switch boundary* before it
reaches the network. The clamp is **bitwise identity on every non-Rayleigh
point** (measured exactly 0.0), so it cannot perturb a returned value — it only
keeps the discarded arm finite and smooth.

In practice the branch is barely reachable for real aerosol: at 550 nm with
σ_g = 1.8 it fires only below r_g ≈ 1.3 nm, which is under the domain's own
5 nm floor. It becomes reachable only in the longwave (r_g < 23 nm at 10 µm).

## Accuracy

Against the NumPy Mie + 1024-point quadrature ground truth in
`neuralmie_jax.reference`, over the range relevant to modal aerosol schemes
(λ 200 nm–30 µm, r_g 10 nm–2 µm, σ_g 1.2–2.0, m_r 1.3–2.6, m_i 1e-8–1):

| quantity | median | p90 | max |
|---|---|---|---|
| `ke_rho` (relative) | 0.023% | 0.083% | 0.18% |
| `ssa` (absolute) | 3.6e-5 | 5.0e-4 | 9.2e-4 |
| `g` (absolute) | 1.2e-4 | 6.5e-4 | 1.4e-3 |

Reproduce with `pytest -m slow src/neuralmie_jax/accuracy_test.py -s`.

**These statistics cap the size parameter at 100** to keep the Mie reference
affordable, so they do *not* characterise the large-`mu_x` corner of the domain
(which reaches ≈1570). Treat the maxima as lower bounds on the true worst case.

Agreement with the 12 published reference cases is **≤1e-6 relative**. That
floor is set by the weight files, not the port: the FKB text format stores
`%.7e` (8 significant digits) while a float32 round-trip needs 9, so the
weights can differ from the Keras originals by ~1 ulp. Golden tolerances are
set at `1e-5` and should not be tightened.

One published case is a known upstream *disagreement*, not a port error: the
final core-shell Rayleigh case gives ke = 60.27 against a Mie truth of 48.21
(~25%), because the core-shell Rayleigh limit volume-mixes at `f = 0.8`. The
upstream demo documents this. Our tests assert the emulator value tightly and
the Mie comparison loosely.

## Weights

`tools/convert_weights.py` derives `src/neuralmie_jax/data/*.npz` from the FKB
text files vendored in `third_party/NEURALMIE/` (checksums and provenance in
`PROVENANCE.md`). CI re-runs the conversion and compares **arrays**, not file
hashes, because `np.savez` embeds zip mtimes.

| network | inputs | architecture | parameters |
|---|---|---|---|
| `sphere` | 4 | 4 → 69×4 (swish) → 3 | 15,045 |
| `coreshell` | 7 | 7 → 112×4 (swish) → 3 | 39,203 |

The FKB format has three traps: line 1 is a count of *header lines* (11), not
layers; line 13 is the final Adam learning rate, not a weight; and each weight
line is **column-major over the Keras `(n_in, n_out)` kernel**, so kernels must
be rebuilt as `vals.reshape(n_out, n_in).T`. Getting the last one wrong yields
plausible but incorrect output rather than an error.

The hidden activation must be `swish` (`x·sigmoid(x)`). Upstream warns that
running these weights under a different activation produces no error and
silently wrong results.

## Reference implementation

`neuralmie_jax.reference` holds a NumPy transcription of TAMie (Mie scattering
for homogeneous and coated spheres) and the lognormal bulk integration that
defines the emulator's target. It reproduces all 12 published Mie values to
3.4e-8. `numba` is optional and only accelerates it.

It is deliberately **not** JAX: the recurrences are sequential, need complex
float64, and use a data-dependent order. Cost scales with the Mie order — one
1024-radius sample is ~0.03 s at small size parameter but ~37 s at the worst
corner, and core-shell is ~4× worse. Cap the size parameter or sample
stratified when generating ground truth.

## Vendoring into another project

`src/neuralmie_jax/neuralmie.py` is self-contained by design: it imports only
the standard library, `numpy` and `jax`, so it can be dropped into another tree
by copying the file plus `data/*.npz`, with no import rewriting. A test
(`VendorabilityTest`) enforces that mechanically so the property cannot rot.

Carry the attribution: keep the module docstring's provenance line, and
reproduce the BSD-2-Clause notice and the DOE disclaimer from `NOTICE`.

## Differences from upstream

Behaviour-preserving unless stated:

1. **Rayleigh switch is a `jnp.where`**, not a Python `if`, so it traces.
2. **The size parameter is clamped to the switch boundary** before the network.
   Prevents `inf` in the untaken arm; provably inert on returned values.
3. **`erfc` differences replace `erf` differences** in the Rayleigh moment
   ratio. The arguments saturate; at σ_g = 2.8 the `erf` form loses ~7e-6
   relative accuracy in float32 versus ~1e-6 for `erfc`.
4. **Real-arithmetic Clausius–Mossotti** instead of complex, avoiding complex
   dtypes and `abs(complex)`'s non-differentiable point. Exact to 5e-16.
5. **Returns `ke_rho` (1/m)** with a `per_mass(rho)` helper, rather than taking
   a density argument that silently changes the output's units.
6. **Domain clipping uses a straight-through estimator**, so an out-of-domain
   input reports the emulator's sensitivity at the boundary instead of a zero
   gradient.
7. **`scipy` is not a dependency**: `erfinv` at the fixed quantiles comes from
   the standard library's inverse normal CDF (agreeing to 6e-16).

## Development

```bash
ruff check .                  # pinned to 0.15.17 to match CI
pytest -q                     # fast suite
pytest -q -m slow             # accuracy sweep against the Mie reference
```

Tests are `*_test.py` co-located with their module and written as
`unittest.TestCase`, matching the conventions of the downstream consumer so
they vendor along with the code.

## License

BSD-2-Clause. See `LICENSE`, `NOTICE` (which carries the required PNNL/DOE
disclaimer), and `third_party/NEURALMIE/LICENSE`.
