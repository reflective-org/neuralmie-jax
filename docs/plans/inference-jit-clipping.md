# Fix inference issues #1 and #2

Scope: fix first-use JIT weight caching and consistent Rayleigh input clipping
in neuralmie-jax. Keep the public API and in-domain numerical behavior.

- [x] Add regressions for cold-cache JIT and clipped Rayleigh/network branches.
  Verify: the new tests fail on the original implementation.
- [x] Keep cached weights concrete and clip inputs before constructing Rayleigh
  indices, preserving straight-through sensitivities.
  Verify: regressions pass for both sphere and core-shell inference.
- [x] Review the diff, update the README, and run lint, the full test suite
  (including published golden cases and accuracy tests), and weight conversion.
  Verify: no reference tolerances or weight arrays change.
- [x] Commit and open a PR linked to #1 and #2; inspect CI status.
  Delivery: [PR #3](https://github.com/reflective-org/neuralmie-jax/pull/3).
  Its Checks tab tracks the lint, Python-version matrix, and weight round trip.

The clipping regressions also reproduced cancellation in the straight-through
expression: a negative imaginary index could clip to zero instead of 1e-8.
The expression now adds an exactly-zero primal term to the clipped value,
retaining the identity derivative without subtracting from the small bound.

Local verification on both JAX 0.11.1 and the minimum JAX 0.4.35 / NumPy 1.26.4
on CPU: 44 tests passed, 1 optional-numba test skipped, and 32 subtests passed
in each environment. Ruff 0.15.17 and the weight-array round trip passed.
Published golden tolerances and all weight arrays are unchanged.
