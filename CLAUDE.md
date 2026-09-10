# CLAUDE.md

Operating rules for the NeuralMie JAX port. Defaults, not suggestions —
deviate only when I say so.

`plan → branch → implement in a verify loop → self-review → commit → push → PR`

## 0. Which repo am I in?

This project spans two repos with **different, conflicting** conventions.
Check before you act.

**`reflective-org/neuralmie-jax`** — our repo. These rules govern.

**`climate-analytics-lab/jax-gcm`** (fork of; PRs target `dev`) — *their*
`CLAUDE.md` at the repo root governs and **overrides this file** on every
conflict. Read it before the first edit. It is not a style guide, it is the
contributing guide. Known overrides:
 - Branch off `dev`, never `main`. Never commit directly to either.
 - Roll related findings into the same PR; only genuinely unrelated ones
   become issues. An issue must exist *before* the PR.
 - `ruff==0.15.17` is the only linter. No formatter, no type checker.
 - Gate before every push:
   `ruff check .` then
   `JAX_PLATFORMS=cpu pytest -n 12 -m "not slow" --cov=jcm --cov-fail-under=90`
   then `JAX_PLATFORMS=cpu pytest -n 4 -m "slow" --cov=jcm
   --cov-config=.coveragerc-pr --cov-fail-under=80`.
   Coverage must be measured at **CI dependency parity** (no extras installed).
 - Docs go in `docs/source/design/*.md` + the `docs/source/design.rst`
   toctree. Never an ad-hoc top-level `*.md`.
 - The Codex bot review posts as *inline review comments*, invisible to
   `gh pr view --json comments`. Read it with
   `gh api repos/climate-analytics-lab/jax-gcm/pulls/<N>/reviews` and
   `.../comments`. Every bot comment gets a threaded reply — "Confirmed and
   fixed in <sha>" or "Refuted: <evidence>" — before handing back.
 - Fix the *class*, not the instance: before closing a finding, enumerate the
   category and grep for the same mistake elsewhere.

## 1. Numerical fidelity is the acceptance criterion

This is a port. "It runs" is not done; "it reproduces the reference" is done.

 - Every ported routine is validated against the upstream implementation to a
   **stated numerical tolerance**, and that tolerance is asserted in a test.
 - The 12 published reference cases from the NEURALMIE demo scripts are
   committed as golden tests. They are the contract.
 - Separate the two failure modes with **separate tests**: (a) weight loading
   / network evaluation — assert the JAX net matches a NumPy re-implementation
   to ~1e-6; (b) emulator accuracy — assert agreement with the Mie+quadrature
   reference to the paper's ~1% on `k_e`. A single blended test cannot tell a
   transposed weight matrix from a genuine accuracy regression.
 - When you cannot match the reference, say so with numbers. Never widen a
   tolerance to make a test pass without saying that is what you did and why.

## 2. Provenance and attribution

Upstream `pnnl/NEURALMIE` is **BSD-2-Clause, Battelle Memorial Institute
2024**, with a US-DOE disclaimer.

 - Retain the copyright notice, the licence conditions, and the DOE
   disclaimer in any file or distribution derived from it.
 - Every transcribed formula or constant carries a comment citing the
   upstream source as `file:line`, plus the paper
   (Geiss & Ma, GMD 2024, doi:10.5194/gmd-2024-30) where it is the authority.
 - Never "clean up" a transcribed constant into a rounder number.

## 3. Think before coding

Don't assume. Don't hide confusion. Surface tradeoffs.

 - State assumptions explicitly. If uncertain, ask.
 - Multiple interpretations → present them, don't pick silently.
 - Simpler approach available → say so. Push back when warranted.
 - Unclear → stop, name what's confusing, ask.

Record the decision **in a comment at the point of the decision**, explaining
*why*. Comments describe the current code, never how it differs from an
earlier version.

## 4. Plan before you touch code

No edits until a plan exists and I've seen it.

 - Restate the goal in one sentence, then list steps.
 - Each step gets a check: `[step] → verify: [check]`.
 - Track it as a task list; keep it current.
 - Multi-step plans go in `docs/plans/<short-name>.md` (this repo only) so
   the work survives a context reset.
 - Plan changed mid-flight → say what changed and why before proceeding.

## 5. JAX discipline

 - Functions are **pure**. No side effects, no hidden state.
 - **No Python `if`/`while` on traced values.** Use `jnp.where`,
   `lax.select`, `lax.cond`, `lax.scan`.
 - A `jnp.where` does **not** protect gradients: the untaken branch is still
   differentiated, so a NaN or inf there poisons the result. Clamp the input
   *before* the unsafe op, then select — the double-where pattern. Any
   function with a guarded branch gets a `jax.grad` finiteness test.
 - Shapes statically known. Broadcasting-native: no rank branching, no
   `reshape` to force a layout.
 - Be explicit about precision. These nets are float32-trained; the Mie
   recurrences need float64. Say which is which and why.
 - Every public function gets `jit`, `grad`, and `vmap` exercised in tests.

## 6. Simplicity first

Minimum code that solves the problem. Nothing speculative.

 - No features beyond what was asked. No abstractions for single-use code.
 - No configurability that wasn't requested. No error handling for
   impossible scenarios.
 - 200 lines that could be 50 → rewrite it.

Ask: "would a senior engineer call this overcomplicated?" If yes, simplify.

**Exception:** faithfulness beats brevity. Do not "simplify" a transcribed
formula into something algebraically equivalent-looking — it defeats
line-by-line review against the reference.

## 7. Surgical changes

Touch only what you must. Clean up only your own mess.

 - Don't improve adjacent code, comments, or formatting. Don't refactor what
   isn't broken. Match existing style even if you'd do it differently.
 - jcm sets no enforced line length and long lines are normal there — never
   reflow existing lines.
 - Unrelated dead code → mention it, don't delete it.
 - Remove imports and helpers that *your* change orphaned.

The test: every changed line traces to the request.

## 8. Goal-driven execution

Define success criteria, then loop until verified:
**change → run the check → read the failure → fix → repeat.**

Don't hand back failing code asking what to do. Iterate until the criteria
are met or you hit a genuine decision that is mine to make. Effort or tedium
is not such a decision.

Deliver the **complete** solution — no band-aid presented as done. If the
right fix is deeper than expected, do the deeper fix. A cap or guard is
acceptable only as an explicitly-labelled stopgap I have agreed to.

## 9. Branches and PRs

 - `<type>/<short-description>`: `feat/`, `fix/`, `chore/`, `refactor/`,
   `docs/`. Branch from an up-to-date trunk (`main` here, `dev` in jcm).
 - Worktrees for parallel or long-running work:
   `git worktree add ../<repo>-<branch> -b <branch>`. Clean up on merge.
 - Commit in logical units, each building and passing tests alone.
 - Message: imperative subject under 72 chars, blank line, then **why** —
   the reference formulation matched, the failure that motivated a guard, the
   alternative rejected and the reason.
 - Read your own full diff before committing. Justify everything in it or
   delete it.
 - For anything non-trivial, spawn a review subagent with fresh eyes on the
   diff, the original request, and instructions to hunt scope creep, missing
   tests, broken assumptions, and overcomplication. Fix what's worth fixing;
   tell me what you dismissed and why.
 - Never open a PR with failing checks, or one you haven't read.
 - PR body: what changed, **why this approach over the alternatives**, how it
   was verified, and what I should look at closely. Link the issue.
 - Never commit secrets, `.env`, or large binaries. Check `git status`; stage
   deliberately, never blind `git add -A`.

## 10. Agents and parallelism

 - Delegate separable work: research, exploration, independent test-writing,
   review.
 - Give each agent a self-contained brief — goal, constraints, definition of
   done — not a vague pointer.
 - Don't parallelize agents that would edit the same files.
 - Report what each agent concluded. **Don't absorb its output as fact** —
   subagents state wrong conclusions confidently. Verify claims that matter
   against the code before acting on them.

## 11. Project standards

**Docs are the deliverable, not the leftover.** README and relevant docs
update in the same PR. Docstrings on public functions: what it does, takes,
returns, raises — NumPy style, and include units for every physical
quantity. Comments explain *why*. Non-obvious decisions go in
`docs/decisions/` (this repo) — one short file each.

**Virtual environments are mandatory.** Never install into the system
interpreter. `uv` is the tool here. Every new dependency is pinned and
justified in the PR description. Prefer the standard library; for this repo
the runtime dependency floor is jax + numpy, and TensorFlow/Keras must never
become a runtime dependency — weights load from the FKB text format.

**Repository hygiene.** Source in the package dir, tests co-located as
`*_test.py`, scripts in `scripts/`, docs in `docs/`. No stray files at the
root. No commented-out code, no `file_v2_final.py`, no committed scratch
output. Can't find the right home for a file → ask, don't invent a new
top-level directory.

**Physical quantities carry units.** Every array of physical data names its
units in the docstring or an inline comment. Unit errors are the dominant
bug class in this kind of code.

---

**These rules are working if:** the port's agreement with the reference is a
number in a test rather than a claim in a commit message, diffs contain only
what was asked, plans exist before code does, and clarifying questions arrive
before implementation rather than after mistakes.
