# Finalizing the `extend-groups` branch

Working notes for wrapping up the chain-ring extension and implicit-evaluation
work on this branch. Audit as of 2026-07-12 (v3.1.0).

## What this branch added

- **Chain-ring extension ("Idea 2", exponent-p^d center).** `autstr/chain_ring.py`
  (linear algebra over R = Z/p^d: valuation/units, Smith normal form, `saturate`,
  `right_invertible`/`right_inverse`, `factor_two_sided`, `solve_left`), and the
  ring generalization (default `d=1`, byte-identical) of `CocycleSites`,
  `CutRankGroups` (word automaton) and `CutRankTreeGroups` (tree-merge automaton).
  Exhaustively validated against the reference law.
- **Implicit first-order evaluation.** `autstr/implicit.py` + `check_implicit` on
  the class layer and every built-in group class — evaluate formulas on the fly
  over the base automata (product / on-the-fly powerset / acceptance flip),
  deciding FO on members whose query/base automata cannot be built.
- **Benchmark** `benchmarks/implicit_vs_explicit.py` (VM sweep).

## Loose threads (audit) and plan to finalize

### A. Distributed-center automaton is not ring-generalized  — MAIN GAP
`CocycleSites` supports `d>1` (reference law + module cut-rank), but
`CocycleRankWidthGroups` (the seven-register claim-and-verify microcode for
*distributed* centers) is F_p-only. Guarded now: its `advice()` raises
`NotImplementedError` for a `d>1` sites instead of silently compiling mod p.

**To finalize:** generalize the microcode to R = Z/p^d — saturated interfaces
(via `chain_ring.saturate` / `factor_two_sided`) plus base-p carries for the
claim-and-verify registers. This is the distributed-center analog of the
tree-merge step (commit `e77647a`) and reuses `chain_ring` wholesale. It closes
the last corner of the chain-ring vision (distributed center × exponent p^d).
*Effort: multi-day.* Then expose functional atoms for it so `check_implicit`
decides ring distributed-center members.

### B. `check` / `evaluate` / `get_structure` hang for heavy ring members  — UX
For large q (e.g. Z/8 word, Z/4 tree) building `cls` (the product automaton) is
infeasible, so these methods hang; only `simulate` and `check_implicit` work.

**To finalize:** raise a clear "advice alphabet too large — use `check_implicit`
/ `simulate`" error when `n_letters` (already computed in `__init__`) would make
`cls` infeasible, instead of hanging. *Effort: ~1h.*

### C. Width r >= 2 over the ring  — SCOPE LIMITATION
Blocked by the 20000-letter advice-alphabet cap (`q^(r^2+r+kr)` for word,
`q^(2r^2+r+2kr+kr^2)` for tree). Documented; lifting it needs *factored* advice
letters (carry the factorization pieces separately) rather than an enumerated
alphabet.

### D. No ring embedding constructor
`fixed_k_sites` / `laminar_sites` / `point_target_sites` / `scattered_sites`
build F_p `CocycleSites` (`d=1`). **To finalize:** a ring convenience
constructor (or a `d` parameter on the embeddings) plus reference-law tests over
the ring via it. *Effort: ~half day.*

### Non-issues (checked)
- `chain_ring.right_inverse` / `inv_mod_pp` are still used by tests after the
  `factor_two_sided` rewrite (which now uses `solve_left`); not dead code.
- All `d=1` paths are byte-identical to the pre-branch behavior (regression-tested).
- `module_cut_rank` matches the paper's saturated-width definition.

## Recommended order
1. **A** — the substantive one; completes the chain-ring vision.
2. **B** and **D** — cheap polish, can land alongside A.
3. **C** — larger (factored letters); defer unless width >= 2 rings are needed.

## Also pending on this branch (not ring-specific)
- README update: document the bounded-rank-width classes, chain-ring depth `d`,
  and `check_implicit`; add a changelog entry.
- Implicit evaluation follow-ups: `evaluate_implicit` (satisfying-set primitive),
  functional atoms for `CocycleRankWidthGroups` (so `check_implicit` bypasses its
  heavy microcode build), and a fully-implicit `successors`-based automaton API.
