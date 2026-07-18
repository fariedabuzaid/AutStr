# Finalizing the `extend-groups` branch

Working notes for wrapping up the chain-ring extension and implicit-evaluation
work on this branch. Original audit 2026-07-12; updated 2026-07-18 after the
close-out pass (target version: v3.1.0).

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

## Loose threads: all four closed (2026-07-18)

### A. Distributed-center automaton ring-generalized  — DONE
`CocycleRankWidthGroups(p, merge_letters=None, d=1)` now runs over R = Z/p^d:
saturated interfaces from `chain_ring.saturate` in `_r1_bases`, all compiler
solves through `chain_ring.solve_left`, constants as d base-p digits in the
letter names (d = 1 byte-identical — verified against the pre-change compiler
on 36 field instances plus fuzz). The one genuinely new ring ingredient is the
**truncated claim invariant**: over the ring the claim register can only carry
the determined truncation `p^s * P` of the claim coordinate; the compiler
tracks the valuation `s` statically, normalizes unit parts through `sg`,
cross-multiplies consistency checks (`zc`/`bk` with ring constants), and
re-anchors with the ring-only micro-op **`zr`** when a later z-position
determines more of the claim than the ones before it. Validated against the
reference law over Z/4 and Z/9 (hand-picked instances incl. the re-anchor
case, plus 30 fuzzed width-1 ring instances). Functional implicit atoms
(`_implicit_atoms`) added, so `check_implicit` decides ring and full-ISA
members without building any automaton. Nested existentials over large ring
members remain expensive (on-the-fly powerset over the q^7 register space):
3 quantifiers is fine on 3-site Z/4 members, heavy on 5-site ones.

### B. Infeasible `cls` builds now raise instead of hanging  — DONE
All three classes (`CutRankGroups`, `CutRankTreeGroups`,
`CocycleRankWidthGroups`) estimate the build's transition-enumeration count in
the lazy `cls` property (`_cls_cost`, caps 2e7 word / 5e7 tree) and raise a
ValueError pointing at `check_implicit`/`simulate`. Z/4 word and p<=3-field
tree members still build; Z/8 word, Z/4 tree, and the full microcode ISA are
gated.

### C. Factored advice letters — width r >= 2 over the ring  — DONE
`CutRankGroups(..., factored=None)` and `CutRankTreeGroups(..., factored=None)`
auto-switch to *factored* letters when the flat alphabet would exceed 20000
letters (flat stays byte-identical below the cap; `factored=False` forces the
old error, `factored=True` forces factored). Word: each position becomes a
marker 'n' plus one letter per ring entry (T row-major, v, R row-major), q+1
advice letters total; the automaton streams the update through an accumulator
(state `('x', d, w, acc, phase)`). Tree: each layout node becomes a bare
marker 'a'/'b'/'d' with the entry chain above it (q+4 letters); binary
stretches stream TL, TR, v, RL, RR, Q over a frame carrying both children's
functionals. Element digits repeat along stretches (universe enforces
constancy). Validated: flat/factored agreement where both exist; width-2 over
Z/4 (word and tree, spine + balanced, incl. a rank-2 sibling block through
`factor_two_sided`); width-3 field; the factored Z/4 r=2 *word* automaton even
builds explicitly (~40s) and matches `simulate`. Factored *tree* `cls` builds
remain gated by B's cost cap (the streaming frames square in the pair
enumeration); `simulate`/`check_implicit` are the intended path there.

### D. Ring embedding constructors  — DONE
`fixed_k_sites`, `laminar_sites`, `point_target_sites`, `scattered_sites` all
take `d=1`; over the ring they build `CocycleSites(p, ..., d)` with
coefficients mod q. Tested: fixed_k over Z/4 agrees with
`CutRankTreeGroups(2, d=2).multiply` (incl. valuation-1 labels), laminar /
point-target Z/4 embeddings have module cut-width 1 and run through the ring
claim-and-verify microcode, scattered width still m over the ring.

## Still pending on this branch (not ring-specific)

- README update: document the bounded-rank-width classes, chain-ring depth `d`,
  factored letters, and `check_implicit`; add a changelog entry.
- Implicit evaluation follow-ups: `evaluate_implicit` (satisfying-set
  primitive), and a fully-implicit `successors`-based automaton API.
- Version stays v3.1.0 for this branch (no further bumps).

## Non-issues (checked)
- `chain_ring.right_inverse` / `inv_mod_pp` are still used by tests after the
  `factor_two_sided` rewrite (which now uses `solve_left`); not dead code.
- All `d=1` paths are byte-identical to the pre-branch behavior
  (regression-tested; the claim-and-verify compiler additionally checked
  letter-for-letter against the pre-change implementation).
- `module_cut_rank` matches the paper's saturated-width definition.
