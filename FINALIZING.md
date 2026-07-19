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

## Implicit-presentation follow-ups — DONE (2026-07-18)

- **`evaluate_implicit` (satisfying-set primitive).**
  `StringSolutionSet`/`TreeSolutionSet` in `autstr/implicit.py`: for a formula
  with *open* free variables over a fixed advice, one forward pass over the
  reachable composite states plus a backward count DP give the exact number of
  satisfying assignments without enumeration (`len`), and iteration lazily
  yields them ({var: word} / {var: tree}). Exposed as `evaluate_implicit` on
  `UniformlyAutomaticClass` / `UniformlyTreeAutomaticClass` (raw
  words/trees) and on `CutRankGroups` / `CutRankTreeGroups` /
  `CocycleRankWidthGroups`, which decode back to (b, a) tuples via new
  `decode` inverses of their encoders (the tree decoders walk advice and
  element tree in parallel, skipping factored entry stretches). Validated
  against brute force (centralizers, inverse pairs, domain counts) on field
  members and end-to-end on Z/8, factored width-2 Z/4, and ring microcode
  members whose automata cannot be built.
- **Fully implicit presentation type.** `ImplicitClass` /
  `ImplicitTreeClass`: a uniformly automatic class given purely functionally
  (atoms as `args -> ImplicitDFA/TA` builders + element alphabet), offering
  `check` and `evaluate` only — nothing is ever compiled. The heavy group
  classes now route `check_implicit`/`evaluate_implicit` through their
  `implicit_cls` property.

## Graphs of bounded rank-width — ADDED (2026-07-18)

`RankWidthGraph` / `RankWidthClass` in `autstr/tree_graphs.py`: the graph
analog of the bounded-rank-width group classes, sharing their `chain_ring`
linear algebra at p = 2, d = 1. The advice is a rank decomposition annotated
with basis-change matrices per child and the sibling-block bilinear form Q
per binary node; adjacency of x and y is w_y^T Q w_x at the meet, so the E
automaton carries only the marked vertices' r-bit interface vectors. MSO0
signature (Sing, Subset, E) with union-of-root-path set marks like the other
graph classes; `check_implicit`/`evaluate_implicit` run over functional
atoms (set assignments padded to the advice shape — the implicit evaluator
is synchronous). Flat letters cap r at 2 (2^{3r^2} binary letters; factored
letters as in the group classes are future work). Validated: family widths
(cliques/paths/K_{a,b} width 1, cycles 2), E == edge set on families and 15
random graphs (explicit + implicit), MSO 2-colourability decided class-wide
at r = 1, neighborhood/domain satisfying sets. tests/test_rank_width.py.

## Documentation pass — DONE (2026-07-18)

- **README**: new sections "Bounded rank-width: groups and graphs from one
  linear algebra" and "Implicit evaluation: members whose automata cannot be
  built" (both with verified snippets), classes-table rows for the cut-rank
  group classes / `autstr.cocycle_groups` / rank-width graphs, and the v3.1
  changelog entry.
- **Notebooks are now stored output-free** and executed as part of the docs
  build: `myst-nb` added to the Sphinx pipeline (`docs/source/conf.py` copies
  `notebooks/*.ipynb` into the source tree at build time — `büchi` renamed to
  the ASCII docname `buechi` — and executes them with
  `nb_execution_mode = 'force'`, errors failing the build); a Notebooks
  toctree in `index.rst`; `myst-nb` + `ipykernel` in the `docs` extra;
  the run_docs workflow installs the graphviz `dot` binary. Three notebooks
  gained title cells (arithmetic, büchi, mso0). `show_diagram` now defaults
  to `view=False` (headless builds must not spawn a viewer).
- Version stays v3.1.0 for this branch (no further bumps).

## Ring-interface finding (2026-07-19, paper-rewrite fuzzing)

Adversarial fuzzing over valuation-rich random forms found latent compile
failures in all three ring (d > 1) compilers on **width-admissible**
instances — the branch's structured test forms (clique/matching/star/
laminar) never hit them. Root cause: **pure closures are non-unique over
Z/p^d and do not nest under column restriction** (over Z/4, span{(1,0,1)}
and span{(1,2,1)} are both minimal pure overmodules of span{(2,0,2)}), so
per-cut saturated interfaces can be incompatible with the next cut's
transition solve.

- **Word compiler: FIXED.** The correct interface is a minimal *generating
  set of the row module* (Smith with the p-powers kept): restrictions of
  row spaces land in row spaces exactly, so transitions always solve.
  Validated by 10,636 fuzzed ring forms (0 compile failures, 0 simulate
  mismatches) + regression test
  (`test_interface_is_row_module_not_saturation`); d = 1 byte-identical.
- **Tree compiler: RESOLVED (2026-07-19, stronger than hoped).** The
  automaton-theoretic requirement is kernel containment (well-definedness
  of the merge as a function of the registers), not matrix factorability.
  Row-module interfaces satisfy it automatically; the merge contribution
  is then a well-defined R-bilinear function on the register images that
  need not extend to a Q matrix over R (Z/4: c = 2*(w/2)(v/2)) -- so the
  merge letter carries a bounded *pairing table* (q^{2r} entries per
  center coordinate, streamed in factored mode; d > 1 forces factored
  letters). Interfaces stay at module cut-rank r -- no r*d blowup, no
  saturation anywhere in the construction; d = 1 keeps the flat Q-matrix
  letters byte-identically. Validated: 3350 fuzzed width-admissible forms
  over Z/4 (widths 1-2), Z/9 and Z/8 -- 0 compile failures, 0 simulate
  mismatches -- including the scratch counterexample instance; the
  membership solves remain as compile-time lemma certificates. Regression
  test: `test_ring_interfaces_and_pairing_tables`.
- **Microcode compiler (CocycleRankWidthGroups, d > 1): OPEN.** 19/400
  fuzzed width-1 Z/4 tensors fail, including "module rank > 1" at *joint
  child intervals* (not subtree cuts — over the ring the six-flattening
  width does not bound them, unlike the field case).

Until resolved: d > 1 tree/microcode members compiled from the structured
families keep working (compile success is still a machine-checked
certificate — the guarded solves are exactly the lemma instances); random
valuation-rich instances may be rejected with a misleading width message.

## Non-issues (checked)
- `chain_ring.right_inverse` / `inv_mod_pp` are still used by tests after the
  `factor_two_sided` rewrite (which now uses `solve_left`); not dead code.
- All `d=1` paths are byte-identical to the pre-branch behavior
  (regression-tested; the claim-and-verify compiler additionally checked
  letter-for-letter against the pre-change implementation).
- `module_cut_rank` matches the paper's saturated-width definition.
