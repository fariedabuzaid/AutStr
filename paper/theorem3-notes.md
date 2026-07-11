# Theorem 3 machinery: distributed centers via tensor cut-rank

Working notes (2026-07-11). Target: the common generalisation of Theorem 2
(`CutRankTreeGroups`, fixed k) and `TreeExtraspecialGroups` (growing K,
laminar targets). Status: definitions and protocol fixed; width measures
implemented and validated (`autstr/cocycle_groups.py`); automaton + advice
compiler = next milestone.

## 1. Objects

**Site tree.** A binary tree whose N nodes are *sites*, each labelled `x`
(generator) or `z` (central generator). Post-order positions 1..N; X, Z
denote the x- and z-position sets, n = |X|, K = |Z|.

**Cocycle tensor.** T with entries T[j,i,v] in F_p for i < j (both in X)
and v in Z; finitely many nonzero. Presents the group G(T): central
extension of F_p^X by F_p^Z, elements (b, a), law

    (b,a)(b',a') = (b + b' + C(a,a'), a + a'),
    C(a,a')_v    = sum_{i<j} T[j,i,v] a_j a'_i.

Bilinear in (a, a') => 2-cocycle => group of order p^{n+K}
(same proof as Lemma `lem:cocycle` in the paper).

**Verification per element triple.** For every v in Z:

    rho_v := (b_z - b_x - b_y)_v  =  C(a_x, a_y)_v .        (*)

x-sites carry the position-wise digit check a_{z,w} = a_{x,w} + a_{y,w}.

## 2. The six crossing flattenings

For a cut S (subtree; post-order interval), classify each nonzero triple
(j, i, v) by which of its sites lie in S. Six matrices (empty rows/columns
dropped; entries T[j,i,v]):

| name | rows            | columns          | traffic                          |
|------|-----------------|------------------|----------------------------------|
| F_y  | i in S          | (j, v) not in S  | y-digit functionals, upward      |
| F_x  | j in S          | (i, v) not in S  | x-digit functionals, upward      |
| F_m  | (j,i) both in S | v not in S       | resolved pair-sums, upward       |
| F_g  | v in S          | (j,i) not both in S | claims (expected inward flow) |
| F_py | i in S          | (j not in S, v in S) | mixed: inside y-coefficient of a future x-digit, destined inward |
| F_px | j in S          | (i not in S, v in S) | mixed, x-side                 |

width(S) := max of the six ranks; width of the layout := max over all
subtree cuts S. NOTE: these are *not* interderivable from one bipartition
rank — reshaping a flattening changes its rank (e.g. T[j,i,v] = [v = f(i)]
has small [(i,v)]x[j] rank but large [i]x[(j,v)] rank), so the definition
must list them. All six are needed operationally; all six are computable
by Gaussian elimination, which is what `cocycle_groups.cut_profile` does.

## 3. The protocol (bottom-up, deterministic)

State at subtree S (all vectors over F_p, dimensions <= width r):

- `wy` = values of an advice-chosen row basis of F_y applied to (a_y)|_S,
  and `wx` similarly for F_x  (as in Theorem 2);
- `phy` = values of a row basis of F_py applied to (a_y)|_S, and `phx`
  for F_px (the mixed exports: coefficients that future x/y digits will
  multiply, with the product flowing back *into* S's claims);
- `m` = coordinates, in a column basis of F_m, of the accumulated
  contribution vector {sum of resolved inside pairs -> v}_{v not in S};
- `g` = claim coordinates: rho|_S (the residual vector of (*) over v in S,
  after subtracting all inside-resolved contributions) must lie in the
  column space of F_g; the advice fixes a full-column-rank basis W_S and
  the automaton carries the unique g with W_S g = rho|_S, rejecting if the
  system is inconsistent. Uniqueness makes the automaton deterministic;
  consistency of the system IS part of the verified property.

Transitions (letter data supplied by the advice compiler):

- **z-site v:** read digits b_{x,v}, b_{y,v}, b_{z,v}; the new residual
  coordinate enters the claim: rebase W_child -> W_S and solve for the
  extended g (advice: rebase matrix + injection column).
- **x-site w (unary/leaf):** digit checks; read-off of pairs {w, i} with i
  below (into local residuals if the target is below — adjusting g through
  the basis, since the contribution vector of a pair that was outside the
  child cut is a column of the child's F_g and hence in the claimed space;
  into m if the target is elsewhere); update wy/wx/phy/phx by
  basis-change + own-digit column (Lemma treestream shape).
- **binary merge u (children L, R):** post-order puts all of L before all
  of R. New-resolved traffic and its factorisations:
  1. split pairs (i in L, j in R), target v outside u: bilinear pairing
     wx_R^T Q wy_L accumulated into m (Theorem 2 mechanism);
  2. split pairs with target v in L: pairing of wx_R against phy_L,
     discharged against L's claim (advice translates into W-coordinates);
  3. split pairs with target v in R: pairing of phx_R? no — orientation:
     i in L is the smaller endpoint, so the y-side is L: pairing of wy_L
     against R's mixed export phx_R, discharged against R's claim;
  4. pairs entirely in L with target v in R: L's m-export evaluated
     against R's claim (advice matrix maps m_L-coordinates to
     W_R-coordinates), and symmetrically;
  5. remaining claims rebase to W_u; remaining exports rebase to the
     u-bases.
  Every advice matrix in 1–5 exists because the needed coefficient rows
  are sub-rows of the children's flattenings (restriction argument), plus
  the two-sided factorisation lemma for the bilinear pairings.
- **Root:** nothing is outside; F_g, F_m, F_y, F_x, F_py, F_px are all
  empty; accept iff the claim is the empty vector (all residuals were
  discharged) — g has dimension 0 at the root by construction.

State space: p^{6r} plus bookkeeping. This subsumes:

- **Theorem 2** (z-sites on a chain above the layout root): F_py, F_px,
  F_g empty at layout cuts (no z inside); F_m rank <= k; F_y/F_x = the old
  crossing blocks. Claims never arise; m plays the role of the old s in
  the basis of F_m. Old state (s, wx, wy) recovered.
- **TreeExtraspecialGroups** (z at leaves, pairs co-located, laminar
  targets): F_y = F_x = F_py = F_px = 0 (pairs never split), F_m = 0
  (targets below their pair), F_g = all-ones (ancestor pairs owe every
  leaf below the cut equally) => width 1, and g = "deficit owed by the
  ancestors"; merge case 4/5 degenerates to "siblings must owe equally".

## 4. Letter-size problem and the microcode encoding

A monolithic letter needs ~(register rebases + pairings + read-offs) ≈
18 r^2 + O(r) digits — at p = 2, r = 1 that is ~2^24 flat letters:
infeasible for the flat-alphabet engine even at width 1 (unlike Theorems
1–2). Two options:

(a) **Microcode advice**: expand each logical site into a bounded chain of
    micro-op nodes, each carrying an opcode tag plus O(1) digits; the
    automaton applies one linear micro-update per node. Alphabet size
    O(p * #opcodes), *independent of r*; advice length O(r^2 N). Elements
    carry forced-0 digits at micro nodes (structural nodes). This is the
    honest theoretical statement too: constant alphabet, linear advice.
(b) **Factored (MTBDD-native) letters** in the engine — the engine
    already stores transitions as shared MTBDDs, so a letter =
    bit-vector interface is natural, but it is an engine change.

Decision: implement (a); it needs no engine changes and improves the
theorem statement.

## 5. Validation plan (this milestone)

`autstr/cocycle_groups.py` implements sites, the reference law, and
`cut_profile`/`cut_width` (the six ranks per cut). Tests validate:

1. the reference law is a group (exhaustive, small);
2. fixed-k embeddings: width == max(old width, k)-ish on clique/matching
   forms, and the law agrees with `CutRankTreeGroups.multiply`;
3. laminar embeddings: width == 1 on random shapes, and the law agrees
   with the `TreeExtraspecialGroups` law;
4. scattered private targets: width grows with n (the definition
   correctly charges inward flows);
5. a genuinely new width-1 family covered by NEITHER existing class:
   z-sites at leaves, each pair targeting only the *leftmost* leaf below
   it (point targets: non-laminar-product law, K growing).

## 5b. Findings from the implementation (2026-07-11)

The protocol and its compiler are implemented (`CocycleRankWidthGroups`,
r = 1) and machine-verified: the automaton's transition function is shared
between `sta_from_delta` and a direct tree `simulate`, so the simulator
tests verify exactly the automaton's run. All corner instances, the
point-target family, mixed-claim instances, an F_px-exercising instance,
and a seeded fuzzer over random width-1 tensors agree exhaustively/sampled
with the reference law; every compiler constant is guarded by an
AssertionError instantiating a restriction lemma, and none fired.

Design corrections discovered by the machine check:

1. **Raw-side merges.** Pre-rebasing both children's registers to the
   joint basis before the merge *destroys* the values the sibling products
   consume: the sibling traffic factors through the *child* cuts (that is
   the theorem), so the merge letter itself must fold the raw sides. Final
   architecture: L keeps wy/phy raw, R keeps wx/phx raw, m is never
   pre-rebased (its dropped columns feed the cross m->claim terms at the
   same merge); the merge letter carries 13 constants: 6 folds (including
   the two mixed-bank couplings), fml/fmr for m, the three sibling
   products, and the two cross m->claim translations.
2. **The letter budget is worse than estimated.** 13 constants means
   2 p^13 merge letters; moreover the micro-op letters act *totally* on
   the register states, so the flat per-pair transition diagrams are dense
   (live on all state pairs), unlike every previous class in the package.
   Consequence: the flat enumeration builder cannot construct the
   full-alphabet automata at any p, and even sub-alphabet builds are
   multi-GB (gate under AUTSTR_HEAVY, run under a systemd-run memory cap).
   The engineering fix is factored (MTBDD-native) transition letters; the
   *theoretical* constant-alphabet claim of section 4 is unaffected but
   the cheap 7-register hot path does not realise it -- the generic
   normalization (buffer registers + generator words) trades alphabet for
   state count and advice length, as the lemma says, not for free.
3. `CocycleRankWidthGroups(p, merge_letters=...)` instantiates the
   presentation over a sub-alphabet collected from compiled advices
   (`used_merge_letters`), which is a legitimate uniform presentation of
   the sub-class its advice language covers; the end-to-end test uses it.
   Measured under a 4G cgroup cap (2026-07-11): the automaton build plus
   exhaustive multiplication checks over three instances PASS within 4G --
   the theorem's automaton runs for real -- while a three-quantifier FO
   evaluation over the same automaton exceeds 4G (dense diagrams x subset
   construction). So on a 12GB laptop: transition function, compiler and
   end-to-end multiplication are fully machine-verified; compiled FO query
   evaluation on this class awaits factored letters or a larger machine.

## 6. Open ends toward the full proof

- Write the restriction lemmas once, parameterised by flattening pair
  (child cut, parent cut, which of the six) — one lemma, six instances.
- The claim-rebase lemma: W_child restricted-and-extended vs W_parent;
  needs "columns of child F_g are columns of parent F_g or resolved at
  the parent" — check the exact statement when the parent is a z-site
  (residual injection) vs x-site vs merge.
- Determinism edge: W_S full column rank is a *choice*; the compiler must
  pick bases so that rank never silently drops between parent/child
  (padding rows are fine, padding columns of W are not — keep W minimal).
- Micro-op schedule: fix the per-site-type opcode sequence and prove the
  composite update equals the abstract transition.
