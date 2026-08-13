# Changelog

All notable changes to AutStr are recorded here. Versions follow
[semantic versioning](https://semver.org): the major number changes when
existing code has to be edited to keep working.

Each release since v2.0 is also a snapshot of an experiment in AI-assisted
algorithm engineering — see the [README](https://github.com/fariedabuzaid/AutStr#readme) for that story.

---

## v4.0.0 — August 2026

**The symbolic release.** Queries stop being strings, structures start
declaring their own vocabulary, and the package stops pretending some of its
structures are more "built in" than others.

### Added

**A generic first-order layer — `autstr.symbolic`.**
Variables, relation symbols and function symbols that compose with ordinary
Python operators, compiled against either engine and against uniformly
automatic classes:

```python
x, y, z = BuechiArithmeticZ().symbolic().vars('x y z')
phi = (x + y).eq(z) & z.lt(100)
phi.check()                      # satisfiable?
(3, 4, 7) in phi                 # membership
list(phi.drop(z))                # enumerate, smallest first
```

- `Signature` declares which relations are function graphs, which Python
  operator each binds to, and — via an `ElementCodec` — how Python values
  encode as elements. Terms nest (`((x + y) + z).eq(10)`), arities come from
  the automata, and variable names survive into results.
- Quantifiers (`.all`, `.drop`, `.exinf`), Boolean connectives, `is_finite`,
  membership, and lazy enumeration in length-lexicographic order.
- The same expressions compile against a class, where `check_member` and
  `evaluate_member` decide one member — explicitly or implicitly.
- Ready-made signature builders: `operation_signature`, `order_signature`,
  `graph_signature`, `relational_signature`.

**Every structure declares its own vocabulary.**
`symbolic()` now takes no argument on every structure the library ships:
each provides `default_signature`, so operators and a codec are there from the
start. Büchi arithmetic over ℕ and over ℤ and the MSO0 powerset structure
gained theirs in this release; MSO0's is new capability rather than
convenience — union, intersection and difference as `+`, `*`, `-`, with
solutions returned as Python sets.

**First-order interpretations — `autstr.interpretations`.**
`interpret` *computes* the interpreted presentation from a domain formula, a
formula per relation, an optional dimension *k* (elements are *k*-tuples) and
an optional definable equivalence:

- quotients over words take the shortlex-least member of a class as its
  representative;
- quotients over **trees** take the least *description*, since no order on
  trees is well-founded (Kuske & Weidner) — the same idea made to work without
  a well-order;
- `representatives` exposes the chosen representative of each class.

**Infinite graphs — `autstr.infinite_graphs`, `autstr.turing`.**
`InfiniteGraph` is the shared surface (edge relation, symmetry check, codec);
concrete members are `IntegerGrid` (the Cayley graph of ℤⁿ, built as an
asynchronous product), `RegularTree` (T_k with successors and the prefix
order), and `ConfigurationGraph` for a Turing machine — whose first-order
theory is decidable while reachability, being halting, is not.

**Ordinals — `autstr.ordinals`.** The ordinals below ω^ω on the string engine
and below ω^(ω^n) on the tree engine, each *interpreted* in arithmetic rather
than authored: three formulas over Büchi arithmetic, with the
reverse-lexicographic order on Cantor coefficients.

**The countable atomless Boolean algebra — `autstr.tree_algebra`.** The clopen
algebra of Cantor space, with meet, join and complement as terms, and
`is_atomless` as a decided sentence rather than an assertion.

**Level 2 collapsible pushdown graphs — `autstr.collapsible`.**
Stacks of stacks with collapse links, presented by Kartzow's tree encoding:
blocks hang off each other as right children, links are recovered from the
tree's shape rather than stored, and every stack operation is a bounded rewrite
at the end of the last path. MSO over these graphs is undecidable, so the tree
encoding is the only automatic route to them.

**Reachability for those graphs — `autstr.collapsible_reach`.**
`Reach` is a relation of the configuration graph, so a first-order formula may
ask about runs of *any* length — the question a Turing machine's configuration
graph cannot answer. Following Kartzow's decomposition, every run splits into
four stretches and reachability is the first-order formula

    Reach(x,y) = ∃d ∃f ∃g. A(x,d) ∧ B(d,f) ∧ C(f,g) ∧ D(g,y)

over four reflexive relations — no fifth automaton and no fixpoint over trees.
Underneath, `Summaries` computes per-word returns and loops by saturation
(the paper's own effectiveness argument routes through µ-calculus model
checking, which is not needed), and the nondeterministic guesses of Kartzow's
automata are carried on annotation tapes that are projected away — the move
MONA makes. `reach_along` builds the label-constrained `Reach_L` by putting
the label automaton into a product system; ε-contraction is a special case.

**Deferred relations.** A presentation may declare a relation now and build it
on first use. `get_relation_symbols()` lists it, a query that mentions it
triggers the build. This is how `Reach` avoids costing anything until asked
for, and it is the opt-out for every expensive relation.

**Smaller additions.** `partial_dfa` / `partial_tree_automaton`;
`restrict_alphabet`; `fold_tapes` for tree automata; `tree_order` and
`domain_within`; sentences usable as operands of a connective; an `Eq`
relation on every presentation.

### Changed

- **`autstr.buildin` is gone.** Every structure the library ships is built in,
  so the name carved nothing. Its contents moved to where their subject lives:

  | was | now |
  |---|---|
  | `autstr.buildin.presentations` | `autstr.arithmetic` (Büchi ℕ and ℤ) and `autstr.powerset` (MSO0) |
  | `autstr.buildin.tree_presentations` | `autstr.tree_arithmetic` (Skolem arithmetic) |
  | `autstr.buildin.automata` | `autstr.utils.automata_tools` |

- **`autstr.arithmetic` is a different module.** It holds the two Büchi
  presentations themselves; the old term-algebra front end
  (`VariableETerm` and friends) is replaced by the symbolic layer.
- `encode` / `decode` are now static methods on the presentations that own
  them.
- Documentation builds execute every notebook, and the repository keeps them
  output-free.

### Fixed

- **Enumeration over a product alphabet.** Every interpreted structure of
  dimension > 1 — including the shipped `Ordinal(2)` — produced unusable
  results from `iter(...)`: tapes were built by string-concatenating letters,
  which holds only while a letter is a single character, so a tuple letter was
  flattened into text no codec could read. Membership was unaffected, which is
  why it went unseen.
- **Serializing such a presentation.** JSON has no tuples, so a reloaded
  product alphabet was a set of lists (unhashable) and the padding symbol a
  list. Both are restored on load.
- A universal quantifier that silently became an existential.
- Sentence markers that dropped the structure's alphabet.
- Relation automata are restricted to the universe on *every* tape, not all but
  the last: a relation that still held of a non-element made universal
  sentences false on encodings that were not elements at all.
- Spliced relations are restored even when a query fails.

### Removed

- **The precompiled `.autstr` artifacts.** They saved 0.03 s, 0.23 s and
  0.44 s against building from scratch, and the built presentations are
  identical to the loaded ones relation by relation. Serialization itself is
  unchanged and fully supported — see the composition notebook.
- `scripts/gen_builtin_presentations.py`, which generated them.

### Upgrading from 3.x

```python
# 3.x
from autstr.buildin.presentations import BuechiArithmeticZ, MSO0
from autstr.buildin.tree_presentations import skolem_arithmetic
from autstr.arithmetic import integers, encode, decode

# 4.0
from autstr.arithmetic import BuechiArithmeticZ          # .encode / .decode on it
from autstr.powerset import MSO0
from autstr.tree_arithmetic import skolem_arithmetic
Z = BuechiArithmeticZ().symbolic()                        # was integers()
```

---

## v3.1.0 — July 2026

The rank-width release: new mathematics on top of the v3 engines.

### Added

- **Class-2 groups of bounded rank-width.** `CutRankGroups` (linear layouts),
  `CutRankTreeGroups` (tree layouts) and `CocycleRankWidthGroups` (distributed
  centers, microcode advice) — the advice spells out rank-≤r factorizations of
  the commutation form's crossing blocks, cut by cut.
- **The chain-ring extension** (`autstr.chain_ring`). Everything generalizes
  from F_p to R = ℤ/pᵈ: Smith normal form, saturated interfaces, the two-sided
  factorization lemma. Centers of exponent pᵈ, widths measured as module
  cut-rank, byte-identical at d = 1.
- **Factored advice letters.** Beyond ~20000 flat letters the cut-rank classes
  stream one ring entry per letter through accumulator states, making width
  r ≥ 2 over the ring representable (q+1 advice letters).
- **Implicit evaluation** (`autstr.implicit`). `check_implicit` /
  `evaluate_implicit` on every class: first-order model checking and
  satisfying-set computation that never build a query — or even a base —
  automaton, reaching members (ℤ/8 and ℤ/9 words, ℤ/4 trees, the
  distributed-center protocol) whose automata are infeasible. `ImplicitClass` /
  `ImplicitTreeClass` are presentations given purely by transition functions.
- **Graphs of bounded rank-width.** `RankWidthClass` — rank decompositions as
  advice, adjacency as a bilinear form on r-bit interface vectors, full MSO.

### Changed

- `show_diagram` no longer opens an external image viewer by default; pass
  `view=True` for the old behavior.

---

## v3.0.0 — July 2026

From strings to trees, with the transition representation replaced underneath
both engines.

### Added

- **Tree-automatic structures.** `autstr.sparse_tree_automata` (bottom-up tree
  automata), `autstr.tree_presentations` and `autstr.tree_uniform` — the tree
  counterparts of the whole stack. New members: **Skolem arithmetic** (ℕ, ·),
  graphs of bounded **tree-width** and **clique-width** with full MSO, and
  tree-indexed **extraspecial p-groups**. Cross-validated by embedding the
  string engine's Büchi arithmetic into the tree engine and re-deciding every
  sentence through both.
- **Composing presentations** (`autstr.composition`): disjoint union,
  synchronous and asynchronous direct products of automatic structures, union
  of uniformly automatic classes, and the direct-product closure of a class.
  Composed, they present every finite direct product of index-≤2 cyclic groups
  and extraspecial p-groups, from either family, and decide that such a product
  is abelian exactly when all of its factors are.

### Changed

- **Transitions are shared multi-terminal BDDs** over the symbol's digits, in
  both engines (`autstr.mtbdd`). `expand` became a variable renaming,
  `complement` stopped touching diagrams at all, and `minimize` became one
  `apply` per state per round. Queries impossible for lack of alphabet width
  now compile: an arity-5 relation over a 14-letter alphabet (14⁵ = 537 824
  flat symbols) went from *infeasible* to 0.2 s; tree-depth-4 bipartiteness
  from 17 s to 0.4 s. The test suite went from ~2 min to ~35 s.

---

## v2.0.0 — July 2026

### Added

- The whole uniformly-automatic layer: the generic advice machinery, bounded
  tree-depth and pathwidth graphs with MSO, finite Boolean algebras, finite
  abelian groups, the ℤ[1/p] presentations, and the non-abelian group classes —
  each verified against exhaustive or exact ground-truth oracles.
- The [benchmark suite](https://github.com/fariedabuzaid/AutStr/tree/main/benchmarks) and the documentation site.

### Changed

- The automata core was profiled and rewritten as batched, sparsity-aware
  NumPy: a **10²–10³× speedup** (the reference query dropped from 85 s to
  0.03 s), with linear memory.
- JAX went from a hard dependency to an optional accelerator; NumPy is the
  canonical core.

---

## v1.x — July 2025

Added the sparse-DFA backend, serialization, and the MSO0 finite-powerset
structure; modernized packaging.

---

## v1.0 — 2022

The original library and its arithmetic front end.
