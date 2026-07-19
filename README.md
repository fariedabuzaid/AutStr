# AutStr

[![PyPI](https://img.shields.io/pypi/v/autstr?color=blue)](https://pypi.org/project/autstr/)
[![Python](https://img.shields.io/badge/python-3.10--3.14-blue?logo=python&logoColor=white)](https://pypi.org/project/autstr/)
[![License: AGPL v3](https://img.shields.io/badge/license-AGPL--3.0-blue)](LICENSE)
[![NumPy](https://img.shields.io/badge/powered%20by-NumPy-013243?logo=numpy&logoColor=white)](https://numpy.org)
[![JAX](https://img.shields.io/badge/optional-JAX-9c27b0)](https://github.com/jax-ml/jax)
[![NetworkX](https://img.shields.io/badge/graphs-NetworkX-2c5bb4)](https://networkx.org)
[![NLTK](https://img.shields.io/badge/parsing-NLTK-154f5b)](https://www.nltk.org)

**Compute with infinite structures in Python — one formalism, many roles.**

AutStr represents infinite mathematical structures — the integers, the rationals
ℤ[1/p], whole classes of finite graphs and groups — as *finite automata*, and
lets you query them with first-order and monadic second-order logic. Because the
representation is exact and the logic is decidable, a single small framework acts
as several tools at once:

- 🧮 **a computer algebra system** for infinite domains — manipulate infinite
  sets and relations (ℤ, ℤ[1/p], …) with exact algebra, not floating point;
- ⊢ **a decision procedure / theorem prover** — decide first-order and MSO
  statements over infinite structures (Presburger and Büchi arithmetic, MSO over
  graphs), returning a proof-carrying yes/no;
- 🔬 **a (finite) algebra & model-theory system** — decide a property across an
  *entire family* of finite structures (all finite abelian groups, all graphs of
  bounded tree-depth) with one compiled automaton;
- ⚙️ **an algorithm synthesizer** — turn a logical *specification* into a
  **provably linear-time algorithm**. Problems that are NP-hard on general inputs
  become linear-time decisions on structurally restricted ones, running at tens
  of millions of elements per second.

All four are the same underlying object — an *automatic presentation* — viewed
from different angles.

> 📖 For a thorough introduction to the library, please consult the
> **[Documentation](https://fariedabuzaid.github.io/AutStr/)**.

---

## Quick start

```bash
pip install autstr
```

The friendliest entry point is the arithmetic package: relations over the
integers are first-class, exactly-represented **infinite** objects that you
combine with relational algebra.

```python
from autstr.arithmetic import VariableETerm as Var

x, y, z = Var('x'), Var('y'), Var('z')

R = (x + y + 3).lt(2 * x)     # the infinite set { (x, y) : x + y + 3 < 2x }
R.isempty()                   # False
(0, 4) in R                   # False   — membership test

band = (x + y).eq(z) & z.gt(0) & z.lt(3)   # x + y = z  ∧  0 < z < 3
solutions = band.drop(['z'])               # project z away (an ∃ quantifier)
for s, _ in zip(solutions, range(3)):      # enumerate lazily, smallest-first
    print(s)                               # (0, 1), (1, 0), (1, 1), ...
```

Nothing is materialized until you iterate; `& | ~` are exact operations on
infinite sets.

---

## Structures and classes

AutStr represents two kinds of thing, and the distinction shapes everything you
do with it.

### A structure — one infinite object

(ℤ, +, <), the rationals (ℚ, +) = ℤ[1/p], Skolem arithmetic (ℕ, ·). An
`AutomaticPresentation` bundles automata for a structure's domain and relations.
Because the first-order theory of an automatic structure is **decidable**,
`check` always terminates with a definite answer — a theorem prover for the
fragment these structures capture — and `evaluate` returns the automaton of *all*
satisfying assignments, which you can enumerate or reuse.

```python
from autstr.buildin.presentations import BuechiArithmeticZ

Z = BuechiArithmeticZ()                          # (ℤ, +, <, |) as automata
Z.check('all x.(exists y.(A(x,y,x)))')           # ∀x ∃y: x+y=x   — True
Z.check('exists x.(all y.(Lt(x,y)))')            # a least integer? — False
```

### A class — a whole family at once

A **uniformly automatic class** presents an entire *family* of finite structures
by giving every automaton one extra tape that reads an **advice string**
synchronously with the elements; fixing the advice instantiates one member. A
query is compiled **once for the class** and then decides *any* member by running
its advice word through the resulting automaton — a single linear pass.

On classes of bounded width this is a constructive, streaming form of
**Courcelle's theorem**: a declarative MSO specification becomes a **linear-time
algorithm**, even for properties that are NP-hard in general.

```python
import networkx as nx
from autstr.graphs import TreeDepthClass, TreeDepthGraph

cls = TreeDepthClass(3)                           # ALL graphs of tree-depth ≤ 3
bipartite = ('exists c.(all x.(all y.((not E(x,y)) or '   # MSO, compiled once
             '((Subset(x,c) and (not Subset(y,c))) or '
             '((not Subset(x,c)) and Subset(y,c))))))')

cls.check(bipartite, TreeDepthGraph.from_networkx(nx.cycle_graph(3)))  # triangle → False, in µs
cls.check(bipartite, TreeDepthGraph.from_networkx(nx.path_graph(6)))   # path → True
```

Deciding the property on a member is linear in its size and batches beautifully
(optionally on a GPU via the JAX backend) — measured, a through-the-origin
R² = 1.0000 across three orders of magnitude, ~90 million vertices / second in
batch:

![Linear-time MSO query evaluation](benchmarks/runtime_curves.svg)

---

## What's implemented

**Structures** — single automatic presentations:

| package | structures |
|---------|------------|
| `autstr.arithmetic`, `autstr.buildin` | Presburger and Büchi arithmetic (ℤ, +, <, \|₂), Skolem arithmetic (ℕ, ·), the MSO0 finite-powerset structure |
| `autstr.algebra` | the localizations **ℤ[1/p]**, finite **Boolean algebras** |

**Classes** — one automaton for a whole family, indexed by advice:

| package | classes | signature |
|---------|---------|-----------|
| `autstr.graphs`, `autstr.tree_graphs` | bounded **tree-depth**, **pathwidth**, **tree-width**, **clique-width**, **rank-width** graphs | full MSO over vertex sets (`Sing`, `Subset`, `E`) |
| `autstr.groups`, `autstr.tree_groups` | finite **abelian** groups, **index-≤2 cyclic** groups (dihedral, quaternion, semidihedral, modular), **extraspecial** p-groups, class-2 groups of bounded **rank-width** (over F_p or ℤ/pᵈ) | group multiplication `M` |
| `autstr.cocycle_groups` | **distributed-center** class-2 groups of bounded rank-width | multiplication `M` |

Three capabilities cut across all of these:

- **Composition** (`autstr.composition`) — disjoint union and direct products of
  structures, and union and finite-product closure of classes.
- **Implicit evaluation** (`autstr.implicit`) — `check_implicit` /
  `evaluate_implicit` decide formulas and compute satisfying sets *without
  building a query automaton*, reaching members whose automata are far too large
  to construct.
- **Trees** (`autstr.tree_uniform`, `autstr.sparse_tree_automata`) — the same
  programme over finite trees read by bottom-up tree automata, the step from
  Büchi's theorem to Rabin's.

The executable notebooks in [`notebooks/`](notebooks/) work through all of it,
one per area — arithmetic & algebra, graphs, groups, composition, and implicit
evaluation.

---

## Installation

```bash
pip install autstr              # NumPy-only core — installs anywhere
pip install autstr[jax]         # + JAX-accelerated batch word processing
pip install autstr[graphs]      # + networkx conversion for the graph classes
pip install autstr[benchmarks]  # + matplotlib for the benchmark plots
```

```bash
python -c "from autstr import __version__; print(f'AutStr v{__version__}')"
```

Requires Python 3.10–3.14. The core depends only on NumPy, nltk, and graphviz.

---

## Changelog & an experiment in AI-assisted algorithm engineering

AutStr began in 2022 as a summer project — a hands-on realization of the automatic
structures its author had studied during his PhD in algorithmic model theory.
Since then, each major release has doubled as a **snapshot of what a frontier AI
coding system can do on hard, verifiable algorithmic work**, with the
mathematical direction and review kept firmly human.

<p align="center">
  <img src="docs/media/history.gif" width="720"
       alt="Gource animation of the AutStr commit history, showing the file tree growing across the human, DeepSeek and Claude phases">
</p>

- **v1.0 (2022) — human.** The original library and arithmetic front-end.
- **v1.x (July 2025) — DeepSeek.** A vibe-coding session (with extensive human
  testing and supervision) that added the sparse-DFA backend, serialization, and
  the MSO0 finite-powerset structure, and modernized packaging.
- **v2.0 (July 2026) — Claude, Anthropic's Fable 5 model.** An intensive two-day
  pair-programming session inside Claude Code that:
  - profiled and rewrote the entire automata core as batched, sparsity-aware
    NumPy — a **10²–10³× speedup** (the reference query dropped from 85 s to
    0.03 s), with linear memory;
  - migrated the library from a hard JAX dependency to a NumPy-canonical core with
    JAX as an optional accelerator;
  - built the whole uniformly-automatic layer — the generic advice machinery,
    bounded tree-depth and pathwidth graphs with MSO, finite Boolean algebras,
    finite abelian groups, the ℤ[1/p] presentations, and the non-abelian group
    classes — each verified against exhaustive or exact ground-truth oracles;
  - added the [benchmark suite](benchmarks/) and these docs.

  The ideas realized in v2 include constructions the author had sketched a decade
  earlier; several went from a whiteboard description to running, tested code
  within hours. The code is the model's; the theory, the choices, and the
  verification protocol were human.

- **v3.0 (July 2026) — Claude, (various models)** A second session, in
  the same protocol, that took the library from strings to trees and replaced the
  transition representation underneath both:
  - **tree-automatic structures.** `autstr.sparse_tree_automata` (bottom-up tree
    automata), `autstr.tree_presentations`, and `autstr.tree_uniform` — the tree
    counterparts of the whole stack. New members: **Skolem arithmetic** (ℕ, ·),
    graphs of bounded **tree-width** and bounded **clique-width** with full MSO,
    and tree-indexed **extraspecial p-groups**. Cross-validated by embedding the
    string engine's Büchi arithmetic into the tree engine and re-deciding every
    sentence through both.
  - **transitions are shared multi-terminal BDDs** over the symbol's digits, in
    both engines. `expand` became a variable renaming, `complement` stopped
    touching diagrams at all, and `minimize` became one `apply` per state per
    round. Queries that had been impossible for lack of alphabet width now
    compile: an arity-5 relation over a 14-letter alphabet (14⁵ = 537 824 flat
    symbols) went from *infeasible* to 0.2 s; tree-depth-4 bipartiteness from
    17 s to 0.4 s. The test suite went from ~2 min to ~35 s.

  - **composing presentations.** `autstr.composition`: disjoint union and
    synchronous/asynchronous direct products of automatic structures, union of
    uniformly automatic classes, and the direct-product closure of a class.
    Composed, they present every finite direct product of index-≤2 cyclic groups
    and extraspecial p-groups, drawn from either family — and decide that such a
    product is abelian exactly when all of its factors are.

- **v3.1 (July 2026) — Claude, Fable 5.** The rank-width release: new
  mathematics on top of the v3 engines, in the same human-directed protocol.
  - **class-2 groups of bounded rank-width.** `CutRankGroups` (linear layouts),
    `CutRankTreeGroups` (tree layouts) and `CocycleRankWidthGroups`
    (distributed centers, microcode advice) — the advice spells out rank-≤r
    factorizations of the commutation form's crossing blocks, cut by cut.
  - **the chain-ring extension.** Everything generalizes from F_p to
    R = ℤ/pᵈ (`autstr.chain_ring`: Smith normal form, saturated interfaces,
    the two-sided factorization lemma) — centers of exponent pᵈ, widths
    measured as module cut-rank, byte-identical at d = 1.
  - **factored advice letters.** Beyond ~20000 flat letters the cut-rank
    classes stream one ring entry per letter through accumulator states,
    making width r ≥ 2 over the ring representable (q+1 advice letters).
  - **implicit evaluation.** `check_implicit` / `evaluate_implicit` on every
    class: first-order model checking and satisfying-set computation that
    never build a query — or even a base — automaton, reaching members (ℤ/8
    and ℤ/9 words, ℤ/4 trees, the distributed-center protocol) whose
    automata are infeasible; `ImplicitClass` / `ImplicitTreeClass` are
    presentations given purely by transition functions.
  - **graphs of bounded rank-width.** `RankWidthClass` — rank decompositions
    as advice, adjacency as a bilinear form on r-bit interface vectors,
    full MSO; the graph face of the same linear algebra.
  - Docs are built by CI with all notebooks **executed during the build**
    (the repository keeps them output-free).
  - One behavior change to a 3.0 API: `show_diagram` no longer opens an
    external image viewer by default (headless builds must not spawn one);
    pass `view=True` for the old behavior.

---

## References

1. **Abu Zaid, F.** *Algorithmic Solutions via Model Theoretic Interpretations.*
   Dissertation, RWTH Aachen University, 2016.
   DOI: [10.18154/RWTH-2017-07663](https://doi.org/10.18154/RWTH-2017-07663)

2. **Abu Zaid, F.** *Uniformly Automatic Classes of Finite Structures.*
   FSTTCS 2018, LIPIcs vol. 122, pp. 10:1–10:21.
   DOI: [10.4230/LIPIcs.FSTTCS.2018.10](https://doi.org/10.4230/LIPIcs.FSTTCS.2018.10)
   *The meta-theorems for finite Boolean algebras, finite groups, and graphs of
   bounded tree-depth implemented by `autstr.uniform`, `autstr.graphs`,
   `autstr.algebra`, and `autstr.groups`.*

3. **Abu Zaid, F., Grädel, E., & Reinhardt, F.** *Advice Automatic Structures and
   Uniformly Automatic Classes.* CSL 2017, LIPIcs vol. 82, pp. 35:1–35:20.
   DOI: [10.4230/LIPIcs.CSL.2017.35](https://doi.org/10.4230/LIPIcs.CSL.2017.35)
   *Introduces automatic presentations with advice — the foundation of the uniform
   classes here; the ℤ[1/p] presentation follows its blueprint for (ℚ, +).*

4. **Blumensath, A., & Grädel, E.** *Automatic Structures.* LICS 2000, pp. 51–62.
   [Proceedings](https://lics.siglog.org/2000/Grdel-AutomaticStructures.html)

5. **Khoussainov, B., & Nerode, A.** *Automatic presentations of structures.*
   LCC 1994, LNCS vol. 960, Springer.
   DOI: [10.1007/3-540-60178-3_93](https://doi.org/10.1007/3-540-60178-3_93)

6. **Khoussainov, B., Rubin, S., & Stephan, F.** *Automatic Structures: Richness
   and Limitations.* LMCS 3(2), 2007.
   arXiv: [cs/0703064](https://arxiv.org/abs/cs/0703064) ·
   DOI: [10.2168/LMCS-3(2:2)2007](https://doi.org/10.2168/LMCS-3%282%3A2%292007)

### Foundations

The idea that a logic can be decided by translating formulas into automata long
predates the term *automatic structure*; this library is a late implementation of
a line of work that runs through:

7. **Büchi, J. R.** *Weak Second-Order Arithmetic and Finite Automata.*
   Zeitschrift für math. Logik und Grundlagen der Mathematik 6 (1960), 66–92.
   DOI: [10.1002/malq.19600060105](https://doi.org/10.1002/malq.19600060105)
   *Monadic second-order logic over (ℕ, +1) is decidable, by translation into
   finite automata. Every `evaluate` call in this library is this construction.*

8. **Rabin, M. O.** *Decidability of Second-Order Theories and Automata on
   Infinite Trees.* Transactions of the AMS 141 (1969), 1–35.
   DOI: [10.2307/1995086](https://doi.org/10.2307/1995086)
   *The same programme over trees. `autstr.sparse_tree_automata` and the
   tree-automatic presentations are the finite-tree fragment of this.*

9. **Courcelle, B.** *The Monadic Second-Order Logic of Graphs I: Recognizable
   Sets of Finite Graphs.* Information and Computation 85(1), 1990, 12–75.
   DOI: [10.1016/0890-5401(90)90043-H](https://doi.org/10.1016/0890-5401%2890%2990043-H)
   *MSO properties of graphs of bounded tree-width are decidable in linear time.
   `autstr.tree_graphs.TreeWidthClass` builds the automaton the theorem promises.*

10. **Courcelle, B., & Olariu, S.** *Upper Bounds to the Clique Width of Graphs.*
    Discrete Applied Mathematics 101 (2000), 77–114.
    DOI: [10.1016/S0166-218X(99)00184-5](https://doi.org/10.1016/S0166-218X%2899%2900184-5)
    *The k-expressions that `autstr.tree_graphs.CliqueWidthClass` reads as advice.*

11. **Makowsky, J. A.** *Algorithmic Uses of the Feferman–Vaught Theorem.*
    Annals of Pure and Applied Logic 126 (2004), 159–213.
    DOI: [10.1016/j.apal.2003.11.002](https://doi.org/10.1016/j.apal.2003.11.002)
    *The composition method behind meta-theorems of this shape.*

### Related tools

- **[MONA](https://www.brics.dk/mona/)** (Klarlund, Møller, Henriksen et al.)
  decides WS1S and WS2S by translating formulas to automata whose transitions are
  shared multi-terminal BDDs over the symbol's bits. AutStr's
  [`autstr.mtbdd`](autstr/mtbdd.py) adopts exactly that representation, for
  exactly MONA's reason: over a convolution alphabet, the flat
  `symbol -> target` table is the bottleneck.
- **[Walnut](https://cs.uwaterloo.ca/~shallit/walnut.html)** (Mousavi, Shallit)
  proves theorems about automatic sequences by deciding first-order statements
  over (ℕ, +) with automata — the same decision procedure, aimed at combinatorics
  on words rather than at presenting structures.

Both are mature and fast, and neither targets *uniformly* automatic classes or
arbitrary automatic presentations, which is where AutStr sits.
