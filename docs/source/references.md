# References

Every construction in AutStr comes from somewhere. This page collects those
sources and says, for each, what in the library it is. The
{doc}`overview <overview>` links here wherever a named theorem or construction
is used.

## Foundations

The idea that a logic can be decided by translating formulas into automata long
predates the term *automatic structure*.

(buechi1960)=
**Büchi, J. R.** *Weak Second-Order Arithmetic and Finite Automata.*
Zeitschrift für math. Logik und Grundlagen der Mathematik 6 (1960), 66–92.
DOI: [10.1002/malq.19600060105](https://doi.org/10.1002/malq.19600060105)
: Monadic second-order logic over (ℕ, +1) is decidable, by translation into
  finite automata. **Every `evaluate` call in this library is this
  construction**, and `autstr.powerset.MSO0` is the structure that makes the
  correspondence literal: first-order logic over it *is* MSO over (ℕ, <).

(rabin1969)=
**Rabin, M. O.** *Decidability of Second-Order Theories and Automata on
Infinite Trees.* Transactions of the AMS 141 (1969), 1–35.
DOI: [10.2307/1995086](https://doi.org/10.2307/1995086)
: The same programme over trees. `autstr.sparse_tree_automata` and every
  tree-automatic presentation here are the finite-tree fragment of this.

(courcelle1990)=
**Courcelle, B.** *The Monadic Second-Order Logic of Graphs I: Recognizable
Sets of Finite Graphs.* Information and Computation 85(1), 1990, 12–75.
DOI: [10.1016/0890-5401(90)90043-H](https://doi.org/10.1016/0890-5401%2890%2990043-H)
: MSO properties of graphs of bounded tree-width are decidable in linear time.
  `autstr.tree_graphs.TreeWidthClass` builds the automaton the theorem
  promises — and, because the class is compiled once and each member is one
  linear pass, in a constructive and streaming form.

(makowsky2004)=
**Makowsky, J. A.** *Algorithmic Uses of the Feferman–Vaught Theorem.*
Annals of Pure and Applied Logic 126 (2004), 159–213.
DOI: [10.1016/j.apal.2003.11.002](https://doi.org/10.1016/j.apal.2003.11.002)
: The composition method behind meta-theorems of this shape, and behind
  `autstr.composition`.

## Automatic structures

(khoussainov1994)=
**Khoussainov, B., & Nerode, A.** *Automatic presentations of structures.*
LCC 1994, LNCS vol. 960, Springer.
DOI: [10.1007/3-540-60178-3_93](https://doi.org/10.1007/3-540-60178-3_93)
: The definition the whole library implements: a structure presented by
  automata for its domain and relations.

(blumensath2000)=
**Blumensath, A., & Grädel, E.** *Automatic Structures.* LICS 2000, pp. 51–62.
[Proceedings](https://lics.siglog.org/2000/Grdel-AutomaticStructures.html)
: Closure under first-order definability, and hence the decidability that
  `check` relies on. `autstr.interpretations` computes the closure under
  interpretations that this theory guarantees.

(khoussainov2007)=
**Khoussainov, B., Rubin, S., & Stephan, F.** *Automatic Structures: Richness
and Limitations.* LMCS 3(2), 2007.
arXiv: [cs/0703064](https://arxiv.org/abs/cs/0703064) ·
DOI: [10.2168/LMCS-3(2:2)2007](https://doi.org/10.2168/LMCS-3%282%3A2%292007)
: Where the boundaries are — which structures cannot be automatic, and why
  some of the constructions here need the tree engine.

(delhomme2004)=
**Delhommé, C.** *Automaticité des ordinaux et des graphes homogènes.*
Comptes Rendus Mathématique 339(1), 2004, 5–10.
: The two lines `autstr.ordinals` sits between: the word-automatic ordinals are
  exactly those below ω^ω, and the tree-automatic ones exactly those below
  ω^(ω^ω). `Ordinal(n)` and `TreeOrdinal(n)` take an exponent rather than being
  single structures precisely because neither boundary ordinal is reachable.

(colcombet2007)=
**Colcombet, T., & Löding, C.** *Transforming structures by set
interpretations.* LMCS 3(2), 2007.
: Existence of injective presentations for tree-automatic quotients — the
  theorem that makes `interpret(..., quotient=ε)` well posed over trees.

(kuske2011)=
**Kuske, D., & Weidner, T.** *Size and computation of injective tree automatic
presentations.* MFCS 2011, LNCS vol. 6907.
: The construction `autstr.interpretations` uses for tree quotients. No
  automatic order on trees is well-founded, so a class need have no least
  member; the representative is instead the least *description*. §4 proves the
  exponential blowup unavoidable, which is why `max_states` is worth passing.

## Uniformly automatic classes

(abuzaid2016)=
**Abu Zaid, F.** *Algorithmic Solutions via Model Theoretic Interpretations.*
Dissertation, RWTH Aachen University, 2016.
DOI: [10.18154/RWTH-2017-07663](https://doi.org/10.18154/RWTH-2017-07663)

(abuzaid2017)=
**Abu Zaid, F., Grädel, E., & Reinhardt, F.** *Advice Automatic Structures and
Uniformly Automatic Classes.* CSL 2017, LIPIcs vol. 82, pp. 35:1–35:20.
DOI: [10.4230/LIPIcs.CSL.2017.35](https://doi.org/10.4230/LIPIcs.CSL.2017.35)
: Introduces automatic presentations with advice — the foundation of
  `autstr.uniform` and every class in the library. The ℤ[1/p] presentation in
  `autstr.algebra` follows its blueprint for (ℚ, +).

(abuzaid2018)=
**Abu Zaid, F.** *Uniformly Automatic Classes of Finite Structures.*
FSTTCS 2018, LIPIcs vol. 122, pp. 10:1–10:21.
DOI: [10.4230/LIPIcs.FSTTCS.2018.10](https://doi.org/10.4230/LIPIcs.FSTTCS.2018.10)
: The meta-theorems for finite Boolean algebras, finite groups, and graphs of
  bounded tree-depth implemented by `autstr.uniform`, `autstr.graphs`,
  `autstr.algebra` and `autstr.groups`.

## Width parameters

(courcelle2000)=
**Courcelle, B., & Olariu, S.** *Upper Bounds to the Clique Width of Graphs.*
Discrete Applied Mathematics 101 (2000), 77–114.
DOI: [10.1016/S0166-218X(99)00184-5](https://doi.org/10.1016/S0166-218X%2899%2900184-5)
: The k-expressions that `autstr.tree_graphs.CliqueWidthClass` reads as advice.

(oum2006)=
**Oum, S., & Seymour, P.** *Approximating clique-width and branch-width.*
Journal of Combinatorial Theory Series B 96(4), 2006, 514–528.
: Rank-width and rank decompositions — the advice of
  `autstr.tree_graphs.RankWidthClass`, and the width measure the class-2 group
  families in `autstr.groups`, `autstr.tree_groups` and
  `autstr.cocycle_groups` are graded by.

## Higher-order pushdown graphs

(kartzow2013)=
**Kartzow, A.** *Collapsible Pushdown Graphs of Level 2 are Tree-Automatic.*
Logical Methods in Computer Science 9(1), 2013.
arXiv: [1303.2453](https://arxiv.org/abs/1303.2453)
: The encoding `autstr.collapsible` implements — blocks as a tree, collapse
  links recovered from its shape rather than stored — and the reachability
  decomposition `autstr.collapsible_reach` builds: §4 splits every run into
  four stretches, §5 turns that into a relation of the graph. Since MSO over
  these graphs is undecidable, this is the only automatic route to them.

## Related tools

**[MONA](https://www.brics.dk/mona/)** (Klarlund, Møller, Henriksen et al.)
: Decides WS1S and WS2S by translating formulas to automata whose transitions
  are shared multi-terminal BDDs over the symbol's bits.
  {py:mod}`autstr.mtbdd` adopts exactly that representation, for exactly MONA's
  reason: over a convolution alphabet, the flat `symbol -> target` table is the
  bottleneck.

**[Walnut](https://cs.uwaterloo.ca/~shallit/walnut.html)** (Mousavi, Shallit)
: Proves theorems about automatic sequences by deciding first-order statements
  over (ℕ, +) with automata — the same decision procedure, aimed at
  combinatorics on words rather than at presenting structures.

Both are mature and fast, and neither targets *uniformly* automatic classes or
arbitrary automatic presentations, which is where AutStr sits.
