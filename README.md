# AutStr
Working with infinite data structures in Python.

## Introduction
What if your algorithms could process infinite structures — like complete infinite graphs or the entire set of natural numbers — with the same ease as manipulating a data frame? AutStr provides an intuitive interface for defining and manipulating exact representations of infinite relational structures. With AutStr, infinite structures become first-class citizens in Python.

Targeted at researchers in algorithmic model theory and curious practitioners alike, AutStr offers:
- **Symbolic representation** of infinite sets and relations
- **Automata-based computation** for efficient manipulation
- **First-order logic interface** for formal queries
- **Visualization tools** for insight into infinite structures

### Installation

```bash
pip install autstr
```

#### Verify Installation
```bash
python -c "from autstr import __version__; print(f'AutStr v{__version__} installed')"
# Should output: AutStr v1.0.1 installed
```

## Getting started
The most convenient way to get started is through the arithmetic package, which supports integers.

### Defining a relation
```python
from autstr.arithmetic import VariableETerm as Var
```
The most basic building blocks are variables:
```python
x = Var('x')
y = Var('y')
```
Variables can be added to integer constants or other variables. A variable can be multiplied by a constant but not by another variable (linear arithmetic):
```python
t0 = x + y + 3
t1 = 2 * x
```
To define a relation, compare terms using built-in operations. AutStr supports $<$ and $=$ comparisons:
```python
R = t0.lt(t1)  # < (less than) Defines binary relation R: (x, y) ∈ R ⇔ x + y + 3 < 2x
```
You can work with this representation of integer solutions similarly to an explicit set:
* Test for emptiness:
```python
R.is_empty()
```
* Test for finiteness:
```python
R.is_finite()
```
* Test tuple membership:
```python
(0, 4) in R
```
* Enumerate all solutions (AutStr guarantees each solution is enumerated exactly once):
```python
for solution, _ in zip(R, range(10)):  # First 10 pairs
    print(solution)
```

### Weak divisibility
In addition to ordinal comparisons, AutStr supports the weak divisibility predicate (base 2):
- $x|y \Leftrightarrow \exists n > 0: y = 2^n \land y \text{ divides } x$

This powerful predicate enables definitions like the powers of 2:
```python
Pt = x | x  # Set of powers of 2

assert 2**10 in Pt
assert 3 not in Pt
```

### Relational algebra
Combine relations using relational algebra operators:
* **Union**: $(x, y) \in E_0 \Leftrightarrow \text{Pt}(x) \lor R(x, y)$
```python
E0 = Pt | R 
```
* **Intersection**: $(x, y) \in E_1 \Leftrightarrow \text{Pt}(x) \land R(x, y)$
```python
E1 = Pt & R
```
* **Complement**: $(x, y) \in E_2 \Leftrightarrow \neg R(x, y)$
```python
E2 = ~R
```
* **Projection (Existential Quantification)**: $y \in E_3 \Leftrightarrow \exists x . R(x, y)$
```python
E3 = R.drop(['x'])  # Or R.ex('x')
```
* **Infinite Quantification**: $y \in E_4 \Leftrightarrow \exists^{\infty} x . E_0(x, y)$
```python
E4 = E0.exinf('x') 
```

### Automatic Presentations
AutStr enables finite automata to represent infinite mathematical structures through **automatic presentations**. This powerful technique encodes:
- The domain as a regular language of strings (`automata['U']`)
- Relations as synchronous multi-tape automata recognizing valid tuples
- Enables efficient model checking for first-order queries

#### Key Features
```python
from autstr.presentations import AutomaticPresentation

# Initialize with automata for domain and relations
ap = AutomaticPresentation(automata={
    'U': universe_dfa,       # Domain automaton
    'R1': relation_dfa1,     # Relation automaton (arity k)
    'R2': relation_dfa2
})
```

1. **Automated Consistency Enforcement**
   - Automatically pads and minimizes automata
   - Ensures relations stay within domain bounds
   - Handles variable-length string encodings

2. **First-order Query Evaluation**
   ```python
   # Evaluate formula and get solution automaton
   solution_automaton = ap.evaluate("∃x.∀y.R1(x,y) ∧ R2(y)")
   
   # Check truth of closed formulas
   is_true = ap.check("∀x.∃y.R1(x,y)")
   ```

3. **Quantifier Support**
   - Existential (`∃`): `projection()`
   - Universal (`∀`): `complement(projection(complement()))`
   - Infinite quantification via automata operations

4. **Dynamic Relation Updates**
   ```python
   # Temporarily modify relations during evaluation
   result = ap.evaluate("R1(x,y)", updates={'R1': new_automaton})
   
   # Permanently update presentation
   ap.update(R1=new_automaton, R2="automaton_spec")
   ```

5. **Formal Logic Integration**
   - Seamless parsing of first-order formulas
   - Handles free variable expansion
   - Supports negation, conjunction, disjunction

#### Workflow Example
```python
# 1. Define domain automaton (e.g., base-2 integers)
universe = DFA(states=..., transitions=...)  

# 2. Create relation automata (e.g., addition)
plus_automaton = build_automaton_for("x+y=z")

# 3. Construct presentation
pres = AutomaticPresentation(automata={'U': universe, 'Plus': plus_automaton})

# 4. Query: "∃z. Plus(5,x,z) ∧ (z > 10)"
query = "∃z. Plus('5',x,z) ∧ ∃k. Plus(z,'10',k)"
solutions = pres.evaluate(query)

# 5. Enumerate solutions (x=6,7,8,...)
for sol in solutions:
    print(decode(sol))  # Output: 6, 7, 8, ...
```
### Algorithmic Design with Infinite Sets: The Sieve of Eratosthenes
AutStr enables novel algorithm design using infinite sets as first-class citizens. This implementation of the Sieve of Eratosthenes maintains the infinite candidate prime set symbolically:

```python
def infinite_sieve(steps):
    """Sieve of Eratosthenes over infinite integers"""
    candidates = (x.gt(1))  # Initial infinite set: {2,3,4,...}
    primes = []
    
    for _ in range(steps):
        # Find smallest candidate (symbolic operation)
        for p in candidates: # Elements are listed in ascending order (absolute values) 
            primes.append(p[0])
            break
        
        # Remove multiples: candidates = candidates \ {k·p | k>1}
        p = primes[-1]
        y = Var("y")
        multiples = (x.eq(p * y)).drop("y")
        candidates = candidates & ~multiples 
        
    return primes, candidates

# Execute first 3 sieving steps
primes, remainig = infinite_sieve(steps=3)
print(f"Primes found: {primes}")  # [2,3,5,7,11]
print(f"Remaining infinite set:") 
VisualDFA(remaining.evaluate()).show_diagram()
```

#### Key Algorithmic Features:
1. **Ordered Iteration**
   ```python
   for p in candidates:  # enumerates candidates in ascending order
   ```
2. **Infinite Set Operations**
   ```python
   multiples = (x.eq(p * y)).drop("y")
   candidates = candidates & ~multiples   
   ```
3. **Lazy Evaluation**
   - Relations remain symbolic until materialization
   - No explicit storage of infinite elements

#### Practical Guidelines:
1. **Prefer Deep/Narrow Formulas**
   ```python
   # Width=3 : 
   wide = "∃x.∃y.∃z.E(x,y) ∧ E(y, z)"
   
   # width=2 (easier):
   narrow = "∃x.∃y.E(x,y) ∧ (∃z.E(y,z))"
   ```
   - Depth can cause non-elementary(!) statespace explosion but scales often much better due to incremental minimization
   - Width causes exponential alphabet growth: $|\Sigma|^k$ but SparseDFAs can avoid explicit enumeration of entire alphabet in many cases.

2. **Complexity Boundaries**
   | Parameter        | Best Case       | Worst Case          |
   |------------------|-----------------|---------------------|
   | **Quantifier Depth** | Constant state space   | Non-elementary state space |
   | **Free Variables**  | constant number of exceptions  | Exponential alphabet size |

#### Theoretical Insight
While this infinite sieve beautifully demonstrates symbolic algorithm design, state complexity grows rapidly for sieved primes

> **Practical Recommendation**: Use infinite representations for conceptual modeling and verification, but switch to finite approximations with bounds for computational work. AutStr excels at proving properties about infinite structures, not processing them exhaustively.

This paradigm shift enables:
- Formal verification of infinite-state algorithms
- Symbolic exploration of hypothetical structures
- Correctness proofs for infinite data transformations

#### Theoretical Foundation
Automatic presentations leverage:
- **Regularity Preservation**: First-order operations maintain automata recognizability
- **Decidability**: First-order theories remain decidable for automatic structures

This implementation provides a complete toolkit for working with automatic structures over countable domains like ℤ, ℚ, and tree-like structures.

### Final Note
AutStr began as a summer passion project in 2022—a practical exploration of the automatic structures I'd studied theoretically during my PhD. This library represents the intersection of academic curiosity and hands-on implementation, born from a desire to make abstract model theory concepts tangible.

Released in July 2025 following a major update, the library now includes significant new features beyond its original vision. This update was developed through a vibe coding session using DeepSeek, with extensive human testing and supervision throughout the process.

Highlights of the update:
- Newly implemented `sparse_dfa` backend for efficient automata operations
- Serialization support for automata and structures
- MSO0 as finite powerset structure over natural numbers (e.g. index sets)
- Modernized packaging (`pyproject.toml`)
- Dependency version updates
- Expanded documentation

#### Performance and Maintenance
Recent updates have removed obvious performance bottlenecks through:

- JAX-accelerated automata operations
- Sparse DFA representations
- Efficient serialization, which allows storing precompiled results

However, significant optimization opportunities remain:
- Vectorization: Many sequential operations could still be parallelized
- Query Optimization: Advanced planning for first-order queries

As this is a passion project developed outside my primary research, active maintenance will be limited. That said:

- Bug reports are welcome and will be prioritized
- Performance contributions are especially appreciated
- Research collaborations involving AutStr are encouraged


## References on Automatic Structures
1. **Abu Zaid, F.**  
   *Algorithmic Solutions via Model Theoretic Interpretations.*  
   Dissertation, RWTH Aachen University, 2016.  
   DOI: [10.18154/RWTH-2017-07663](https://doi.org/10.18154/RWTH-2017-07663)  

2. **Blumensath, A., & Grädel, E.**  
   *Automatic Structures.*  
   Proceedings of the 15th Annual IEEE Symposium on Logic in Computer Science (LICS 2000).  
   Pages: 51–62.  
   URL: [LICS 2000 Proceedings](https://lics.siglog.org/2000/Grdel-AutomaticStructures.html)  

3. **Khoussainov, B., & Nerode, A.**  
   *Automatic presentations of structures.*  
   In D. Leivant (Ed.), Logic and Computational Complexity (LCC 1994). Lecture Notes in Computer Science, vol 960.  
   Springer. DOI: [10.1007/3-540-60178-3_93](https://doi.org/10.1007/3-540-60178-3_93)  

4. **Khoussainov, B., Rubin, S., & Stephan, F.**  
   *Automatic Structures: Richness and Limitations.*  
   Logical Methods in Computer Science, Volume 3, Issue 2 (2007).  
   arXiv: [cs/0703064](https://arxiv.org/abs/cs/0703064)  
   DOI: [10.2168/LMCS-3(2:2)2007](https://doi.org/10.2168/LMCS-3%282%3A2%292007)  
