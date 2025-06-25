# AutStr
Working with infinite data structures in Python.

## Introduction
Have you ever wondered what would happen if you could input an infinite structure (e.g., an infinite graph) into your algorithm instead of a finite one? With AutStr, you can do exactly that for certain infinite structures. AutStr provides an easy-to-use interface for defining relational structures over predefined infinite base structures. Currently, AutStr offers built-in support for a robust extension of linear integer arithmetic, while additional base structures over arbitrary countable domains can be defined via the low-level API.

### Installation with `uv` 
Here's how to install AutStr using `uv`, the high-performance Python package installer:

#### Install with `uv`
```bash
# Install uv globally (if not already installed)
pip install uv

# Create and activate virtual environment
uv venv .venv
source .venv/bin/activate  # Linux/macOS
.\.venv\Scripts\activate  # Windows

# Install from GitHub
uv pip install "git+https://github.com/fariedabuzaid/AutStr.git"

# For development (editable mode)
uv pip install -e "git+https://github.com/fariedabuzaid/AutStr.git#egg=autstr"

# Install documentation extras
uv pip install "autstr[docs] @ git+https://github.com/fariedabuzaid/AutStr.git"
```

#### Alternative: Install from Local Clone
```bash
git clone https://github.com/fariedabuzaid/AutStr.git
cd AutStr

# Install core package
uv pip install .

# Install with docs dependencies
uv pip install ".[docs]"
```

#### Verify Installation
```bash
python -c "from autstr import __version__; print(f'AutStr v{__version__} installed')"
# Should output: AutStr v0.1 installed
```

### Troubleshooting
If you encounter issues:
```bash
# Clean installation
uv pip install --reinstall --no-cache autstr

# Force rebuild from source
uv pip install --force-reinstall --no-binary :all: autstr

# Check environment consistency
uv pip check
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
R = t0 < t1  # Defines binary relation R: (x, y) ∈ R ⇔ x + y + 3 < 2x
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
from autstr.buildin.automata import AutomaticPresentation

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
from autstr.arithmetic import VariableETerm as Var
x = Var('x')

def infinite_sieve(steps):
    """Sieve of Eratosthenes over infinite integers"""
    candidates = (x >= 2)  # Initial infinite set: {2,3,4,...}
    primes = []
    
    for _ in range(steps):
        # Find smallest candidate (symbolic operation)
        p = candidates.min_element()
        primes.append(p)
        
        # Remove multiples: candidates = candidates \ {k·p | k>1}
        multiples = (x > p) & (x % p == 0)
        candidates = candidates - multiples  # Set difference
        
    return primes, candidates

# Execute first 5 sieving steps
primes, remaining = infinite_sieve(steps=5)
print(f"Primes found: {primes}")  # [2,3,5,7,11]
print(f"Remaining infinite set: {remaining.automaton}")
```

#### Key Algorithmic Features:
1. **Symbolic Minimum Extraction**
   ```python
   p = candidates.min_element()  # Computes smallest element without enumeration
   ```
2. **Infinite Set Operations**
   ```python
   multiples = (x > p) & (x % p == 0)  # Defines infinite composite set
   candidates = candidates - multiples  # Exact set difference
   ```
3. **Lazy Evaluation**
   - Relations remain symbolic until materialization
   - No explicit storage of infinite elements

#### Practical Guidelines:
1. **Prefer Deep/Narrow Formulas**
   ```python
   # Width=3 (hard): 
   wide = "∃x.∃y.∃z.Prime(x)∧Prime(y)∧Twin(x,y,z)"
   
   # Depth=3 (easier):
   narrow = "∃x.Prime(x) ∧ (∃y.Prime(y) ∧ (∃z.Twin(x,y,z))"
   ```
   - Depth scales often better due to incremental minimization
   - Width causes exponential alphabet growth: $|\Sigma|^k$

2. **Complexity Boundaries**
   | Parameter        | Best Case       | Worst Case          |
   |------------------|-----------------|---------------------|
   | **Quantifier Depth** | Constant    | Non-elementary      |
   | **Free Variables**  | Exponential alphabet size  | Exponential alphabet size |
   
4. **Optimization Strategies**
   - **Minimize aggressively**:
     ```python
     candidates = (candidates - multiples).minify()  # Force state reduction
     ```
   - **Avoid high-arity**:
     ```python
     # Decompose wide relations:
     R1 = project(relation, ['x','y'])
     R2 = project(relation, ['z'])
     ```
   - **Use bounded quantification**:
     ```python
     # Instead of ∀x.φ(x), use:
     bounded = ∀(x, x<1000).φ(x)  # Finite domain restriction
     ```

#### Theoretical Insight
While this infinite sieve beautifully demonstrates symbolic algorithm design:
1. State complexity grows as $\prod p_i$ for sieved primes
2. Sieving primes up to $p_k$ requires $O(e^{\theta(k)})$ states
3. First 10 primes would need > 6.5 billion states

> **Practical Recommendation**: Use infinite representations for conceptual modeling and verification, but switch to finite approximations with bounds for computational work. AutStr excels at proving properties about infinite structures, not processing them exhaustively.

This paradigm shift enables:
- Formal verification of infinite-state algorithms
- Symbolic exploration of hypothetical structures
- Correctness proofs for infinite data transformations

#### Theoretical Foundation
Automatic presentations leverage:
- **Regularity Preservation**: First-order operations maintain automata recognizability
- **Decidability**: First-order theories remain decidable for automatic structures
- **Efficient Model Checking**: Automata operations run in polynomial time relative to automaton size

This implementation provides a complete toolkit for working with automatic structures over countable domains like ℤ, ℚ, and tree-like structures.

### Final Note
AutStr began as a summer passion project in 2022—a practical exploration of the automatic structures I'd studied theoretically during my PhD. This library represents the intersection of academic curiosity and hands-on implementation, born from a desire to make abstract model theory concepts tangible.

Released in June 2025 as-is, the library remains fundamentally unchanged from its original vision except for:
- Modernized packaging (`pyproject.toml`)
- Dependency version updates
- Expanded documentation

While not actively maintained, AutStr stands as:
1. A functional implementation of basic automatic structure theory
2. A testament to the elegance of infinite-state computation
3. An invitation to explore algorithmic model theory hands-on

> "Some things are worth building not because they scale, but because they reveal."  
> — Faried Abu Zaid, June 2025

For researchers and enthusiasts: May this implementation spark new insights into the beautiful complexity of infinite structures. For practical applications, consider pairing with finite approximations or domain-specific abstractions.
