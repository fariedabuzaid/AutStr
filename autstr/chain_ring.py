"""Linear algebra over the finite chain ring R = Z/p^d.

This is the algebraic foundation for the *chain-ring extension* of the
bounded-rank-width group classes: letting the center of a class-2 group have
exponent p^d turns the commutator cocycle into an R-bilinear form over
R = Z/p^d. R is a finite chain ring --
a local principal ideal ring whose ideals are the chain

    R ) pR ) p^2 R ) ... ) p^d R = 0,

every element being u * p^s with u a unit and s = v(.) its valuation.

Over a field every module has a basis and every submodule is a direct summand;
over R neither holds, which is what separates the routines here from ordinary
linear algebra. A minimal generating set of a row module may consist of
non-unit rows -- the row (2, 0) over Z/4 generates 2R, not a direct summand --
so a two-sided factorisation cannot be read off such a set directly. The
routines therefore work with a free basis of the *saturation* of a module
(``saturate``), which keeps the outer interfaces of ``factor_two_sided`` free
and confines the valuation to its middle factor Q.

At d = 1 the ring is the field F_p, a module is its own saturation, and every
routine reduces to the familiar field case, so field and ring callers share one
``saturate`` / ``factor_two_sided`` interface.

This module is a leaf (nothing in ``autstr`` is imported here) so that the group
constructions can build on it without an import cycle; the small mod-p echelon
helper ``_rref_mod_p`` below mirrors ``groups._rref_mod`` for locating unit
r x r minors.
"""
from typing import List, Tuple

import numpy as np


def _rref_mod_p(A: np.ndarray, p: int) -> Tuple[np.ndarray, List[int]]:
    """Reduced row echelon over F_p: (nonzero rows, pivot columns). A local
    copy of ``groups._rref_mod`` kept here to avoid an import cycle."""
    A = np.asarray(A, dtype=np.int64).copy() % p
    m, n = A.shape
    pivots: List[int] = []
    row = 0
    for col in range(n):
        if row == m:
            break
        sel = next((i for i in range(row, m) if A[i, col]), None)
        if sel is None:
            continue
        A[[row, sel]] = A[[sel, row]]
        A[row] = (A[row] * pow(int(A[row, col]), p - 2, p)) % p
        for i in range(m):
            if i != row and A[i, col]:
                A[i] = (A[i] - A[i, col] * A[row]) % p
        pivots.append(col)
        row += 1
    return A[:row], pivots


# ---------------------------------------------------------------- scalars

def modulus(p: int, d: int) -> int:
    """The ring size q = p^d."""
    return p ** d


def valuation(x: int, p: int, d: int) -> int:
    """The p-adic valuation v(x) in {0, .., d} of x in R; v(0) = d."""
    x = int(x) % (p ** d)
    if x == 0:
        return d
    v = 0
    while x % p == 0:
        x //= p
        v += 1
    return v


def is_unit(x: int, p: int, d: int) -> bool:
    """A unit of R is a valuation-0 element (x not divisible by p)."""
    return int(x) % p != 0


def unit_inverse(u: int, p: int, d: int) -> int:
    """The inverse of a unit u in R = Z/p^d."""
    q = p ** d
    u %= q
    if u % p == 0:
        raise ValueError(f"{u} is not a unit of Z/{q}")
    return pow(int(u), -1, q)


def to_digits(x: int, p: int, d: int) -> Tuple[int, ...]:
    """An R-element as its d base-p digits, least significant first."""
    x = int(x) % (p ** d)
    out = []
    for _ in range(d):
        out.append(x % p)
        x //= p
    return tuple(out)


def from_digits(digits, p: int) -> int:
    """Reassemble an R-element from base-p digits, least significant first."""
    x = 0
    for digit in reversed(list(digits)):
        x = x * p + int(digit) % p
    return x


# ------------------------------------------------------- r x r inverse

def inv_mod_pp(B: np.ndarray, p: int, d: int) -> np.ndarray:
    """Inverse of a square matrix that is invertible over R = Z/p^d.

    Gaussian elimination with unit pivots; a unit pivot always exists because
    an R-invertible matrix is invertible mod p. Raises if B is singular over R.
    """
    q = p ** d
    r = B.shape[0]
    if B.shape[1] != r:
        raise ValueError("inv_mod_pp expects a square matrix")
    A = np.concatenate([np.asarray(B, dtype=np.int64) % q,
                        np.eye(r, dtype=np.int64)], axis=1)
    for col in range(r):
        piv = next((i for i in range(col, r) if is_unit(A[i, col], p, d)), None)
        if piv is None:
            raise ValueError("matrix is not invertible over Z/p^d")
        if piv != col:
            A[[col, piv]] = A[[piv, col]]
        A[col] = (A[col] * unit_inverse(A[col, col], p, d)) % q
        for i in range(r):
            if i != col and A[i, col] % q != 0:
                A[i] = (A[i] - A[i, col] * A[col]) % q
    return A[:, r:] % q


# ------------------------------------------------ Smith normal form

def _col_swap(A, Winv, i, j):
    A[:, [i, j]] = A[:, [j, i]]
    Winv[[i, j], :] = Winv[[j, i], :]


def _col_scale(A, Winv, i, u, p, d):
    q = p ** d
    A[:, i] = (A[:, i] * u) % q
    Winv[i, :] = (Winv[i, :] * unit_inverse(u, p, d)) % q


def _col_addmul(A, Winv, src, dst, c, p, d):
    # column dst += c * column src  (A @ E, E = I + c e_src e_dst^T);
    # the inverse column op is the row op  row_src(Winv) -= c * row_dst(Winv).
    q = p ** d
    A[:, dst] = (A[:, dst] + c * A[:, src]) % q
    Winv[src, :] = (Winv[src, :] - c * Winv[dst, :]) % q


def smith_normal_form(M: np.ndarray, p: int, d: int) -> Tuple[List[int], np.ndarray]:
    """Diagonalise M over R = Z/p^d.

    Returns ``(exps, Winv)`` where ``exps`` are the valuations of the nonzero
    invariant factors (each < d) in order, and ``Winv`` is a unimodular n x n
    matrix whose first ``len(exps)`` rows are a free basis of the saturation of
    ``rowsp(M)``: concretely ``rowsp(M) == rowsp(diag(p^exps) @ Winv[:t])``.
    """
    q = p ** d
    A = np.asarray(M, dtype=np.int64).copy() % q
    m, n = A.shape
    Winv = np.eye(n, dtype=np.int64)
    exps: List[int] = []
    for t in range(min(m, n)):
        # pivot = a remaining entry of minimal valuation
        best = None
        for i in range(t, m):
            for j in range(t, n):
                if A[i, j] % q == 0:
                    continue
                val = valuation(A[i, j], p, d)
                if best is None or val < best[0]:
                    best = (val, i, j)
                    if val == 0:
                        break
            if best is not None and best[0] == 0:
                break
        if best is None:
            break                                    # rest of the matrix is 0
        val, bi, bj = best
        if bi != t:
            A[[t, bi]] = A[[bi, t]]
        if bj != t:
            _col_swap(A, Winv, t, bj)
        pv = p ** val
        unit = (int(A[t, t]) // pv) % q               # A[t,t] = unit * p^val
        _col_scale(A, Winv, t, unit_inverse(unit, p, d), p, d)  # pivot -> p^val
        for i in range(m):                            # clear column t (row ops)
            if i != t and A[i, t] % q != 0:
                k = int(A[i, t]) // pv
                A[i] = (A[i] - k * A[t]) % q
        for j in range(n):                            # clear row t (col ops)
            if j != t and A[t, j] % q != 0:
                k = int(A[t, j]) // pv
                _col_addmul(A, Winv, t, j, (-k) % q, p, d)
        exps.append(val)
    return exps, Winv % q


def saturate(M: np.ndarray, p: int, d: int) -> Tuple[np.ndarray, List[int]]:
    """A free basis of the saturation of ``rowsp(M)`` over R = Z/p^d.

    Returns ``(basis, exps)``: ``basis`` (rho x n) are the rows of a free direct
    summand equal to the saturation (pure closure) of the row module, and
    ``exps`` the valuations with ``rowsp(M) == rowsp(diag(p^exps) @ basis)``.
    The free rank ``rho = len(basis) = dim_{F_p}(M / pM)`` is the *module
    cut-rank* -- the number of invariant factors -- and equals the ordinary
    F_p rank when d = 1.
    """
    exps, Winv = smith_normal_form(M, p, d)
    basis = Winv[:len(exps), :].copy() % (p ** d)
    return basis, list(exps)


def module_cut_rank(M: np.ndarray, p: int, d: int) -> int:
    """The free rank rho of the saturation of ``rowsp(M)`` (module cut-rank)."""
    return len(smith_normal_form(M, p, d)[0])


# --------------------------------------------- right-invertible interfaces

def right_invertible(V: np.ndarray, p: int, d: int) -> bool:
    """True iff the rows of V are a free basis of a direct summand of R^n,
    equivalently V has full row rank mod p, equivalently some r x r minor is a
    unit. This is the "saturated interface" hypothesis required by
    ``factor_two_sided``."""
    V = np.asarray(V, dtype=np.int64)
    r = V.shape[0]
    _, pivots = _rref_mod_p(V, p)
    return len(pivots) == r


def right_inverse(V: np.ndarray, p: int, d: int) -> np.ndarray:
    """A right inverse Y (n x r) with ``V @ Y == I_r`` over R, for a
    right-invertible V (r x n). Raises if V is not right-invertible."""
    q = p ** d
    V = np.asarray(V, dtype=np.int64) % q
    r, n = V.shape
    _, pivots = _rref_mod_p(V, p)
    if len(pivots) != r:
        raise ValueError("V is not right-invertible over Z/p^d")
    sub_inv = inv_mod_pp(V[:, pivots], p, d)          # r x r, unit determinant
    Y = np.zeros((n, r), dtype=np.int64)
    for a, col in enumerate(pivots):
        Y[col, :] = sub_inv[a, :]
    return Y % q


def solve_left(V: np.ndarray, B: np.ndarray, p: int, d: int) -> np.ndarray:
    """The general ring solve ``X`` with ``X @ V == B`` over R = Z/p^d.

    ``V`` is (r x m) and may be rank-deficient (e.g. a padded basis with zero
    rows); ``B`` is (s x m); the result ``X`` is (s x r). Every row of B must
    lie in ``rowsp(V)`` -- otherwise there is no solution and a ValueError is
    raised. Free coordinates of the solution are set to 0.

    Solves ``A @ Y = C`` with ``A = V^T`` (m x r) and ``C = B^T`` by Smith-style
    diagonalisation of A: row operations are mirrored on C and column
    operations are accumulated in ``Wc`` so the solution maps back as
    ``Y = Wc @ Z``. This is the ring generalisation of the field solver
    ``autstr.groups._solve_xa_eq_b`` (X A = B) used by the linear layout
    compiler, and drives the saturated streaming update over the chain ring.
    """
    q = p ** d
    V = np.asarray(V, dtype=np.int64) % q
    B = np.asarray(B, dtype=np.int64) % q
    r, m = V.shape
    s = B.shape[0]
    if B.shape[1] != m:
        raise ValueError("solve_left: V and B must share their column count")
    A = V.T.copy() % q                          # m x r
    C = B.T.copy() % q                          # m x s
    Wc = np.eye(r, dtype=np.int64)
    exps: List[int] = []
    for t in range(min(m, r)):
        best = None
        for i in range(t, m):
            for j in range(t, r):
                if A[i, j] % q == 0:
                    continue
                val = valuation(A[i, j], p, d)
                if best is None or val < best[0]:
                    best = (val, i, j)
                    if val == 0:
                        break
            if best is not None and best[0] == 0:
                break
        if best is None:
            break
        val, bi, bj = best
        if bi != t:
            A[[t, bi]] = A[[bi, t]]
            C[[t, bi]] = C[[bi, t]]
        if bj != t:
            A[:, [t, bj]] = A[:, [bj, t]]
            Wc[:, [t, bj]] = Wc[:, [bj, t]]
        pv = p ** val
        uinv = unit_inverse((int(A[t, t]) // pv) % q, p, d)
        A[:, t] = (A[:, t] * uinv) % q           # normalise pivot to p^val
        Wc[:, t] = (Wc[:, t] * uinv) % q
        for i in range(m):                       # clear column t (row ops -> C)
            if i != t and A[i, t] % q != 0:
                k = int(A[i, t]) // pv
                A[i] = (A[i] - k * A[t]) % q
                C[i] = (C[i] - k * C[t]) % q
        for j in range(r):                       # clear row t (col ops -> Wc)
            if j != t and A[t, j] % q != 0:
                k = int(A[t, j]) // pv
                A[:, j] = (A[:, j] - k * A[:, t]) % q
                Wc[:, j] = (Wc[:, j] - k * Wc[:, t]) % q
        exps.append(val)
    Z = np.zeros((r, s), dtype=np.int64)
    for t, e in enumerate(exps):                 # p^e z = rhs, per pivot row
        pe = p ** e
        for col in range(s):
            rhs = int(C[t, col]) % q
            if rhs % pe != 0:
                raise ValueError("solve_left: no X with X @ V == B "
                                 "(a row of B is not in rowsp(V))")
            Z[t, col] = (rhs // pe) % (p ** (d - e))
    for t in range(len(exps), m):                # zero rows of A: need C == 0
        if np.any(C[t, :] % q != 0):
            raise ValueError("solve_left: no X with X @ V == B "
                             "(a row of B is not in rowsp(V))")
    return ((Wc @ Z) % q).T % q                  # X = Y^T, Y = Wc @ Z


def factor_two_sided(X: np.ndarray, Vbar: np.ndarray, Wbar: np.ndarray,
                     p: int, d: int) -> np.ndarray:
    """The two-sided factorisation over R = Z/p^d.

    Given ``X`` (m x m'), a saturated ``Vbar`` (r x m') and a saturated
    ``Wbar`` (r' x m) such that every row of X lies in ``rowsp(Vbar)`` and every
    column of X lies in ``colsp(Wbar^T)``, returns ``Q`` (r' x r) with
    ``X == Wbar^T @ Q @ Vbar``. The valuations are absorbed into Q, not the
    (free) interfaces. Raises if the containment hypotheses are violated.

    Solved as two ring linear systems (``Wbar^T Y = X`` then ``Q Vbar = Y``),
    which -- unlike a right-inverse -- tolerates zero-padded bases whose true
    rank (the module cut-rank) is below r, and reduces to the field two-step at
    d = 1. Correctness of the result is confirmed by reconstruction.
    """
    q = p ** d
    X = np.asarray(X, dtype=np.int64) % q
    Vbar = np.asarray(Vbar, dtype=np.int64) % q
    Wbar = np.asarray(Wbar, dtype=np.int64) % q
    Y = solve_left(Wbar, X.T, p, d).T                 # Wbar^T Y = X, Y = Q Vbar
    Q = solve_left(Vbar, Y, p, d)                     # Q Vbar = Y
    if not np.array_equal((Wbar.T @ Q @ Vbar) % q, X):
        raise ValueError("factorisation hypotheses violated: X is not in "
                         "Wbar^T . R^{r'xr} . Vbar (rows/cols not in the "
                         "interface spans)")
    return Q
