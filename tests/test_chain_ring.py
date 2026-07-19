"""Linear algebra over the chain ring R = Z/p^d (autstr.chain_ring).

Reproduces the exhaustive validation of paper/scratch-chainring.tex: the naive
two-sided factorisation is false over R (counterexample at r=1 over Z/4), but
the saturated form holds on every admissible block. Also checks the saturation
(Smith-normal-form) primitive and that everything reduces to the field case at
d = 1.
"""
import itertools as it

import numpy as np
import pytest

from autstr import chain_ring as cr
from autstr.groups import _rref_mod


# ------------------------------------------------------------ helpers

def _mats(n, r, c):
    for f in it.product(range(n), repeat=r * c):
        yield np.array(f, dtype=np.int64).reshape(r, c)


def _colspan(WT, n):
    """{ WT @ b : b in R^r } as a set of tuples."""
    r = WT.shape[1]
    return {tuple(((WT @ np.array(b)) % n).tolist())
            for b in it.product(range(n), repeat=r)}


def _qimage(WT, V, n):
    """{ WT @ Q @ V : Q in R^{r x r} } flattened -- the naive achievable set."""
    r = WT.shape[1]
    return {tuple(((WT @ Q @ V) % n).flatten().tolist())
            for Q in _mats(n, r, r)}


def _rowsp(M, n):
    """The full row module { c @ M : c in R^r } as a set of tuples."""
    r = M.shape[0]
    return {tuple(((np.array(c) @ M) % n).tolist())
            for c in it.product(range(n), repeat=r)}


# ------------------------------------------------- scalars and digits

class TestScalars:
    @pytest.mark.parametrize("p,d", [(2, 2), (2, 3), (3, 2)])
    def test_valuation_and_units(self, p, d):
        q = p ** d
        for x in range(q):
            v = cr.valuation(x, p, d)
            assert (x % p == 0) == (not cr.is_unit(x, p, d))
            if cr.is_unit(x, p, d):
                assert (x * cr.unit_inverse(x, p, d)) % q == 1
                assert v == 0
        assert cr.valuation(0, p, d) == d

    @pytest.mark.parametrize("p,d", [(2, 3), (3, 2)])
    def test_digit_roundtrip(self, p, d):
        for x in range(p ** d):
            digs = cr.to_digits(x, p, d)
            assert len(digs) == d
            assert cr.from_digits(digs, p) == x


# -------------------------------------------------- the naive lemma fails

class TestNaiveFails:
    def test_prop_fail_witness_z4(self):
        """The explicit Proposition (prop:fail) counterexample over Z/4:
        V = W = (2,0), X = [[2,0],[0,0]] -- rows/cols in the spans, yet no
        scalar q gives X = W^T q V, because 4 = 0."""
        V = np.array([[2, 0]]); W = np.array([[2, 0]])
        X = np.array([[2, 0], [0, 0]])
        assert not cr.right_invertible(V, 2, 2)      # (2,0) is not saturated
        # rows of X in rowsp(V), cols of X in colsp(W^T)
        assert all(tuple(row.tolist()) in _rowsp(V, 4) for row in X)
        assert all(tuple(col.tolist()) in _colspan(W.T, 4) for col in X.T)
        # but X is not W^T q V for any scalar q
        assert all(not np.array_equal((W.T * q @ V) % 4, X) for q in range(4))
        # so the saturated factoriser must refuse these interfaces
        with pytest.raises(ValueError):
            cr.factor_two_sided(X, V, W, 2, 2)

    @pytest.mark.parametrize("n", [4, 8, 9, 16])
    def test_search_finds_a_counterexample(self, n):
        """Structured p-scaled + random bases hit a naive counterexample."""
        p = 2 if n % 2 == 0 else 3
        rng = np.random.default_rng(0)
        found = False
        cols = 2
        scales = [1, p] + ([p * p] if n % (p * p) == 0 else [])
        for r in (1, 2):
            bases = []
            for sc in it.product(scales, repeat=r):
                for perm in it.permutations(range(cols), min(r, cols)):
                    M = np.zeros((r, cols), dtype=np.int64)
                    for i in range(min(r, cols)):
                        M[i, perm[i]] = sc[i]
                    bases.append(M % n)
            for _ in range(10):
                bases.append(rng.integers(0, n, size=(r, cols)))
            for V in bases:
                for W in bases:
                    WT = W.T
                    cspan = _colspan(WT, n)
                    qimg = _qimage(WT, V, n)
                    for A in _mats(n, cols, r):
                        X = (A @ V) % n
                        if not all(tuple(X[:, j].tolist()) in cspan
                                   for j in range(cols)):
                            continue
                        if tuple(X.flatten().tolist()) not in qimg:
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        assert found, f"no naive counterexample located over Z/{n}"


# --------------------------------------------- the saturated lemma holds

class TestSaturatedHolds:
    def test_z4_saturation_fix(self):
        """Saturating the interface resolves the Z/4 counterexample:
        X = (1,0)^T (2) (1,0), valuation moved into Q."""
        X = np.array([[2, 0], [0, 0]])
        Vs = np.array([[1, 0]]); Ws = np.array([[1, 0]])
        assert cr.right_invertible(Vs, 2, 2) and cr.right_invertible(Ws, 2, 2)
        Q = cr.factor_two_sided(X, Vs, Ws, 2, 2)
        assert Q.tolist() == [[2]]
        assert np.array_equal((Ws.T @ Q @ Vs) % 4, X)

    @pytest.mark.parametrize("n,r", [(4, 1), (4, 2), (8, 2), (9, 2)])
    def test_no_counterexample_among_free_bases(self, n, r):
        """With right-invertible (saturated) V, W the factorisation succeeds on
        every admissible X -- Lemma (lem:ring) holds on all cases tested."""
        p = 2 if n % 2 == 0 else 3
        d = {4: 2, 8: 3, 9: 2}[n]
        cols = max(r, 2)
        rng = np.random.default_rng(r)
        free = []
        tries = 0
        # a handful of distinct saturated bases; X is then swept exhaustively
        while len(free) < 4 and tries < 2000:
            tries += 1
            M = rng.integers(0, n, size=(r, cols))
            if cr.right_invertible(M, p, d):
                free.append(M)
        assert free, "expected to sample some saturated bases"
        tested = 0
        for V in free:
            for W in free:
                WT = W.T
                cspan = _colspan(WT, n)
                for A in _mats(n, cols, r):
                    X = (A @ V) % n
                    if not all(tuple(X[:, j].tolist()) in cspan
                               for j in range(cols)):
                        continue
                    tested += 1
                    Q = cr.factor_two_sided(X, V, W, p, d)   # raises on failure
                    assert np.array_equal((WT @ Q @ V) % n, X)
        assert tested > 0


# ------------------------------------------------------ saturation (SNF)

class TestSaturate:
    @pytest.mark.parametrize("p,d", [(2, 2), (2, 3), (3, 2)])
    def test_random_saturation_properties(self, p, d):
        q = p ** d
        rng = np.random.default_rng(7)
        for _ in range(200):
            m, n = rng.integers(1, 4), rng.integers(1, 4)
            M = rng.integers(0, q, size=(m, n))
            basis, exps = cr.saturate(M, p, d)
            rho = len(basis)
            # free rank = dim_Fp(M/pM) = number of invariant factors
            assert rho == cr.module_cut_rank(M, p, d)
            assert all(0 <= e < d for e in exps)
            # the basis is a free direct summand (right-invertible)
            if rho:
                assert cr.right_invertible(basis, p, d)
            # rowsp(M) == rowsp(diag(p^e) @ basis)
            if rho:
                scaled = (np.diag([p ** e for e in exps]) @ basis) % q
            else:
                scaled = np.zeros((0, n), dtype=np.int64)
            got = _rowsp(scaled, q) if rho else {tuple([0] * n)}
            want = _rowsp(M, q)
            assert got == want

    def test_valuation_one_generator_still_counts(self):
        """(2) over Z/4 generates 2R != a summand; its module cut-rank is 1,
        though the matrix vanishes mod 2. Saturation is (1) with exponent 1."""
        M = np.array([[2]])
        basis, exps = cr.saturate(M, 2, 2)
        assert exps == [1]
        assert basis.tolist() == [[1]]
        assert cr.module_cut_rank(M, 2, 2) == 1

    @pytest.mark.parametrize("p", [2, 3])
    def test_field_case_matches_rref_rank(self, p):
        """At d = 1 the module cut-rank is the ordinary F_p rank."""
        rng = np.random.default_rng(3)
        for _ in range(100):
            m, n = rng.integers(1, 5), rng.integers(1, 5)
            M = rng.integers(0, p, size=(m, n))
            assert cr.module_cut_rank(M, p, 1) == len(_rref_mod(M % p, p)[1])


# ------------------------------------------- right inverse / invertibility

class TestRightInverse:
    @pytest.mark.parametrize("p,d", [(2, 2), (2, 3), (3, 2)])
    def test_right_inverse_reconstructs_identity(self, p, d):
        q = p ** d
        rng = np.random.default_rng(11)
        made = 0
        while made < 40:
            r, n = int(rng.integers(1, 3)), int(rng.integers(2, 4))
            if r > n:
                continue
            V = rng.integers(0, q, size=(r, n))
            if not cr.right_invertible(V, p, d):
                continue
            Y = cr.right_inverse(V, p, d)
            assert np.array_equal((V @ Y) % q, np.eye(r, dtype=np.int64))
            made += 1

    def test_non_saturated_has_no_right_inverse(self):
        V = np.array([[2, 0]])          # 2R is not a summand over Z/4
        assert not cr.right_invertible(V, 2, 2)
        with pytest.raises(ValueError):
            cr.right_inverse(V, 2, 2)
