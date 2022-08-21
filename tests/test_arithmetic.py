from autstr.arithmetic import VariableETerm as Var

class TestArithmetic:
    def test_neg(self):
        x = Var('x')
        y = Var('y')

        expr = (-x).eq(y)

        assert not (expr.isfinite())
        assert not (expr.isempty())

        for a, b in expr:
            if a > 10:
                break

            b0 = -a
            assert b0 == b

    def test_add(self):
        x = Var('x')
        y = Var('y')
        z = Var('z')

        expr = (x + y).eq(z)

        assert not (expr.isfinite())
        assert not (expr.isempty())

        for a, b, c in expr:
            if c > 10:
                break

            cp = a + b
            assert (cp == c)

    def test_mul(self):
        x = Var('x')
        y = Var('y')

        expr = (2 * x).eq(y)

        assert not (expr.isfinite())
        assert not (expr.isempty())

        for a, b in expr:
            if a > 5:
                break

            bp = 2 * a
            assert (bp == b)

    def test_sub(self):
        x = Var('x')
        y = Var('y')
        z = Var('z')

        expr = (x - y).eq(z)

        assert not (expr.isfinite())
        assert not (expr.isempty())

        for a, b, c in expr:
            if a > 10:
                break

            c0 = a - b
            assert (c0 == c)

    def test_lt(self):
        x = Var('x')

        expr = x.lt(10)

        assert not (expr.isfinite())
        assert not (expr.isempty())

        for a, in expr:
            condition = a < 10
            assert (condition)
            if abs(a) > 5:
                break

    def test_composite_relations(self):
        x = Var('x')
        y = Var('y')
        z = Var('z')

        constrains = (x + y).eq(z)

        assert not (constrains.isfinite())
        assert not (constrains.isempty())

        for a, b, c in constrains:
            condition = a + b == c
            assert (condition)

            if max([a, b, c]) > 3:
                break

        constrains = constrains & z.gt(0) & z.lt(3)
        assert not (constrains.isfinite())
        assert not (constrains.isempty())

        for a, b, c in constrains:
            condition = (a + b == c) and (c > 0) and (c < 3)
            assert (condition)

            if abs(max([a, b])) > 3:
                break

        constrains = constrains.drop(['z'])

        for a, b in constrains:
            condition = 0 < a + b < 3
            assert (condition)

            if abs(max([a, b])) > 3:
                break

        constrains = ~constrains

        for a, b in constrains:
            condition = 0 >= a + b or a + b >= 3
            assert (condition)

            if abs(max([a, b])) > 3:
                break

    def test_exinf(self):
        x = Var('x')
        y = Var('y')

        expr = x.eq(y).exinf('y')
        assert (expr.isempty())

        expr = x.lt(y).exinf('y')
        assert not (expr.isempty())
        assert not (expr.isfinite())

    def test_contains(self):
        x = Var('x')
        y = Var('y')
        z = Var('z')

        expr = (x + y).eq(z)
        assert ((1, 1, 2) in expr)
        assert not ((10, 2, 13) in expr)

