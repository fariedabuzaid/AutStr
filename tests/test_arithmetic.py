import unittest
from autstr.arithmetic import VariableETerm as Var


class TestArithmetic(unittest.TestCase):
    def test_neg(self):
        x = Var('x')
        y = Var('y')

        expr = (-x).eq(y)

        self.assertFalse(expr.isfinite())
        self.assertFalse(expr.isempty())

        for a, b in expr:
            if a > 10:
                break

            b0 = -a
            self.assertEqual(b0, b)


    def test_add(self):
        x = Var('x')
        y = Var('y')
        z = Var('z')

        expr = (x + y).eq(z)

        self.assertFalse(expr.isfinite())
        self.assertFalse(expr.isempty())

        for a, b, c in expr:
            if c > 10:
                break

            cp = a + b
            self.assertEqual(cp, c)

    def test_mul(self):
        x = Var('x')
        y = Var('y')

        expr = (2 * x).eq(y)

        self.assertFalse(expr.isfinite())
        self.assertFalse(expr.isempty())

        for a, b in expr:
            if a > 5:
                break

            bp = 2 * a
            self.assertEqual(bp, b)

    def test_sub(self):
        x = Var('x')
        y = Var('y')
        z = Var('z')

        expr = (x - y).eq(z)

        self.assertFalse(expr.isfinite())
        self.assertFalse(expr.isempty())

        for a, b, c in expr:
            if a > 10:
                break

            c0 = a - b
            self.assertEqual(c0, c)

    def test_lt(self):
        x = Var('x')

        expr = x.lt(10)

        self.assertFalse(expr.isfinite())
        self.assertFalse(expr.isempty())

        for a, in expr:
            condition = a < 10
            self.assertTrue(condition)
            if abs(a) > 5:
                break

    def test_composite_relations(self):
        x = Var('x')
        y = Var('y')
        z = Var('z')

        constrains = (x + y).eq(z)

        self.assertFalse(constrains.isfinite())
        self.assertFalse(constrains.isempty())

        for a, b, c in constrains:
            condition = a + b == c
            self.assertTrue(condition)

            if max([a, b, c]) > 3:
                break

        constrains = constrains & z.gt(0) & z.lt(3)
        self.assertFalse(constrains.isfinite())
        self.assertFalse(constrains.isempty())

        for a, b, c in constrains:
            condition = (a + b == c) and (c > 0) and (c < 3)
            self.assertTrue(condition)

            if abs(max([a, b])) > 3:
                break

        constrains = constrains.drop(['z'])

        for a, b in constrains:
            condition = 0 < a + b < 3
            self.assertTrue(condition)

            if abs(max([a, b])) > 3:
                break

        constrains = ~constrains

        for a, b in constrains:
            condition = 0 >= a + b or a + b >= 3
            self.assertTrue(condition)

            if abs(max([a, b])) > 3:
                break



if __name__ == '__main__':
    unittest.main()
