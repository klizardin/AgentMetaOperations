from geom.utils import radix_sorted, radix_sorted_indexes

import unittest
import random


class UtilsTestCase(unittest.TestCase):
    def test_something(self):
        a = [random.randint(0, 999) for _ in range(100)]
        sorted_a = sorted(a, key=lambda x: x)

        c = radix_sorted(a, 3, key=lambda x: x)
        b1 = [s == r for s, r in zip(sorted_a, c)]
        self.assertTrue(all(b1))

        indexes = radix_sorted_indexes(a, 3, key=lambda x: x)
        c1 = [a[i] for i in indexes]
        b = [s == r for s, r in zip(c1, sorted_a)]
        self.assertTrue(all(b))


if __name__ == '__main__':
    unittest.main()
