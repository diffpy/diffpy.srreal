#!/usr/bin/env python

"""Unit tests for diffpy.srreal.pairquantity
"""

import unittest
from diffpy.srreal.pairquantity import PairQuantity


##############################################################################
class TestPairQuantity(unittest.TestCase):

    def setUp(self):
        self.pq = PairQuantity()
        return

    def test_evaluatortype(self):
        pq = self.pq
        self.assertTrue(pq.evaluatortype in ('BASIC', 'OPTIMIZED'))
        pq.evaluatortype = 'BASIC'
        self.assertEqual('BASIC', pq.evaluatortype)
        setattr(pq, 'evaluatortype', 'OPTIMIZED')
        self.assertEqual('OPTIMIZED', pq.evaluatortype)
        self.assertRaises(ValueError, setattr, pq, 'evaluatortype', 'invalid')
        self.assertRaises(ValueError, setattr, pq, 'evaluatortype', 'basic')
        self.assertRaises(ValueError, setattr, pq, 'evaluatortype', 'BASic')
        return

# End of class TestPairQuantity

if __name__ == '__main__':
    unittest.main()

# End of file
