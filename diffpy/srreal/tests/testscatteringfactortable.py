#!/usr/bin/env python

"""Unit tests for diffpy.srreal.scatteringfactortable
"""


import os
import unittest
import cPickle

from diffpy.srreal.scatteringfactortable import ScatteringFactorTable


class LocalTable(ScatteringFactorTable):
    def clone(self):  return LocalTable(self)
    def create(self): return LocalTable()
    def _standardLookup(self, smbl, q):   return q + 1
    def radiationType(self):   return "rubbish"
    def type(self):   return "localtable"
LocalTable()._registerThisType()


##############################################################################
class TestScatteringFactorTable(unittest.TestCase):

    def setUp(self):
        self.sftx = ScatteringFactorTable.createByType('X')
        self.sftn = ScatteringFactorTable.createByType('N')
        return

    def tearDown(self):
        return

    def test_pickling(self):
        """check pickling of ScatteringFactorTable instances.
        """
        self.assertEqual(0, len(self.sftx.getCustomSymbols()))
        self.sftx.setCustomAs('Na', 'Na', 123)
        self.sftx.setCustomAs('Calias', 'C')
        self.assertEqual(2, len(self.sftx.getCustomSymbols()))
        sftx1 = cPickle.loads(cPickle.dumps(self.sftx))
        self.assertEqual(2, len(sftx1.getCustomSymbols()))
        self.assertAlmostEqual(123, sftx1.lookup('Na'), 12)
        self.assertEqual(self.sftx.lookup('C'), sftx1.lookup('Calias'))
        self.assertEqual(self.sftx.type(), sftx1.type())
        return

    def test_pickling_derived(self):
        """check pickling of a derived classes.
        """
        lsft = LocalTable()
        self.assertEqual(3, lsft._standardLookup('Na', 2))
        self.assertEqual(set(), lsft.getCustomSymbols())
        lsft.foobar = 'asdf'
        lsft.setCustomAs('Na', 'Na', 123)
        self.assertEqual(1, len(lsft.getCustomSymbols()))
        lsft1 = cPickle.loads(cPickle.dumps(lsft))
        self.assertEqual(1, len(lsft1.getCustomSymbols()))
        self.assertAlmostEqual(123, lsft1.lookup('Na'), 12)
        self.assertEqual('asdf', lsft1.foobar)
        self.assertEqual(lsft.type(), lsft1.type())
        self.assertEqual(3, lsft1._standardLookup('Cl', 2))
        self.assertEqual(1, lsft1.lookup('H'))
        return

# End of class TestC

if __name__ == '__main__':
    unittest.main()

# End of file
