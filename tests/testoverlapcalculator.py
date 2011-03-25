#!/usr/bin/env python

"""Unit tests for diffpy.srreal.overlapcalculator
"""

# version
__id__ = '$Id$'

import os
import unittest
import cPickle
import numpy

from srrealtestutils import TestCaseObjCrystOptional
from srrealtestutils import loadDiffPyStructure, loadObjCrystCrystal
from diffpy.srreal.overlapcalculator import OverlapCalculator

##############################################################################
class TestOverlapCalculator(unittest.TestCase):

    def setUp(self):
        self.olc = OverlapCalculator()
        if not hasattr(self, 'rutile'):
            type(self).rutile = loadDiffPyStructure('rutile.cif')
        if not hasattr(self, 'nickel'):
            type(self).nickel = loadDiffPyStructure('Ni.stru')
        if not hasattr(self, 'niprim'):
            type(self).niprim = loadDiffPyStructure('Ni_primitive.stru')
        return


    def tearDown(self):
        return


    def test___init__(self):
        """check OverlapCalculator.__init__()
        """
        self.assertEqual(0, self.olc.rmin)
        self.assertTrue(100 < self.olc.rmax)
        self.assertEqual(0, self.olc.rmaxused)
        self.assertEqual(0.0, self.olc.totalsquareoverlap)
        return


    def test___call__(self):
        """check BondCalculator.__call__()
        """
        olc = self.olc
        olc.rmax = 0
        self.assertEqual(0, len(olc(self.rutile)))
        olc.rmax = 2.0
        self.assertEqual(12, len(olc(self.rutile)))
        self.assertEqual(0, len(olc(self.niprim)))
        olc.rmax = 2.5
        self.assertEqual(12, len(olc(self.niprim)))
        return


    def test_pickling(self):
        '''check pickling and unpickling of OverlapCalculator.
        '''
        olc = self.olc
        olc.rmin = 0.1
        olc.rmax = 12.3
        olc.setPairMask(1, 2, False)
        olc.foobar = 'asdf'
        spkl = cPickle.dumps(olc)
        olc1 = cPickle.loads(spkl)
        self.assertFalse(olc is olc1)
        for a in olc._namesOfDoubleAttributes():
            self.assertEqual(getattr(olc, a), getattr(olc1, a))
        self.assertFalse(olc1.getPairMask(1, 2))
        self.assertTrue(olc1.getPairMask(0, 0))
        self.assertEqual('asdf', olc1.foobar)
        self.assertTrue(numpy.array_equal(
            olc.sitesquareoverlaps, olc1.sitesquareoverlaps))
        return


#   def test_distances(self):
#       """check BondCalculator.distances
#       """
#       self.olc.eval(self.nickel)
#       dst = self.olc.distances
#       self.assertTrue(numpy.array_equal(dst,
#           BondCalculator()(self.nickel)))
#       self.assertTrue(numpy.array_equal(dst, numpy.sort(dst)))
#       self.olc.maskAllPairs(False)
#       for i in range(4):
#           self.olc.setPairMask(0, i, True)
#       dst0a = self.olc(self.nickel)
#       self.olc.maskAllPairs(False)
#       for i in range(4):
#           self.olc.setPairMask(3, i, True)
#       dst3a = self.olc(self.nickel)
#       self.olc.maskAllPairs(True)
#       dstp = self.olc(self.niprim)
#       self.assertTrue(numpy.allclose(dst0a, dst3a))
#       self.assertTrue(numpy.allclose(dst0a, dstp))
#       return
#
#
#   def test_directions(self):
#       """check BondCalculator.directions
#       """
#       dst = self.olc(self.rutile)
#       drs = self.olc.directions
#       nms = numpy.sqrt(numpy.sum(numpy.power(drs, 2), axis=1))
#       self.assertTrue(numpy.allclose(dst, nms))
#       return
#
#
#   def test_sites(self):
#       """check BondCalculator.sites
#       """
#       olc = self.olc
#       dst = olc(self.rutile)
#       self.assertEqual(len(dst), len(olc.sites0))
#       self.assertEqual(len(dst), len(olc.sites1))
#       self.assertEqual(0, numpy.min(olc.sites0))
#       self.assertEqual(5, numpy.max(olc.sites0))
#       self.assertEqual(0, numpy.min(olc.sites1))
#       self.assertEqual(5, numpy.max(olc.sites1))
#       dij = [(tuple(d) + (i0, i1)) for d, i0, i1 in zip(
#                   olc.directions, olc.sites0, olc.sites1)]
#       self.assertEqual(len(dij), len(set(dij)))
#       olc.maskAllPairs(False)
#       olc(self.rutile)
#       self.assertEqual([], olc.sites0)
#       olc.setPairMask(3, 3, True)
#       olc(self.rutile)
#       self.assertTrue(len(olc.sites0))
#       self.assertEqual(set([3]), set(olc.sites0 + olc.sites1))
#       return
#
#
#   def test_types(self):
#       """check BondCalculator.types
#       """
#       olc = self.olc
#       dst = olc(self.rutile)
#       self.assertEqual(len(dst), len(olc.types0))
#       self.assertEqual(len(dst), len(olc.types1))
#       self.assertEqual(set(('Ti', 'O')), set(olc.types0))
#       self.assertEqual(set(('Ti', 'O')), set(olc.types1))
#       self.assertNotEquals(olc.types0, olc.types1)
#       olc.maskAllPairs(False)
#       olc(self.rutile)
#       self.assertEqual([], olc.types0)
#       self.assertEqual([], olc.types1)
#       olc.setPairMask(3, 3, True)
#       olc(self.rutile)
#       self.assertTrue(len(olc.types0))
#       self.assertEqual(set(['O']), set(olc.types0 + olc.types1))
#       return
#
#
#   def test_filterCone(self):
#       """check BondCalculator.filterCone()
#       """
#       olc = self.olc
#       olc.rmax = 2.5
#       self.assertEqual(12, len(olc(self.niprim)))
#       olc.filterCone([0, 0, +1], 1)
#       self.assertEqual(1, len(olc(self.niprim)))
#       olc.filterCone([0, 0, -1], 1)
#       self.assertEqual(2, len(olc(self.niprim)))
#       olc.filterOff()
#       self.assertEqual(12, len(olc(self.niprim)))
#       olc.filterCone([0, 0.1, +1], 6)
#       olc(self.niprim)
#       self.assertEqual(1, len(olc(self.niprim)))
#       olc.filterCone([0, 0, +1], 180)
#       self.assertEqual(12, len(olc(self.niprim)))
#       return
#
#
#   def test_filterOff(self):
#       """check BondCalculator.filterOff()
#       """
#       olc = self.olc
#       olc.rmax = 2.5
#       olc.filterCone([1, 2, 3], -1)
#       self.assertEqual(0, len(olc(self.niprim).tolist()))
#       olc.filterOff()
#       self.assertEqual(12, len(olc(self.niprim)))
#       return
#
#
#   def test_setPairMask(self):
#       '''check different setPairMask arguments.
#       '''
#       olc = self.olc
#       dall = olc(self.nickel)
#       olc.maskAllPairs(False)
#       self.assertEqual(0, len(olc(self.nickel)))
#       for i in range(4):
#           olc.setPairMask(0, i, True)
#       dst0a = olc(self.nickel)
#       olc.maskAllPairs(False)
#       olc.setPairMask(range(4), 0, True)
#       dst0b = olc(self.nickel)
#       self.assertTrue(numpy.array_equal(dst0a, dst0b))
#       olc.maskAllPairs(False)
#       olc.setPairMask(0, -7, True)
#       dst0c = olc(self.nickel)
#       self.assertTrue(numpy.array_equal(dst0a, dst0c))
#       olc.maskAllPairs(False)
#       olc.setPairMask(0, 'all', True)
#       dst0d = olc(self.nickel)
#       self.assertTrue(numpy.array_equal(dst0a, dst0d))
#       olc.setPairMask('all', 'all', False)
#       self.assertEqual(0, len(olc(self.nickel)))
#       olc.setPairMask('all', range(4), True)
#       dall2 = olc(self.nickel)
#       self.assertTrue(numpy.array_equal(dall, dall2))
#       self.assertRaises(ValueError, olc.setPairMask, 'fooo', 2, True)
#       self.assertRaises(ValueError, olc.setPairMask, 'aLL', 2, True)
#       return
#
## End of class TestBondCalculator
#
#
###############################################################################
#class TestBondCalculatorObjCryst(TestCaseObjCrystOptional):
#
#   def setUp(self):
#       self.olc = BondCalculator()
#       if not hasattr(self, 'rutile'):
#           type(self).rutile = loadObjCrystCrystal('rutile.cif')
#       if not hasattr(self, 'nickel'):
#           type(self).nickel = loadObjCrystCrystal('Ni.cif')
#       return
#
#
#   def tearDown(self):
#       return
#
#
#   def test___call__(self):
#       """check BondCalculator.__call__()
#       """
#       olc = self.olc
#       olc.rmax = 0
#       self.assertEqual(0, len(olc(self.rutile).tolist()))
#       olc.rmax = 2.0
#       self.assertEqual(3, len(olc(self.rutile)))
#       olc.rmax = 2.5
#       self.assertEqual(12, len(olc(self.nickel)))
#       return
#
#
#   def test_sites(self):
#       """check BondCalculator.sites
#       """
#       olc = self.olc
#       dst = olc(self.rutile)
#       self.assertEqual(len(dst), len(olc.sites0))
#       self.assertEqual(len(dst), len(olc.sites1))
#       self.assertEqual(0, numpy.min(olc.sites0))
#       self.assertEqual(1, numpy.max(olc.sites0))
#       self.assertEqual(0, numpy.min(olc.sites1))
#       self.assertEqual(1, numpy.max(olc.sites1))
#       dij = [(tuple(d) + (i0, i1)) for d, i0, i1 in zip(
#                   olc.directions, olc.sites0, olc.sites1)]
#       self.assertEqual(len(dij), len(set(dij)))
#       olc.maskAllPairs(False)
#       olc(self.rutile)
#       self.assertEqual([], olc.sites0)
#       olc.setPairMask(1, 1, True)
#       olc(self.rutile)
#       self.assertTrue(len(olc.sites0))
#       self.assertEqual(set([1]), set(olc.sites0 + olc.sites1))
#       return
#
#
#   def test_types(self):
#       """check BondCalculator.types
#       """
#       olc = self.olc
#       dst = olc(self.rutile)
#       self.assertEqual(len(dst), len(olc.types0))
#       self.assertEqual(len(dst), len(olc.types1))
#       self.assertEqual(set(('Ti', 'O')), set(olc.types0))
#       self.assertEqual(set(('Ti', 'O')), set(olc.types1))
#       self.assertNotEquals(olc.types0, olc.types1)
#       olc.maskAllPairs(False)
#       olc(self.rutile)
#       self.assertEqual([], olc.types0)
#       self.assertEqual([], olc.types1)
#       olc.setPairMask(1, 1, True)
#       olc(self.rutile)
#       self.assertTrue(len(olc.types0))
#       self.assertEqual(set(['O']), set(olc.types0 + olc.types1))
#       return
#
#
#   def test_filterCone(self):
#       """check BondCalculator.filterCone()
#       """
#       olc = self.olc
#       olc.rmax = 2.5
#       olc.filterCone([+0.5, +0.5, 0], 1)
#       self.assertEqual(1, len(olc(self.nickel)))
#       olc.filterCone([-0.5, -0.5, 0], 1)
#       self.assertEqual(2, len(olc(self.nickel)))
#       olc.filterOff()
#       self.assertEqual(12, len(olc(self.nickel)))
#       olc.filterCone([+0.5, +0.5, 0.05], 6)
#       olc(self.nickel)
#       self.assertEqual(1, len(olc(self.nickel)))
#       olc.filterCone([+0.5, +0.5, 0], 180)
#       self.assertEqual(12, len(olc(self.nickel)))
#       return
#
#
#   def test_filterOff(self):
#       """check BondCalculator.filterOff()
#       """
#       olc = self.olc
#       olc.rmax = 2.5
#       olc.filterCone([1, 2, 3], -1)
#       self.assertEqual(0, len(olc(self.nickel)))
#       olc.filterOff()
#       self.assertEqual(12, len(olc(self.nickel)))
#       return
#

if __name__ == '__main__':
    unittest.main()

# End of file
