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

    pool = None

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
        # kill any potential multiprocessing pool
        self.pool = None
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
        """check OverlapCalculator.__call__()
        """
        olc = self.olc
        sso1 = olc(self.rutile)
        self.assertEqual(6, len(sso1))
        self.assertFalse(numpy.any(sso1))
        self.assertEqual(0.0, olc.rmaxused)
        rtb = olc.atomradiitable
        rtb.fromString('Ti:1.6, O:0.66')
        sso2 = olc(self.rutile)
        self.assertEqual(6, len(sso2[sso2 > 0]))
        self.assertEqual(3.2, olc.rmaxused)
        sso3 = olc(self.rutile, rmax=1.93)
        self.assertEqual(0.0, sum(sso3))
        self.assertEqual(1.93, olc.rmaxused)
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


    def test_parallel(self):
        """check parallel run of OverlapCalculator
        """
        import multiprocessing
        from diffpy.srreal.parallel import createParallelCalculator
        ncpu = 4
        self.pool = multiprocessing.Pool(processes=ncpu)
        olc = self.olc
        polc = createParallelCalculator(OverlapCalculator(),
                ncpu, self.pool.imap_unordered)
        olc.atomradiitable.fromString('Ti:1.6, O:0.66')
        polc.atomradiitable = olc.atomradiitable
        self.assertTrue(numpy.array_equal(olc(self.rutile), polc(self.rutile)))
        self.assertTrue(olc.totalsquareoverlap > 0.0)
        self.assertEqual(olc.totalsquareoverlap, polc.totalsquareoverlap)
        self.assertEqual(sorted(zip(olc.sites0, olc.sites1)),
                sorted(zip(polc.sites0, polc.sites1)))
        olc.atomradiitable.resetAll()
        self.assertEqual(0.0, sum(olc(self.rutile)))
        self.assertEqual(0.0, sum(polc(self.rutile)))
        return


    def test_distances(self):
        """check OverlapCalculator.distances
        """
        olc = self.olc
        olc(self.nickel)
        self.assertEqual(0, len(olc.distances))
        olc.atomradiitable.setCustom('Ni', 1.25)
        olc(self.nickel)
        self.assertEqual(4 * 12 / 2, len(olc.distances))
        dmin = numpy.sqrt(0.5) * self.nickel.lattice.a
        self.assertAlmostEqual(dmin, numpy.min(olc.distances))
        self.assertAlmostEqual(dmin, numpy.max(olc.distances))
        olc.maskAllPairs(False)
        olc.setPairMask(0, 'all', True)
        olc(self.nickel)
        self.assertEqual(12, len(olc.distances))
        return
 
 
    def test_directions(self):
        """check OverlapCalculator.directions
        """
        olc = self.olc
        olc(self.nickel)
        self.assertEqual([], olc.directions)
        olc.atomradiitable.setCustom('Ni', 1.25)
        olc.eval(self.nickel)
        drs = self.olc.directions
        nms = numpy.sqrt(numpy.sum(numpy.power(drs, 2), axis=1))
        self.assertTrue(0 < len(olc.directions))
        self.assertTrue(numpy.allclose(olc.distances, nms))
        return

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
## End of class TestOverlapCalculator
#
#
###############################################################################
#class TestOverlapCalculatorObjCryst(TestCaseObjCrystOptional):
#
#   def setUp(self):
#       self.olc = OverlapCalculator()
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
#       """check OverlapCalculator.__call__()
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
#       """check OverlapCalculator.sites
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
#       """check OverlapCalculator.types
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
#       """check OverlapCalculator.filterCone()
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
#       """check OverlapCalculator.filterOff()
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
