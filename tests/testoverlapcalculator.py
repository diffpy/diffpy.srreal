#!/usr/bin/env python

"""Unit tests for diffpy.srreal.overlapcalculator
"""

# version
__id__ = '$Id$'

import os
import unittest
import cPickle
import copy
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

    def test_gradients(self):
        """check OverlapCalculator.gradients
        """
        olc = self.olc
        olc.atomradiitable.fromString('Ti:1.6, O:0.66')
        olc(self.rutile)
        self.assertEqual(6, len(olc.gradients))
        self.assertTrue(numpy.allclose([0, 0, 0], numpy.sum(olc.gradients)))
        g2 = olc.gradients[2]
        self.assertTrue(abs(g2[0]) > 0.1)
        tso0 = olc.totalsquareoverlap
        dx = 1e-8
        rutile2 = loadDiffPyStructure('rutile.cif')
        rutile2[2].xyz_cartn += [dx, 0.0, 0.0]
        olc.eval(rutile2)
        g2nx = (olc.totalsquareoverlap - tso0) / dx
        self.assertAlmostEqual(g2[0], g2nx, 6)
        return

    def test_sitesquareoverlaps(self):
        """check OverlapCalculator.sitesquareoverlaps
        """
        olc = self.olc
        self.assertTrue(numpy.array_equal([], olc.sitesquareoverlaps))
        olc(self.rutile)
        self.assertTrue(numpy.array_equal(6 * [0.0], olc.sitesquareoverlaps))
        olc.atomradiitable.fromString('Ti:1.6, O:0.66')
        sso = olc(self.rutile)
        self.assertTrue(numpy.array_equal(sso, olc.sitesquareoverlaps))
        self.assertTrue(numpy.all(sso))
        return

    def test_totalsquareoverlap(self):
        """check OverlapCalculator.totalsquareoverlap
        """
        olc = self.olc
        self.assertEqual(0.0, olc.totalsquareoverlap)
        olc(self.rutile)
        self.assertEqual(0.0, olc.totalsquareoverlap)
        olc.atomradiitable.fromString('Ti:1.6, O:0.66')
        olc(self.rutile)
        self.assertTrue(1.20854162728, olc.totalsquareoverlap)
        return

    def test_meansquareoverlap(self):
        """check OverlapCalculator.meansquareoverlap
        """
        olc = self.olc
        self.assertEqual(0.0, olc.meansquareoverlap)
        olc(self.nickel)
        self.assertEqual(0.0, olc.meansquareoverlap)
        olc.atomradiitable.setCustom('Ni', 1.25)
        olc(self.nickel)
        mso0 = olc.meansquareoverlap
        self.assertTrue(mso0 > 0.0)
        sso1 = olc(self.niprim)
        self.assertEqual(1, len(sso1))
        self.assertAlmostEqual(mso0, olc.meansquareoverlap)
        olc.atomradiitable.fromString('Ti:1.6, O:0.66')
        olc(self.rutile)
        self.assertAlmostEqual(0.201423604547, olc.meansquareoverlap)
        return

    def test_flipDiffTotal(self):
        """check OverlapCalculator.flipDiffTotal
        """
        olc = self.olc
        olc.atomradiitable.fromString('Ti:1.6, O:0.66')
        olc(self.rutile)
        self.assertEqual(0.0, olc.flipDiffTotal(0, 0))
        self.assertEqual(0.0, olc.flipDiffTotal(0, 1))
        self.assertEqual(0.0, olc.flipDiffTotal(2, 5))
        tso0 = olc.totalsquareoverlap
        olc2 = copy.copy(olc)
        rutile2 = loadDiffPyStructure('rutile.cif')
        rutile2[0].element = 'O'
        rutile2[2].element = 'Ti'
        olc2(rutile2)
        fdt02 = olc2.totalsquareoverlap - tso0
        self.assertTrue(fdt02 > 0.01)
        self.assertAlmostEqual(fdt02, olc.flipDiffTotal(0, 2))
        return

# End of class TestOverlapCalculator

###############################################################################
class TestOverlapCalculatorObjCryst(TestCaseObjCrystOptional):

    def setUp(self):
        self.olc = OverlapCalculator()
        if not hasattr(self, 'rutile'):
            type(self).rutile = loadObjCrystCrystal('rutile.cif')
        if not hasattr(self, 'nickel'):
            type(self).nickel = loadObjCrystCrystal('Ni.cif')
        return

    def tearDown(self):
        return

    def test_totalsquareoverlap(self):
        """check OverlapCalculator.totalsquareoverlap for ObjCryst crystal
        """
        olc = self.olc
        self.assertEqual(0.0, olc.totalsquareoverlap)
        olc(self.rutile)
        self.assertEqual(0.0, olc.totalsquareoverlap)
        olc.atomradiitable.fromString('Ti:1.6, O:0.66')
        olc(self.rutile)
        self.assertTrue(1.20854162728, olc.totalsquareoverlap)
        return

    def test_meansquareoverlap(self):
        """check OverlapCalculator.meansquareoverlap for ObjCryst crystal
        """
        olc = self.olc
        self.assertEqual(0.0, olc.meansquareoverlap)
        olc(self.rutile)
        self.assertEqual(0.0, olc.meansquareoverlap)
        olc.atomradiitable.fromString('Ti:1.6, O:0.66')
        olc(self.rutile)
        self.assertAlmostEqual(0.201423604547, olc.meansquareoverlap)
        return

    def test_flipDiffTotal(self):
        """check OverlapCalculator.flipDiffTotal for an ObjCryst crystal
        """
        olc = self.olc
        olc(self.rutile)
        self.assertEqual(0.0, olc.flipDiffTotal(0, 1))
        olc.atomradiitable.fromString('Ti:1.6, O:0.66')
        olc(self.rutile)
        tso0 = olc.totalsquareoverlap
        olc2 = copy.copy(olc)
        olc2.atomradiitable.fromString('Ti:0.66, O:1.6')
        olc2(self.rutile)
        fdt01 = olc2.totalsquareoverlap - tso0
        self.assertAlmostEqual(fdt01, olc.flipDiffTotal(0, 1))
        return

    def test_flipDiffMean(self):
        """check OverlapCalculator.flipDiffMean for an ObjCryst crystal
        """
        olc = self.olc
        olc(self.rutile)
        self.assertEqual(0.0, olc.flipDiffMean(0, 1))
        olc.atomradiitable.fromString('Ti:1.6, O:0.66')
        olc(self.rutile)
        mso0 = olc.meansquareoverlap
        olc2 = copy.copy(olc)
        olc2.atomradiitable.fromString('Ti:0.66, O:1.6')
        olc2(self.rutile)
        fdm01 = olc2.meansquareoverlap - mso0
        self.assertAlmostEqual(fdm01, olc.flipDiffMean(0, 1))
        self.assertAlmostEqual(fdm01, olc.flipDiffTotal(0, 1) / 6)
        return

# End of class TestOverlapCalculatorObjCryst

if __name__ == '__main__':
    unittest.main()

# End of file
