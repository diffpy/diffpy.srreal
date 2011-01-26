#!/usr/bin/env python

"""Unit tests for diffpy.srreal.bondcecalculator
"""

# version
__id__ = '$Id$'

import os
import unittest
import cPickle
import numpy

# useful variables
thisfile = locals().get('__file__', 'file.py')
tests_dir = os.path.dirname(os.path.abspath(thisfile))
testdata_dir = os.path.join(tests_dir, 'testdata')

from srrealtestutils import TestCaseObjCrystOptional
from srrealtestutils import loadDiffPyStructure, loadObjCrystCrystal
from diffpy.srreal.bondcalculator import BondCalculator

##############################################################################
class TestBondCalculator(unittest.TestCase):

    def setUp(self):
        self.bdc = BondCalculator()
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
        """check BondCalculator.__init__()
        """
        self.assertEqual(0, self.bdc.rmin)
        self.assertEqual(5, self.bdc.rmax)
        self.assertEqual([], self.bdc.distances())
        return


    def test___call__(self):
        """check BondCalculator.__call__()
        """
        bdc = self.bdc
        bdc.rmax = 0
        self.assertEqual([], bdc(self.rutile))
        bdc.rmax = 2.0
        self.assertEqual(12, len(bdc(self.rutile)))
        self.assertEqual(0, len(bdc(self.niprim)))
        bdc.rmax = 2.5
        self.assertEqual(12, len(bdc(self.niprim)))
        return


    def test_distances(self):
        """check BondCalculator.distances()
        """
        self.bdc.eval(self.nickel)
        dst = self.bdc.distances()
        self.failUnless(numpy.array_equal(dst,
            BondCalculator()(self.nickel)))
        self.failUnless(numpy.array_equal(dst, numpy.sort(dst)))
        self.bdc.maskAllPairs(False)
        for i in range(4):
            self.bdc.setPairMask(0, i, True)
        dst0a = self.bdc(self.nickel)
        self.bdc.maskAllPairs(False)
        for i in range(4):
            self.bdc.setPairMask(3, i, True)
        dst3a = self.bdc(self.nickel)
        self.bdc.maskAllPairs(True)
        dstp = self.bdc(self.niprim)
        self.failUnless(numpy.allclose(dst0a, dst3a))
        self.failUnless(numpy.allclose(dst0a, dstp))
        return


    def test_directions(self):
        """check BondCalculator.directions()
        """
        dst = self.bdc(self.rutile)
        drs = self.bdc.directions()
        nms = numpy.sqrt(numpy.sum(numpy.power(drs, 2), axis=1))
        self.failUnless(numpy.array_equal(dst, nms))
        return


    def test_sites(self):
        """check BondCalculator.sites()
        """
        bdc = self.bdc
        dst = bdc(self.rutile)
        self.assertEqual(len(dst), len(bdc.sites0()))
        self.assertEqual(len(dst), len(bdc.sites1()))
        self.assertEqual(0, numpy.min(bdc.sites0()))
        self.assertEqual(5, numpy.max(bdc.sites0()))
        self.assertEqual(0, numpy.min(bdc.sites1()))
        self.assertEqual(5, numpy.max(bdc.sites1()))
        dij = [(tuple(d) + (i0, i1)) for d, i0, i1 in zip(
                    bdc.directions(), bdc.sites0(), bdc.sites1())]
        self.assertEqual(len(dij), len(set(dij)))
        bdc.maskAllPairs(False)
        bdc(self.rutile)
        self.assertEqual([], bdc.sites0())
        bdc.setPairMask(3, 3, True)
        bdc(self.rutile)
        self.failUnless(len(bdc.sites0()))
        self.assertEqual(set([3]), set(bdc.sites0() + bdc.sites1()))
        return


    def test_filterCone(self):
        """check BondCalculator.filterCone()
        """
        bdc = self.bdc
        bdc.rmax = 2.5
        self.assertEqual(12, len(bdc(self.niprim)))
        bdc.filterCone([0, 0, +1], 1)
        self.assertEqual(1, len(bdc(self.niprim)))
        bdc.filterCone([0, 0, -1], 1)
        self.assertEqual(2, len(bdc(self.niprim)))
        bdc.filterOff()
        self.assertEqual(12, len(bdc(self.niprim)))
        bdc.filterCone([0, 0.1, +1], 6)
        bdc(self.niprim)
        self.assertEqual(1, len(bdc(self.niprim)))
        bdc.filterCone([0, 0, +1], 180)
        self.assertEqual(12, len(bdc(self.niprim)))
        return


    def test_filterOff(self):
        """check BondCalculator.filterOff()
        """
        bdc = self.bdc
        bdc.rmax = 2.5
        bdc.filterCone([1, 2, 3], -1)
        self.assertEqual([], bdc(self.niprim))
        bdc.filterOff()
        self.assertEqual(12, len(bdc(self.niprim)))
        return


# End of class TestBondCalculator


##############################################################################
class TestBondCalculatorObjCryst(TestCaseObjCrystOptional):

    def setUp(self):
        self.bdc = BondCalculator()
        if not hasattr(self, 'rutile'):
            type(self).rutile = loadObjCrystCrystal('rutile.cif')
        if not hasattr(self, 'nickel'):
            type(self).nickel = loadObjCrystCrystal('Ni.cif')
        return


    def tearDown(self):
        return


    def test___call__(self):
        """check BondCalculator.__call__()
        """
        bdc = self.bdc
        bdc.rmax = 0
        self.assertEqual([], bdc(self.rutile))
        bdc.rmax = 2.0
        self.assertEqual(3, len(bdc(self.rutile)))
        bdc.rmax = 2.5
        self.assertEqual(12, len(bdc(self.nickel)))
        return


    def test_filterCone(self):
        """check BondCalculator.filterCone()
        """
        bdc = self.bdc
        bdc.rmax = 2.5
        bdc.filterCone([+0.5, +0.5, 0], 1)
        self.assertEqual(1, len(bdc(self.nickel)))
        bdc.filterCone([-0.5, -0.5, 0], 1)
        self.assertEqual(2, len(bdc(self.nickel)))
        bdc.filterOff()
        self.assertEqual(12, len(bdc(self.nickel)))
        bdc.filterCone([+0.5, +0.5, 0.05], 6)
        bdc(self.nickel)
        self.assertEqual(1, len(bdc(self.nickel)))
        bdc.filterCone([+0.5, +0.5, 0], 180)
        self.assertEqual(12, len(bdc(self.nickel)))
        return


    def test_filterOff(self):
        """check BondCalculator.filterOff()
        """
        bdc = self.bdc
        bdc.rmax = 2.5
        bdc.filterCone([1, 2, 3], -1)
        self.assertEqual([], bdc(self.nickel))
        bdc.filterOff()
        self.assertEqual(12, len(bdc(self.nickel)))
        return


if __name__ == '__main__':
    unittest.main()

# End of file
