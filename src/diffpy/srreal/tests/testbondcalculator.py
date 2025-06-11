#!/usr/bin/env python

"""Unit tests for diffpy.srreal.bondcalculator."""


import pickle
import unittest

import numpy

from diffpy.srreal.bondcalculator import BondCalculator
from diffpy.srreal.tests.testutils import (
    _msg_nopyobjcryst,
    has_pyobjcryst,
    loadDiffPyStructure,
    loadObjCrystCrystal,
    pickle_with_attr,
)

# ----------------------------------------------------------------------------


class TestBondCalculator(unittest.TestCase):

    def setUp(self):
        self.bdc = BondCalculator()
        if not hasattr(self, "rutile"):
            type(self).rutile = loadDiffPyStructure("rutile.cif")
        if not hasattr(self, "nickel"):
            type(self).nickel = loadDiffPyStructure("Ni.stru")
        if not hasattr(self, "niprim"):
            type(self).niprim = loadDiffPyStructure("Ni_primitive.stru")
        return

    def tearDown(self):
        return

    def test___init__(self):
        """Check BondCalculator.__init__()"""
        self.assertEqual(0, self.bdc.rmin)
        self.assertEqual(5, self.bdc.rmax)
        self.assertEqual(0, len(self.bdc.distances))
        bdc1 = BondCalculator(rmin=2, rmax=7)
        self.assertEqual(2, bdc1.rmin)
        self.assertEqual(7, bdc1.rmax)
        self.assertRaises(TypeError, BondCalculator, invalid=55)
        return

    def test___call__(self):
        """Check BondCalculator.__call__()"""
        bdc = self.bdc
        bdc.rmax = 0
        self.assertEqual(0, len(bdc(self.rutile)))
        bdc.rmax = 2.0
        self.assertEqual(24, len(bdc(self.rutile)))
        self.assertEqual(0, len(bdc(self.niprim)))
        bdc.rmax = 2.5
        self.assertEqual(12, len(bdc(self.niprim)))
        self.assertEqual(0, len(bdc(self.niprim, rmax=2)))
        self.assertRaises(TypeError, bdc, self.niprim, rmax=5, invalid=7)
        self.assertEqual(2, bdc.rmax)
        return

    def test_pickling(self):
        """Check pickling and unpickling of BondCalculator."""
        bdc = self.bdc
        bdc.rmin = 0.1
        bdc.rmax = 12.3
        bdc.setPairMask(1, 2, False)
        bdc(self.nickel)
        spkl = pickle.dumps(bdc)
        bdc1 = pickle.loads(spkl)
        self.assertFalse(bdc is bdc1)
        for a in bdc._namesOfDoubleAttributes():
            self.assertEqual(getattr(bdc, a), getattr(bdc1, a))
        self.assertFalse(bdc1.getPairMask(1, 2))
        self.assertTrue(bdc1.getPairMask(0, 0))
        self.assertTrue(numpy.array_equal(bdc.distances, bdc1.distances))
        self.assertRaises(RuntimeError, pickle_with_attr, bdc, foo="bar")
        return

    def test_pickling_derived_structure(self):
        """Check pickling of BondCalculator with DerivedStructureAdapter."""
        from diffpy.srreal.tests.testutils import DerivedStructureAdapter

        bdc = self.bdc
        stru0 = DerivedStructureAdapter()
        bdc.setStructure(stru0)
        self.assertEqual(1, stru0.cpqcount)
        spkl = pickle.dumps(bdc)
        bdc1 = pickle.loads(spkl)
        self.assertTrue(stru0 is bdc.getStructure())
        stru1 = bdc1.getStructure()
        self.assertTrue(type(stru1) is DerivedStructureAdapter)
        self.assertFalse(stru1 is stru0)
        self.assertEqual(1, stru1.cpqcount)
        return

    def test_distances(self):
        """Check BondCalculator.distances."""
        self.bdc.eval(self.nickel)
        dst = self.bdc.distances
        self.assertTrue(numpy.array_equal(dst, BondCalculator()(self.nickel)))
        self.assertTrue(numpy.array_equal(dst, numpy.sort(dst)))
        self.bdc.maskAllPairs(False)
        for i in range(4):
            self.bdc.setPairMask(0, i, True)
        dst0a = self.bdc(self.nickel)
        idx0a = self.bdc.sites0 == 0
        self.bdc.maskAllPairs(False)
        for i in range(4):
            self.bdc.setPairMask(3, i, True)
        dst3a = self.bdc(self.nickel)
        self.bdc.maskAllPairs(True)
        dstp = self.bdc(self.niprim)
        self.assertTrue(numpy.allclose(dst0a, dst3a))
        self.assertTrue(numpy.allclose(dst0a[idx0a], dstp))
        return

    def test_directions(self):
        """Check BondCalculator.directions."""
        dst = self.bdc(self.rutile)
        drs = self.bdc.directions
        nms = numpy.sqrt(numpy.sum(numpy.power(drs, 2), axis=1))
        self.assertTrue(numpy.allclose(dst, nms))
        return

    def test_sites(self):
        """Check BondCalculator.sites."""
        bdc = self.bdc
        dst = bdc(self.rutile)
        self.assertEqual(len(dst), len(bdc.sites0))
        self.assertEqual(len(dst), len(bdc.sites1))
        self.assertEqual(0, numpy.min(bdc.sites0))
        self.assertEqual(5, numpy.max(bdc.sites0))
        self.assertEqual(0, numpy.min(bdc.sites1))
        self.assertEqual(5, numpy.max(bdc.sites1))
        dij = [
            (tuple(d) + (i0, i1))
            for d, i0, i1 in zip(bdc.directions, bdc.sites0, bdc.sites1)
        ]
        self.assertEqual(len(dij), len(set(dij)))
        bdc.maskAllPairs(False)
        bdc(self.rutile)
        self.assertEqual(0, len(bdc.sites0))
        bdc.setPairMask(3, 3, True)
        bdc(self.rutile)
        self.assertTrue(len(bdc.sites0))
        self.assertEqual(set([3]), set(bdc.sites0).union(bdc.sites1))
        return

    def test_types(self):
        """Check BondCalculator.types."""
        bdc = self.bdc
        dst = bdc(self.rutile)
        self.assertEqual(len(dst), len(bdc.types0))
        self.assertEqual(len(dst), len(bdc.types1))
        self.assertEqual(set(("Ti", "O")), set(bdc.types0))
        self.assertEqual(set(("Ti", "O")), set(bdc.types1))
        self.assertFalse((bdc.types0 == bdc.types1).all())
        bdc.maskAllPairs(False)
        bdc(self.rutile)
        self.assertEqual(0, bdc.types0.size)
        self.assertEqual(0, bdc.types1.size)
        bdc.setPairMask(3, 3, True)
        bdc(self.rutile)
        self.assertTrue(len(bdc.types0))
        self.assertEqual(set(["O"]), set(bdc.types0).union(set(bdc.types1)))
        return

    def test_filterCone(self):
        """Check BondCalculator.filterCone()"""
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
        """Check BondCalculator.filterOff()"""
        bdc = self.bdc
        bdc.rmax = 2.5
        bdc.filterCone([1, 2, 3], -1)
        self.assertEqual(0, len(bdc(self.niprim).tolist()))
        bdc.filterOff()
        self.assertEqual(12, len(bdc(self.niprim)))
        return

    def test_setPairMask(self):
        """Check different setPairMask arguments."""
        bdc = self.bdc
        dall = bdc(self.nickel)
        bdc.maskAllPairs(False)
        self.assertEqual(0, len(bdc(self.nickel)))
        for i in range(4):
            bdc.setPairMask(0, i, True)
        dst0a = bdc(self.nickel)
        bdc.setPairMask(range(4), 0, True, others=False)
        dst0b = bdc(self.nickel)
        self.assertTrue(numpy.array_equal(dst0a, dst0b))
        bdc.maskAllPairs(False)
        bdc.setPairMask(0, -7, True)
        dst0c = bdc(self.nickel)
        self.assertTrue(numpy.array_equal(dst0a, dst0c))
        bdc.maskAllPairs(False)
        bdc.setPairMask(0, "all", True)
        dst0d = bdc(self.nickel)
        self.assertTrue(numpy.array_equal(dst0a, dst0d))
        bdc.setPairMask("all", "all", False)
        self.assertEqual(0, len(bdc(self.nickel)))
        bdc.setPairMask("all", range(4), True)
        dall2 = bdc(self.nickel)
        self.assertTrue(numpy.array_equal(dall, dall2))
        self.assertRaises(ValueError, bdc.setPairMask, "fooo", 2, True)
        self.assertRaises(ValueError, bdc.setPairMask, "aLL", 2, True)
        return

    def test_setTypeMask(self):
        """Check different setTypeMask arguments."""
        bdc = self.bdc
        dall = bdc(self.rutile)
        bdc.setTypeMask("all", "All", False)
        self.assertTrue(numpy.array_equal(dall, bdc(self.rutile)))
        bdc.setTypeMask("all", "ALL", False)
        self.assertEqual(0, len(bdc(self.rutile)))
        bdc.maskAllPairs(True)
        bdc.setTypeMask("Ti", ["O"], True, others=False)
        bdc()
        tps = set(zip(bdc.types0, bdc.types1))
        self.assertEqual(2, len(tps))
        self.assertTrue(("Ti", "O") in tps)
        self.assertTrue(("O", "Ti") in tps)
        bdc.setTypeMask(["Ti"], self.rutile.element, 0, others=1)
        bdc()
        tps = set(zip(bdc.types0, bdc.types1))
        self.assertEqual(set([("O", "O")]), tps)
        return


# End of class TestBondCalculator

# ----------------------------------------------------------------------------


@unittest.skipUnless(has_pyobjcryst, _msg_nopyobjcryst)
class TestBondCalculatorObjCryst(unittest.TestCase):

    def setUp(self):
        self.bdc = BondCalculator()
        if not hasattr(self, "rutile"):
            type(self).rutile = loadObjCrystCrystal("rutile.cif")
        if not hasattr(self, "nickel"):
            type(self).nickel = loadObjCrystCrystal("Ni.cif")
        return

    def tearDown(self):
        return

    def test___call__(self):
        """Check BondCalculator.__call__()"""
        bdc = self.bdc
        bdc.rmax = 0
        self.assertEqual(0, len(bdc(self.rutile).tolist()))
        bdc.rmax = 2.0
        self.assertEqual(6 + 3, len(bdc(self.rutile)))
        bdc.rmax = 2.5
        self.assertEqual(12, len(bdc(self.nickel)))
        return

    def test_sites(self):
        """Check BondCalculator.sites."""
        bdc = self.bdc
        dst = bdc(self.rutile)
        self.assertEqual(len(dst), len(bdc.sites0))
        self.assertEqual(len(dst), len(bdc.sites1))
        self.assertEqual(0, numpy.min(bdc.sites0))
        self.assertEqual(1, numpy.max(bdc.sites0))
        self.assertEqual(0, numpy.min(bdc.sites1))
        self.assertEqual(1, numpy.max(bdc.sites1))
        dij = [
            (tuple(d) + (i0, i1))
            for d, i0, i1 in zip(bdc.directions, bdc.sites0, bdc.sites1)
        ]
        self.assertEqual(len(dij), len(set(dij)))
        bdc.maskAllPairs(False)
        bdc(self.rutile)
        self.assertEqual(0, len(bdc.sites0))
        bdc.setPairMask(1, 1, True)
        bdc(self.rutile)
        self.assertTrue(len(bdc.sites0))
        self.assertEqual(set([1]), set(bdc.sites0).union(bdc.sites1))
        return

    def test_types(self):
        """Check BondCalculator.types."""
        bdc = self.bdc
        dst = bdc(self.rutile)
        self.assertEqual(len(dst), len(bdc.types0))
        self.assertEqual(len(dst), len(bdc.types1))
        self.assertEqual(set(("Ti", "O")), set(bdc.types0))
        self.assertEqual(set(("Ti", "O")), set(bdc.types1))
        self.assertNotEqual(bdc.types0.tolist(), bdc.types1.tolist())
        bdc.maskAllPairs(False)
        bdc(self.rutile)
        self.assertEqual(0, bdc.types0.size)
        self.assertEqual(0, bdc.types1.size)
        bdc.setPairMask(1, 1, True)
        bdc(self.rutile)
        self.assertTrue(len(bdc.types0))
        self.assertEqual(set(["O"]), set(bdc.types0).union(set(bdc.types1)))
        return

    def test_filterCone(self):
        """Check BondCalculator.filterCone()"""
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
        """Check BondCalculator.filterOff()"""
        bdc = self.bdc
        bdc.rmax = 2.5
        bdc.filterCone([1, 2, 3], -1)
        self.assertEqual(0, len(bdc(self.nickel)))
        bdc.filterOff()
        self.assertEqual(12, len(bdc(self.nickel)))
        return


# End of class TestBondCalculatorObjCryst

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()

# End of file
