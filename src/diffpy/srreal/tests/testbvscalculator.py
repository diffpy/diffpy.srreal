#!/usr/bin/env python

"""Unit tests for diffpy.srreal.bvscalculator."""


import pickle
import unittest

from diffpy.srreal.bvscalculator import BVSCalculator
from diffpy.srreal.tests.testutils import loadDiffPyStructure, pickle_with_attr


##############################################################################
class TestBVSCalculator(unittest.TestCase):

    def setUp(self):
        self.bvc = BVSCalculator()
        if not hasattr(self, "rutile"):
            type(self).rutile = loadDiffPyStructure("rutile.cif")
            # rutile.cif does not have charge data, we need to add them here
            iondict = {"Ti": "Ti4+", "O": "O2-"}
            for a in self.rutile:
                a.element = iondict[a.element]
        return

    def tearDown(self):
        return

    def test___init__(self):
        """Check BVSCalculator.__init__()"""
        self.assertEqual(1e-5, self.bvc.valenceprecision)
        bvc1 = BVSCalculator(valenceprecision=1e-4)
        self.assertEqual(1e-4, bvc1.valenceprecision)
        return

    def test___call__(self):
        """Check BVSCalculator.__call__()"""
        vcalc = self.bvc(self.rutile)
        self.assertEqual(len(self.rutile), len(vcalc))
        self.assertEqual(tuple(self.bvc.value), tuple(vcalc))
        self.assertTrue(vcalc[0] > 0)
        self.assertTrue(vcalc[-1] < 0)
        self.assertAlmostEqual(0.0, sum(vcalc), 12)
        self.assertAlmostEqual(0.0, sum(self.bvc.valences), 12)
        for vo, vc in zip(self.bvc.valences, vcalc):
            self.assertTrue(abs((vo - vc) / vo) < 0.1)
        return

    def test_bvdiff(self):
        """Check BVSCalculator.bvdiff."""
        self.bvc(self.rutile)
        self.assertEqual(6, len(self.bvc.bvdiff))
        # rutile is overbonded
        for bvd in self.bvc.bvdiff:
            self.assertTrue(bvd < 0)
        return

    def test_bvmsdiff(self):
        """Check BVSCalculator.bvmsdiff."""
        self.assertEqual(0, self.bvc.bvmsdiff)
        self.bvc(self.rutile)
        self.assertAlmostEqual(0.0158969, self.bvc.bvmsdiff, 6)
        return

    def test_bvrmsdiff(self):
        """Check BVSCalculator.bvrmsdiff."""
        from math import sqrt

        self.assertEqual(0, self.bvc.bvrmsdiff)
        self.bvc(self.rutile)
        self.assertTrue(self.bvc.bvrmsdiff > 0)
        self.assertAlmostEqual(sqrt(self.bvc.bvmsdiff), self.bvc.bvrmsdiff, 12)
        bvrmsd0 = self.bvc.bvrmsdiff
        # check mixed occupancy
        rutilemix = self.rutile.copy()
        for a in self.rutile:
            rutilemix.addNewAtom(a)
        for a in rutilemix:
            a.occupancy = 0.5
        self.bvc(rutilemix)
        self.assertEqual(12, len(self.bvc.value))
        self.assertAlmostEqual(bvrmsd0, self.bvc.bvrmsdiff, 12)
        return

    def test_eval(self):
        """Check BVSCalculator.eval()"""
        vcalc = self.bvc.eval(self.rutile)
        self.assertEqual(tuple(vcalc), tuple(self.bvc.value))
        return

    def test_valences(self):
        """Check BVSCalculator.valences."""
        self.bvc(self.rutile)
        self.assertEqual((4, 4, -2, -2, -2, -2), tuple(self.bvc.valences))
        return

    def test_value(self):
        """Check BVSCalculator.value."""
        self.assertEqual(0, len(self.bvc.value))
        return

    def test_pickling(self):
        """Check pickling and unpickling of BVSCalculator."""
        bvsc = BVSCalculator()
        bvsc.rmin = 0.1
        bvsc.rmax = 12.3
        bvsc.valenceprecision = 0.3e-4
        spkl = pickle.dumps(bvsc)
        bvsc1 = pickle.loads(spkl)
        self.assertFalse(bvsc is bvsc1)
        for a in bvsc._namesOfDoubleAttributes():
            self.assertEqual(getattr(bvsc, a), getattr(bvsc1, a))
        self.assertRaises(RuntimeError, pickle_with_attr, bvsc, foo="bar")
        return

    def test_mask_pickling(self):
        """Check if mask gets properly pickled and restored."""
        self.bvc.maskAllPairs(False)
        self.bvc.setPairMask(0, 1, True)
        self.assertTrue(False is self.bvc.getPairMask(0, 0))
        self.assertTrue(True is self.bvc.getPairMask(0, 1))
        bvc1 = pickle.loads(pickle.dumps(self.bvc))
        self.assertTrue(False is bvc1.getPairMask(0, 0))
        self.assertTrue(True is bvc1.getPairMask(0, 1))
        return

    def test_table_pickling(self):
        """Check if bvparamtable gets correctly pickled and restored."""
        self.bvc.bvparamtable.setCustom("A", 1, "B", -2, 7, 8)
        bvc1 = pickle.loads(pickle.dumps(self.bvc))
        bpab = bvc1.bvparamtable.lookup("A+", "B2-")
        self.assertEqual("A", bpab.atom0)
        self.assertEqual(1, bpab.valence0)
        self.assertEqual("B", bpab.atom1)
        self.assertEqual(-2, bpab.valence1)
        self.assertEqual(7, bpab.Ro)
        self.assertEqual(8, bpab.B)
        return

    def test_pickling_derived_structure(self):
        """Check pickling of BVSCalculator with DerivedStructureAdapter."""
        from diffpy.srreal.tests.testutils import DerivedStructureAdapter

        bvc = self.bvc
        stru0 = DerivedStructureAdapter()
        bvc.setStructure(stru0)
        self.assertEqual(1, stru0.cpqcount)
        spkl = pickle.dumps(bvc)
        bvc1 = pickle.loads(spkl)
        self.assertTrue(stru0 is bvc.getStructure())
        stru1 = bvc1.getStructure()
        self.assertTrue(type(stru1) is DerivedStructureAdapter)
        self.assertFalse(stru1 is stru0)
        self.assertEqual(1, stru1.cpqcount)
        return

    def test_table_atom_valence(self):
        """Check calculation with defined valences in bvparamtable."""
        bvc = self.bvc
        barerutile = self.rutile.copy()
        for a in barerutile:
            a.element = a.element.rstrip("+-012345678")
        self.assertEqual({"Ti": 2, "O": 4}, barerutile.composition)
        self.assertFalse(any(bvc(barerutile)))
        bptb = bvc.bvparamtable
        bptb.setAtomValence("Ti", +4)
        bptb.setAtomValence("O", -2)
        vcalc = bvc(barerutile)
        self.assertEqual(4, bptb.getAtomValence("Ti"))
        self.assertEqual(-2, bptb.getAtomValence("O"))
        self.assertEqual(set((+4, -2)), set(round(x) for x in vcalc))
        self.assertEqual(set((+4, -2)), set(bvc.valences))
        bptb.resetAtomValences()
        self.assertFalse(any(bvc(barerutile)))
        return


# End of class TestBVSCalculator

if __name__ == "__main__":
    unittest.main()

# End of file
