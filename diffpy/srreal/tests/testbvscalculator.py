#!/usr/bin/env python

"""Unit tests for diffpy.srreal.bvscalculator
"""


import unittest
import cPickle

from diffpy.srreal.bvscalculator import BVSCalculator
from diffpy.Structure import Structure
from diffpy.srreal.tests.testutils import loadDiffPyStructure

##############################################################################
class TestBVSCalculator(unittest.TestCase):

    def setUp(self):
        self.bvc = BVSCalculator()
        if not hasattr(self, 'rutile'):
            type(self).rutile = loadDiffPyStructure('rutile.cif')
            # rutile.cif does not have charge data, we need to add them here
            iondict = {'Ti' : 'Ti4+',  'O' : 'O2-'}
            for a in self.rutile:  a.element = iondict[a.element]
        return


    def tearDown(self):
        return


    def test___init__(self):
        """check BVSCalculator.__init__()
        """
        self.assertEqual(1e-5, self.bvc.valenceprecision)
        bvc1 = BVSCalculator(valenceprecision=1e-4)
        self.assertEqual(1e-4, bvc1.valenceprecision)
        return


    def test___call__(self):
        """check BVSCalculator.__call__()
        """
        vcalc = self.bvc(self.rutile)
        self.assertEqual(len(self.rutile), len(vcalc))
        self.assertEqual(tuple(self.bvc.value), tuple(vcalc))
        self.failUnless(vcalc[0] > 0)
        self.failUnless(vcalc[-1] < 0)
        self.assertAlmostEqual(0.0, sum(vcalc), 12)
        self.assertAlmostEqual(0.0, sum(self.bvc.valences), 12)
        for vo, vc in zip(self.bvc.valences, vcalc):
            self.failUnless(abs((vo - vc) / vo) < 0.1)
        return


    def test_bvdiff(self):
        """check BVSCalculator.bvdiff
        """
        self.bvc(self.rutile)
        self.assertEqual(6, len(self.bvc.bvdiff))
        # rutile is overbonded
        for bvd in self.bvc.bvdiff:
            self.failUnless(bvd < 0)
        return


    def test_bvmsdiff(self):
        """check BVSCalculator.bvmsdiff
        """
        self.assertEqual(0, self.bvc.bvmsdiff)
        self.bvc(self.rutile)
        self.assertAlmostEqual(0.0158969, self.bvc.bvmsdiff, 6)
        return


    def test_bvrmsdiff(self):
        """check BVSCalculator.bvrmsdiff
        """
        from math import sqrt
        self.assertEqual(0, self.bvc.bvrmsdiff)
        self.bvc(self.rutile)
        self.failUnless(self.bvc.bvrmsdiff > 0)
        self.assertAlmostEqual(sqrt(self.bvc.bvmsdiff),
                self.bvc.bvrmsdiff, 12)
        bvrmsd0 = self.bvc.bvrmsdiff
        # check mixed occupancy
        rutilemix = Structure(self.rutile)
        for a in self.rutile:
            rutilemix.addNewAtom(a)
        for a in rutilemix:
            a.occupancy = 0.5
        self.bvc(rutilemix)
        self.assertEqual(12, len(self.bvc.value))
        self.assertAlmostEqual(bvrmsd0, self.bvc.bvrmsdiff, 12)
        return


    def test_eval(self):
        """check BVSCalculator.eval()
        """
        vcalc = self.bvc.eval(self.rutile)
        self.assertEqual(tuple(vcalc), tuple(self.bvc.value))
        return


    def test_valences(self):
        """check BVSCalculator.valences
        """
        self.bvc(self.rutile)
        self.assertEqual((4, 4, -2, -2, -2, -2),
                tuple(self.bvc.valences))
        return


    def test_value(self):
        """check BVSCalculator.value
        """
        self.assertEqual(0, len(self.bvc.value))
        return


    def test_pickling(self):
        '''check pickling and unpickling of BVSCalculator.
        '''
        bvsc = BVSCalculator()
        bvsc.rmin = 0.1
        bvsc.rmax = 12.3
        bvsc.valenceprecision = 0.3e-4
        bvsc.foobar = 'asdf'
        spkl = cPickle.dumps(bvsc)
        bvsc1 = cPickle.loads(spkl)
        self.failIf(bvsc is bvsc1)
        for a in bvsc._namesOfDoubleAttributes():
            self.assertEqual(getattr(bvsc, a), getattr(bvsc1, a))
        self.assertEqual('asdf', bvsc1.foobar)
        return


    def test_mask_pickling(self):
        '''Check if mask gets properly pickled and restored.
        '''
        self.bvc.maskAllPairs(False)
        self.bvc.setPairMask(0, 1, True)
        self.failUnless(False is self.bvc.getPairMask(0, 0))
        self.failUnless(True is self.bvc.getPairMask(0, 1))
        bvc1 = cPickle.loads(cPickle.dumps(self.bvc))
        self.failUnless(False is bvc1.getPairMask(0, 0))
        self.failUnless(True is bvc1.getPairMask(0, 1))
        return


    def test_table_pickling(self):
        '''Check if bvparamtable gets correctly pickled and restored.
        '''
        self.bvc.bvparamtable.setCustom('A', 1, 'B', -2, 7, 8)
        bvc1 = cPickle.loads(cPickle.dumps(self.bvc))
        bpab = bvc1.bvparamtable.lookup('A+', 'B2-')
        self.assertEqual("A", bpab.atom0)
        self.assertEqual(1, bpab.valence0)
        self.assertEqual("B", bpab.atom1)
        self.assertEqual(-2, bpab.valence1)
        self.assertEqual(7, bpab.Ro)
        self.assertEqual(8, bpab.B)
        return


    def test_pickling_derived_structure(self):
        '''check pickling of BVSCalculator with DerivedStructureAdapter.
        '''
        from diffpy.srreal.tests.testutils import DerivedStructureAdapter
        bvc = self.bvc
        stru0 = DerivedStructureAdapter()
        bvc.setStructure(stru0)
        self.assertEqual(1, stru0.cpqcount)
        spkl = cPickle.dumps(bvc)
        bvc1 = cPickle.loads(spkl)
        self.failUnless(stru0 is bvc.getStructure())
        stru1 = bvc1.getStructure()
        self.failUnless(type(stru1) is DerivedStructureAdapter)
        self.failIf(stru1 is stru0)
        self.assertEqual(1, stru1.cpqcount)
        return


# End of class TestBVSCalculator

if __name__ == '__main__':
    unittest.main()

# End of file
