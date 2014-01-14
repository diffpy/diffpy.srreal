#!/usr/bin/env python

"""Unit tests for diffpy.srreal.pairquantity
"""

import unittest
import cPickle
from diffpy.srreal.pairquantity import PairQuantity


##############################################################################
class TestPairQuantity(unittest.TestCase):

    def setUp(self):
        self.pq = PairQuantity()
        return

    def test_evaluatortype(self):
        """check PairQuantity.evaluatortype property.
        """
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


    def test_setStructure(self):
        """check PairQuantity.setStructure()
        """
        from diffpy.Structure import Structure, Atom
        from diffpy.srreal.structureadapter import EMPTY
        stru = Structure([Atom("Ar", [0.1, 0.2, 0.3])])
        self.pq.setStructure(stru)
        adpt = self.pq.getStructure()
        self.assertEqual(1, adpt.countSites())
        self.assertEqual("Ar", adpt.siteAtomType(0))
        self.pq.setStructure(EMPTY)
        adpt = self.pq.getStructure()
        self.assertEqual(0, adpt.countSites())
        return


    def test_getStructure(self):
        """check PairQuantity.getStructure()
        """
        adpt = self.pq.getStructure()
        self.assertEqual(0, adpt.countSites())
        return


    def test_pickling(self):
        '''check pickling and unpickling of PairQuantity.
        '''
        from diffpy.srreal.tests.testutils import DerivedStructureAdapter
        stru0 = DerivedStructureAdapter()
        self.pq.setStructure(stru0)
        self.assertEqual(1, stru0.cpqcount)
        spkl = cPickle.dumps(self.pq)
        pq1 = cPickle.loads(spkl)
        self.failUnless(stru0 is self.pq.getStructure())
        stru1 = pq1.getStructure()
        self.failUnless(type(stru1) is DerivedStructureAdapter)
        self.failIf(stru1 is stru0)
        self.assertEqual(1, stru1.cpqcount)
        return

# End of class TestPairQuantity

if __name__ == '__main__':
    unittest.main()

# End of file
