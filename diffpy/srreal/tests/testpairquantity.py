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


    def test_ticker(self):
        """check PairQuantity.ticker()
        """
        from diffpy.srreal.eventticker import EventTicker
        et0 = EventTicker(self.pq.ticker())
        self.pq.rmax = 3.77
        et1 = self.pq.ticker()
        self.assertNotEqual(et0, et1)
        self.failUnless(et0 < et1)
        return


    def test_ticker_override(self):
        """check Python override of PairQuantity.ticker.
        """
        pqd = PQDerived()
        self.assertEqual(0, pqd.tcnt)
        et0 = pqd.ticker()
        self.assertEqual(1, pqd.tcnt)
        et1 = PairQuantity.ticker(pqd)
        self.assertEqual(1, pqd.tcnt)
        self.assertEqual(et0, et1)
        et0.click()
        self.assertEqual(et0, et1)
        # check that implicit ticker call from PQEvaluator is
        # handled by Python override of the ticker method.
        pqd.evaluatortype = 'OPTIMIZED'
        pqd.eval()
        self.assertEqual(2, pqd.tcnt)
        return


    def test__addPairContribution(self):
        """Check Python override of PairQuantity._addPairContribution.
        """
        pqcnt = PQCounter()
        self.assertEqual(0, pqcnt(carbonzchain(0)))
        self.assertEqual(0, pqcnt(carbonzchain(1)))
        self.assertEqual(1, pqcnt(carbonzchain(2)))
        self.assertEqual(10, pqcnt(carbonzchain(5)))
        return


    def test_optimized_evaluation(self):
        """Check OPTIMIZED evaluation in Python-defined calculator class.
        """
        c8 = carbonzchain(8)
        c9 = carbonzchain(9)
        pq = PQDerived()
        pq.evaluatortype = 'OPTIMIZED'
        pq.eval(c8)
        # PQDerived does not override _stashPartialValue, therefore
        # the optimized evaluation should fail
        self.assertRaises(RuntimeError, pq.eval, c8)
        ocnt = PQCounter()
        ocnt.evaluatortype = 'OPTIMIZED'
        self.assertEqual(28, ocnt(c8))
        self.assertEqual(28, ocnt(c8))
        self.assertEqual('OPTIMIZED', ocnt.evaluatortypeused)
        self.assertEqual(36, ocnt(c9))
        self.assertEqual('OPTIMIZED', ocnt.evaluatortypeused)
        self.assertEqual(28, ocnt(c8))
        self.assertEqual('OPTIMIZED', ocnt.evaluatortypeused)
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

# helper for testing PairQuantity overrides

class PQDerived(PairQuantity):

    tcnt = 0

    def ticker(self):
        self.tcnt += 1
        return PairQuantity.ticker(self)

# End of class PQDerived

# helper for testing support for optimized evaluation

class PQCounter(PairQuantity):

    def __init__(self):
        super(PQCounter, self).__init__()
        self._resizeValue(1)
        self.rmax = 10
        return

    def __call__(self, structure=None):
        rv, = self.eval(structure)
        return rv

    def _addPairContribution(self, bnds, sumscale):
        self._value[0] += 0.5 * sumscale
        return

    def _stashPartialValue(self):
        self.__stashed_value = self._value[0]
        return

    def _restorePartialValue(self):
        self._value[0] = self.__stashed_value
        del self.__stashed_value
        return

# End of class PQCounter

def carbonzchain(n):
    "Helper function that returns a z-chain of Carbon atoms."
    from diffpy.Structure import Structure, Atom
    rv = Structure([Atom('C', [0, 0, z]) for z in range(n)])
    return rv


if __name__ == '__main__':
    unittest.main()

# End of file
