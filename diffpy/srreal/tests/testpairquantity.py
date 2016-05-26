#!/usr/bin/env python

"""Unit tests for diffpy.srreal.pairquantity
"""

import unittest
import cPickle
from diffpy.srreal.pairquantity import PairQuantity
from diffpy.srreal.pdfcalculator import PDFCalculator


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
        self.assertRaises(ValueError, setattr, pq, 'evaluatortype', 'invalid')
        self.assertRaises(ValueError, setattr, pq, 'evaluatortype', 'basic')
        self.assertRaises(ValueError, setattr, pq, 'evaluatortype', 'BASic')
        # check OPTIMIZED setup with PDFCalculator where it is supported
        pdfc = PDFCalculator()
        self.assertEqual('BASIC', pdfc.evaluatortype)
        pdfc.evaluatortype = 'OPTIMIZED'
        self.assertEqual('OPTIMIZED', pdfc.evaluatortype)
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
        self.assertTrue(et0 < et1)
        return


    def test_ticker_override(self):
        """check Python override of PairQuantity.ticker.
        """
        pqcnt = PQCounter()
        self.assertEqual(0, pqcnt.tcnt)
        et0 = pqcnt.ticker()
        self.assertEqual(1, pqcnt.tcnt)
        et1 = PairQuantity.ticker(pqcnt)
        self.assertEqual(1, pqcnt.tcnt)
        self.assertEqual(et0, et1)
        et0.click()
        self.assertEqual(et0, et1)
        # BASIC evaluator does not call the ticker method.
        pqcnt.eval()
        self.assertEqual(1, pqcnt.tcnt)
        # Check if ticker call from OPTIMIZED evaluator is handled
        # with our Python override.
        pqcnt.evaluatortype = 'OPTIMIZED'
        self.assertEqual(1, pqcnt.tcnt)
        pqcnt.eval()
        self.assertEqual(2, pqcnt.tcnt)
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
        pqd = PQDerived()
        # wrapper for evaluation using specified evaluatortype.
        # Use pq.eval twice to trigger optimized evaluation.
        eval_as = lambda evtp, pq, stru : (
            setattr(pq, 'evaluatortype', evtp),
            pq.eval(stru), pq.eval())[-1]
        eval_as('BASIC', pqd, c8)
        self.assertEqual('BASIC', pqd.evaluatortype)
        # pqd does not support OPTIMIZED evaluation.  Its use will
        # raise ValueError or RuntimeError for older libdiffpy.
        # Here we check for StandardError that covers them both.
        self.assertRaises(StandardError,
                          eval_as, 'OPTIMIZED', pqd, c8)
        # PQCounter supports OPTIMIZED evaluation mode.
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
        self.assertTrue(stru0 is self.pq.getStructure())
        stru1 = pq1.getStructure()
        self.assertTrue(type(stru1) is DerivedStructureAdapter)
        self.assertFalse(stru1 is stru0)
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

class PQCounter(PQDerived):

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
