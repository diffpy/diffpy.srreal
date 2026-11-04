#!/usr/bin/env python

"""Unit tests for diffpy.srreal.pairquantity."""

import pickle
import unittest

import numpy
from testutils import mod_structure

from diffpy.srreal.pairquantity import PairQuantity
from diffpy.srreal.pdfcalculator import PDFCalculator
from diffpy.srreal.srreal_ext import BasePairQuantity, StructureDifference

# ----------------------------------------------------------------------------


class TestBasePairQuantity(unittest.TestCase):

    def setUp(self):
        self.bpq = BasePairQuantity()
        return

    def test_pickling(self):
        "verify pickling is disabled for the C++ base class."
        self.assertRaises(RuntimeError, pickle.dumps, self.bpq)
        return


# End of class TestBasePairQuantity

# ----------------------------------------------------------------------------


class TestPairQuantity(unittest.TestCase):

    def setUp(self):
        self.pq = PairQuantity()
        return

    def test_evaluatortype(self):
        """Check PairQuantity.evaluatortype property."""
        pq = self.pq
        self.assertTrue(pq.evaluatortype in ("BASIC", "OPTIMIZED"))
        pq.evaluatortype = "BASIC"
        self.assertEqual("BASIC", pq.evaluatortype)
        self.assertRaises(ValueError, setattr, pq, "evaluatortype", "invalid")
        self.assertRaises(ValueError, setattr, pq, "evaluatortype", "basic")
        self.assertRaises(ValueError, setattr, pq, "evaluatortype", "BASic")
        # check all supported evaluators in PDFCalculator
        pdfc = PDFCalculator()
        self.assertEqual("OPTIMIZED", pdfc.evaluatortype)
        pdfc.evaluatortype = "BASIC"
        self.assertEqual("BASIC", pdfc.evaluatortype)
        pdfc.evaluatortype = "CHECK"
        self.assertEqual("CHECK", pdfc.evaluatortype)
        pdfc.evaluatortype = "OPTIMIZED"
        self.assertEqual("OPTIMIZED", pdfc.evaluatortype)
        return

    def test_setStructure(self):
        """Check PairQuantity.setStructure()"""
        Structure = mod_structure.Structure
        Atom = mod_structure.Atom
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

    def test_setPairMask_args(self):
        """Check argument type handling in setPairMask."""
        spm = self.pq.setPairMask
        gpm = self.pq.getPairMask
        self.assertRaises(TypeError, spm, 0.0, 0, False)
        self.assertRaises(TypeError, spm, complex(0.5), 0, False)
        self.assertTrue(gpm(0, 0))
        spm(numpy.int32(1), 0, True, others=False)
        self.assertTrue(gpm(0, 1))
        self.assertTrue(gpm(1, 0))
        self.assertFalse(gpm(0, 0))
        self.assertFalse(gpm(2, 7))
        return

    def test_numpy_index_layouts(self):
        """Index conversion follows logical values, not physical
        storage."""
        values = numpy.arange(6, dtype=numpy.intc)
        swapped = values.astype(values.dtype.newbyteorder("S"))
        unaligned = numpy.ndarray(
            (3,), dtype=numpy.int64, buffer=bytearray(25), offset=1
        )
        unaligned[:] = [0, 2, 4]
        for indices in [values[::2], values[::-1], swapped, unaligned]:
            difference = StructureDifference()
            difference.pop0 = indices
            difference.add1 = indices
            self.assertEqual(difference.pop0, indices.tolist())
            self.assertEqual(difference.add1, indices.tolist())
            self.pq.setPairMask(indices, 7, True, others=False)
            for i in range(6):
                self.assertEqual(self.pq.getPairMask(i, 7), i in indices)

    def test_numpy_index_bounds(self):
        """Wide integers must not wrap into valid indices."""
        limits = numpy.iinfo(numpy.intc)
        difference = StructureDifference()
        for dtype in [
            numpy.int8,
            numpy.uint8,
            numpy.int16,
            numpy.uint16,
            numpy.int32,
            numpy.uint32,
            numpy.int64,
            numpy.uint64,
        ]:
            difference.pop0 = numpy.array([0, 1, 3], dtype=dtype)
            self.assertEqual(difference.pop0, [0, 1, 3])
        difference.pop0 = numpy.array(
            [limits.min, limits.max], dtype=numpy.int64
        )
        self.assertEqual(difference.pop0, [limits.min, limits.max])
        for dtype, values in [
            (numpy.int64, [limits.min - 1, limits.max + 1, 2**32]),
            (numpy.uint32, [limits.max + 1, numpy.iinfo(numpy.uint32).max]),
            (numpy.uint64, [limits.max + 1, numpy.iinfo(numpy.uint64).max]),
        ]:
            for value in values:
                indices = numpy.array([value], dtype=dtype)
                difference.pop0 = [5]
                with self.assertRaises(OverflowError):
                    difference.pop0 = indices
                self.assertEqual(difference.pop0, [5])
                with self.assertRaises(OverflowError):
                    self.pq.setPairMask(indices, 0, True)

    def test_numpy_indices_reject_nonintegers(self):
        for indices in [
            numpy.array([1.5]),
            numpy.array(["1"]),
            numpy.datetime64("1970-01-03"),
            numpy.timedelta64(2, "D"),
        ]:
            with self.assertRaises(TypeError):
                StructureDifference().pop0 = indices
            with self.assertRaises(TypeError):
                self.pq.setPairMask(indices, 0, True)

    def test_getStructure(self):
        """Check PairQuantity.getStructure()"""
        adpt = self.pq.getStructure()
        self.assertEqual(0, adpt.countSites())
        return

    def test_ticker(self):
        """Check PairQuantity.ticker()"""
        from diffpy.srreal.eventticker import EventTicker

        et0 = EventTicker(self.pq.ticker())
        self.pq.rmax = 3.77
        et1 = self.pq.ticker()
        self.assertNotEqual(et0, et1)
        self.assertTrue(et0 < et1)
        return

    def test_ticker_override(self):
        """Check Python override of PairQuantity.ticker."""
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
        pqcnt.evaluatortype = "OPTIMIZED"
        self.assertEqual(1, pqcnt.tcnt)
        pqcnt.eval()
        self.assertEqual(2, pqcnt.tcnt)
        return

    def test__addPairContribution(self):
        """Check Python override of
        PairQuantity._addPairContribution."""
        pqcnt = PQCounter()
        self.assertEqual(0, pqcnt(carbonzchain(0)))
        self.assertEqual(0, pqcnt(carbonzchain(1)))
        self.assertEqual(1, pqcnt(carbonzchain(2)))
        self.assertEqual(10, pqcnt(carbonzchain(5)))
        return

    def test_optimized_evaluation(self):
        """Check OPTIMIZED evaluation in Python-defined calculator
        class."""
        c8 = carbonzchain(8)
        c9 = carbonzchain(9)
        pqd = PQDerived()

        # wrapper for evaluation using specified evaluatortype.
        # Use pq.eval twice to trigger optimized evaluation.
        def eval_as(evtp, pq, stru):
            setattr(pq, "evaluatortype", evtp)
            pq.eval(stru)
            return pq.eval()

        eval_as("BASIC", pqd, c8)
        self.assertEqual("BASIC", pqd.evaluatortype)
        # pqd does not support OPTIMIZED evaluation.  Its use will
        # raise ValueError or RuntimeError for older libdiffpy.
        # Here we check for StandardError that covers them both.
        self.assertRaises(Exception, eval_as, "OPTIMIZED", pqd, c8)
        # PQCounter supports OPTIMIZED evaluation mode.
        ocnt = PQCounter()
        ocnt.evaluatortype = "OPTIMIZED"
        self.assertEqual(28, ocnt(c8))
        self.assertEqual(28, ocnt(c8))
        self.assertEqual("OPTIMIZED", ocnt.evaluatortypeused)
        self.assertEqual(36, ocnt(c9))
        self.assertEqual("OPTIMIZED", ocnt.evaluatortypeused)
        self.assertEqual(28, ocnt(c8))
        self.assertEqual("OPTIMIZED", ocnt.evaluatortypeused)
        return

    def test_pickling(self):
        """Check pickling and unpickling of PairQuantity."""
        from testutils import DerivedStructureAdapter

        stru0 = DerivedStructureAdapter()
        stru0.label = "saved structure"
        self.pq.setStructure(stru0)
        self.assertEqual("saved structure", stru0.label)
        spkl = pickle.dumps(self.pq)
        pq1 = pickle.loads(spkl)
        self.assertTrue(stru0 is self.pq.getStructure())
        stru1 = pq1.getStructure()
        self.assertTrue(type(stru1) is DerivedStructureAdapter)
        self.assertFalse(stru1 is stru0)
        self.assertEqual("saved structure", stru1.label)
        # check pickling of attributes
        pcnt = PQCounter()
        pcnt.foo = "asdf"
        pcnt2 = pickle.loads(pickle.dumps(pcnt))
        self.assertTrue(isinstance(pcnt2, PQCounter))
        self.assertEqual("asdf", pcnt2.foo)
        return


# End of class TestPairQuantity

# ----------------------------------------------------------------------------

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
        (rv,) = self.eval(structure)
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
    Structure = mod_structure.Structure
    Atom = mod_structure.Atom
    rv = Structure([Atom("C", [0, 0, z]) for z in range(n)])
    return rv


if __name__ == "__main__":
    unittest.main()

# End of file
