#!/usr/bin/env python

"""Unit tests for diffpy.srreal.scatteringfactortable
"""


import unittest
import cPickle
import numpy

from diffpy.srreal.scatteringfactortable import ScatteringFactorTable


class LocalTable(ScatteringFactorTable):
    def clone(self):
        import copy
        return copy.copy(self)
    def create(self): return LocalTable()
    def _standardLookup(self, smbl, q):   return q + 1
    def radiationType(self):   return "LTB"
    def type(self):   return "localtable"
    def ticker(self):
        self.tcnt += 1
        return ScatteringFactorTable.ticker(self)
    tcnt = 0

LocalTable()._registerThisType()


##############################################################################
class TestScatteringFactorTable(unittest.TestCase):

    def setUp(self):
        self.sftx = ScatteringFactorTable.createByType('X')
        self.sftn = ScatteringFactorTable.createByType('N')
        return

    def tearDown(self):
        return

    def test_class_registry(self):
        """check if instances are aliased by radiationType().
        """
        ltb = ScatteringFactorTable.createByType('LTB')
        self.assertTrue(type(ltb) is LocalTable)
        ltb2 = ScatteringFactorTable.createByType('localtable')
        self.assertTrue(type(ltb2) is LocalTable)
        return

    def test_ticker(self):
        """check ScatteringFactorTable.ticker()
        """
        from diffpy.srreal.eventticker import EventTicker
        et0 = EventTicker(self.sftx.ticker())
        self.sftx.setCustomAs('D', 'H')
        et1 = self.sftx.ticker()
        self.assertNotEqual(et0, et1)
        self.assertTrue(et0 < et1)
        return

    def test_ticker_override(self):
        """check Python override of ScatteringFactorTable.ticker.
        """
        from diffpy.srreal.pdfcalculator import PDFCalculator
        lsft = LocalTable()
        self.assertEqual(0, lsft.tcnt)
        et0 = lsft.ticker()
        self.assertEqual(1, lsft.tcnt)
        et1 = ScatteringFactorTable.ticker(lsft)
        self.assertEqual(1, lsft.tcnt)
        self.assertEqual(et0, et1)
        et0.click()
        self.assertEqual(et0, et1)
        # check that implicit ticker call from PDFCalculator is
        # handled by Python override of the ticker method.
        pc = PDFCalculator()
        pc.scatteringfactortable = lsft
        pc.ticker()
        self.assertEqual(2, lsft.tcnt)
        return

    def test_pickling(self):
        """check pickling of ScatteringFactorTable instances.
        """
        self.assertEqual(0, len(self.sftx.getCustomSymbols()))
        self.sftx.setCustomAs('Na', 'Na', 123)
        self.sftx.setCustomAs('Calias', 'C')
        self.assertEqual(2, len(self.sftx.getCustomSymbols()))
        sftx1 = cPickle.loads(cPickle.dumps(self.sftx))
        self.assertEqual(2, len(sftx1.getCustomSymbols()))
        self.assertAlmostEqual(123, sftx1.lookup('Na'), 12)
        self.assertEqual(self.sftx.lookup('C'), sftx1.lookup('Calias'))
        self.assertEqual(self.sftx.type(), sftx1.type())
        return

    def test_pickling_derived(self):
        """check pickling of a derived classes.
        """
        lsft = LocalTable()
        self.assertEqual(3, lsft._standardLookup('Na', 2))
        self.assertEqual(set(), lsft.getCustomSymbols())
        lsft.foobar = 'asdf'
        lsft.setCustomAs('Na', 'Na', 123)
        self.assertEqual(1, len(lsft.getCustomSymbols()))
        lsft1 = cPickle.loads(cPickle.dumps(lsft))
        self.assertEqual(1, len(lsft1.getCustomSymbols()))
        self.assertAlmostEqual(123, lsft1.lookup('Na'), 12)
        self.assertEqual('asdf', lsft1.foobar)
        self.assertEqual(lsft.type(), lsft1.type())
        self.assertEqual(3, lsft1._standardLookup('Cl', 2))
        self.assertEqual(1, lsft1.lookup('H'))
        return

    def test_derived_create(self):
        """Check override of ScatteringFactorTable.create in Python class.
        """
        lsft = LocalTable()
        lsft.setCustomAs('Xy', 'Na')
        lsft2 = lsft.create()
        self.assertTrue(isinstance(lsft2, LocalTable))
        self.assertEqual(set(), lsft2.getCustomSymbols())
        return

    def test_derived_clone(self):
        """Check override of ScatteringFactorTable.clone in Python class.
        """
        lsft = LocalTable()
        lsft.setCustomAs('Xy', 'Na')
        lsft2 = lsft.clone()
        self.assertTrue(isinstance(lsft2, LocalTable))
        self.assertEqual(set(['Xy']), lsft2.getCustomSymbols())
        return

    def test_lookup(self):
        """Check ScatteringFactorTable.lookup handling of array arguments.
        """
        qa = numpy.linspace(0, 50)
        sftx = self.sftx
        fmn0 = numpy.array([sftx.lookup('Mn', x) for x in qa])
        fmn1 = sftx.lookup('Mn', qa)
        self.assertTrue(numpy.array_equal(fmn0, fmn1))
        self.assertTrue(numpy.array_equal(
            fmn0.reshape(5, 10), sftx.lookup('Mn', qa.reshape(5, 10))))
        self.assertTrue(numpy.array_equal(
            fmn0.reshape(5, 2, 5), sftx.lookup('Mn', qa.reshape(5, 2, 5))))
        self.assertTrue(numpy.array_equal(fmn0, sftx.lookup('Mn', list(qa))))
        self.assertRaises(TypeError, sftx.lookup, 'Na', 'asdf')
        self.assertRaises(TypeError, sftx.lookup, 'Na', {})
        return

# End of class TestScatteringFactorTable

if __name__ == '__main__':
    unittest.main()

# End of file
