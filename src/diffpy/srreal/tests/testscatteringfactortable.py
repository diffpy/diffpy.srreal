#!/usr/bin/env python

"""Unit tests for diffpy.srreal.scatteringfactortable
"""


import unittest
import pickle
import numpy

from diffpy.srreal.tests.testutils import pickle_with_attr
from diffpy.srreal.scatteringfactortable import ScatteringFactorTable
from diffpy.srreal.scatteringfactortable import SFTXray, SFTElectron
from diffpy.srreal.scatteringfactortable import SFTNeutron, SFTElectronNumber
from diffpy.srreal.pdfcalculator import PDFCalculator, DebyePDFCalculator

# ----------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------

class TestScatteringFactorTable(unittest.TestCase):

    def setUp(self):
        self.sftx = ScatteringFactorTable.createByType('X')
        self.sftn = ScatteringFactorTable.createByType('N')
        LocalTable()._registerThisType()
        return

    def tearDown(self):
        ScatteringFactorTable._deregisterType('localtable')
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
        sftx1 = pickle.loads(pickle.dumps(self.sftx))
        self.assertEqual(2, len(sftx1.getCustomSymbols()))
        self.assertAlmostEqual(123, sftx1.lookup('Na'), 12)
        self.assertEqual(self.sftx.lookup('C'), sftx1.lookup('Calias'))
        self.assertEqual(self.sftx.type(), sftx1.type())
        pwa = pickle_with_attr
        self.assertRaises(RuntimeError, pwa, SFTXray(), foo='bar')
        self.assertRaises(RuntimeError, pwa, SFTElectron(), foo='bar')
        self.assertRaises(RuntimeError, pwa, SFTNeutron(), foo='bar')
        self.assertRaises(RuntimeError, pwa, SFTElectronNumber(), foo='bar')
        return

    def test_picking_owned(self):
        '''verify pickling of envelopes owned by PDF calculators.
        '''
        pc = PDFCalculator()
        dbpc = DebyePDFCalculator()
        ltb = LocalTable()
        ltb.setCustomAs('Na', 'Na', 37)
        ltb.foo = 'bar'
        pc.scatteringfactortable = ltb
        dbpc.scatteringfactortable = ltb
        self.assertIs(ltb, pc.scatteringfactortable)
        self.assertIs(ltb, dbpc.scatteringfactortable)
        pc2 = pickle.loads(pickle.dumps(pc))
        dbpc2 = pickle.loads(pickle.dumps(dbpc))
        self.assertEqual('localtable', pc2.scatteringfactortable.type())
        self.assertEqual('localtable', dbpc2.scatteringfactortable.type())
        self.assertEqual(37, pc2.scatteringfactortable.lookup('Na'))
        self.assertEqual(37, dbpc2.scatteringfactortable.lookup('Na'))
        self.assertEqual('bar', pc2.scatteringfactortable.foo)
        self.assertEqual('bar', dbpc2.scatteringfactortable.foo)
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
        lsft1 = pickle.loads(pickle.dumps(lsft))
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

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()

# End of file
