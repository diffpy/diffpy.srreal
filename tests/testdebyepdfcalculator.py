#!/usr/bin/env python

"""Unit tests for pdfcalculator.py
"""

# version
__id__ = '$Id$'

import os
import unittest

# useful variables
thisfile = locals().get('__file__', 'file.py')
tests_dir = os.path.dirname(os.path.abspath(thisfile))
testdata_dir = os.path.join(tests_dir, 'testdata')

from diffpy.srreal.pdfcalculator import DebyePDFCalculator, PDFCalculator
from testpdfcalculator import _loadTestStructure, _maxNormDiff

##############################################################################
class TestDebyePDFCalculator(unittest.TestCase):

    bucky = None

    def setUp(self):
        from diffpy.Structure import Structure
        self.dpdfc = DebyePDFCalculator()
        if not TestDebyePDFCalculator.bucky:
            TestDebyePDFCalculator.bucky = _loadTestStructure('C60bucky.stru')
        return

    def tearDown(self):
        return

    def test___call__(self):
        """check DebyePDFCalculator.__call__()
        """
        return

    def test___init__(self):
        """check DebyePDFCalculator.__init__()
        """
        return

    def test___getattr__(self):
        """check DebyePDFCalculator.__getattr__()
        """
        self.assertEqual(0.0, self.dpdfc.qmin)
        self.dpdfc._setDoubleAttr('qmin', 1.23)
        self.assertEqual(1.23, self.dpdfc.qmin)
        return

    def test___setattr__(self):
        """check DebyePDFCalculator.__setattr__()
        """
        self.assertNotEquals(1.23, self.dpdfc._getDoubleAttr('rmin'))
        self.dpdfc.rmin = 1.23
        self.assertEqual(1.23, self.dpdfc._getDoubleAttr('rmin'))
        return

    def test__getDoubleAttr(self):
        """check DebyePDFCalculator._getDoubleAttr()
        """
        gdba = self.dpdfc._getDoubleAttr
        self.assertEqual(1.0, gdba('scale'))
        self.assertEqual(0.0, gdba('qdamp'))
        self.assertRaises(Exception, gdba, 'notanattribute')
        return

    def test__hasDoubleAttr(self):
        """check DebyePDFCalculator._hasDoubleAttr()
        """
        self.failUnless(self.dpdfc._hasDoubleAttr('scale'))
        self.failIf(self.dpdfc._hasDoubleAttr('notanattribute'))
        return

    def test__namesOfDoubleAttributes(self):
        """check DebyePDFCalculator._namesOfDoubleAttributes()
        """
        self.failUnless(type(self.dpdfc._namesOfDoubleAttributes()) is set)
        self.failUnless('qmax' in self.dpdfc._namesOfDoubleAttributes())
        return

#   def test__registerDoubleAttribute(self):
#       """check DebyePDFCalculator._registerDoubleAttribute()
#       """
#       return

    def test__setDoubleAttr(self):
        """check DebyePDFCalculator._setDoubleAttr()
        """
        gdba = self.dpdfc._getDoubleAttr
        sdba = self.dpdfc._setDoubleAttr
        self.assertEqual(0.0, gdba('rmin'))
        sdba('rmin', 3.0)
        self.assertEqual(3.0, gdba('rmin'))
        return

#   def test_eval(self):
#       """check DebyePDFCalculator.eval()
#       """
#       return
#
#   def test_getF(self):
#       """check DebyePDFCalculator.getF()
#       """
#       return

    def test_getPDF_C60bucky(self):
        """check DebyePDFCalculator.getPDF()
        """
        qmax = self.dpdfc.qmax
        r0, g0 = PDFCalculator(qmax=qmax)(self.bucky)
        r1, g1 = self.dpdfc(self.bucky)
        mxnd = _maxNormDiff(g0, g1)
        self.failUnless(mxnd < 0.0006)
        return

#   def test_getPeakWidthModel(self):
#       """check DebyePDFCalculator.getPeakWidthModel()
#       """
#       return
#
#   def test_getQgrid(self):
#       """check DebyePDFCalculator.getQgrid()
#       """
#       return
#
#   def test_getRadiationType(self):
#       """check DebyePDFCalculator.getRadiationType()
#       """
#       return
#
#   def test_getRgrid(self):
#       """check DebyePDFCalculator.getRgrid()
#       """
#       return
#
#   def test_getScatteringFactorTable(self):
#       """check DebyePDFCalculator.getScatteringFactorTable()
#       """
#       return
#
#   def test_isOptimumQstep(self):
#       """check DebyePDFCalculator.isOptimumQstep()
#       """
#       return
#
#   def test_setOptimumQstep(self):
#       """check DebyePDFCalculator.setOptimumQstep()
#       """
#       return
#
#   def test_setPeakWidthModel(self):
#       """check DebyePDFCalculator.setPeakWidthModel()
#       """
#       return
#
#   def test_setPeakWidthModelByType(self):
#       """check DebyePDFCalculator.setPeakWidthModelByType()
#       """
#       return
#
#   def test_setScatteringFactorTable(self):
#       """check DebyePDFCalculator.setScatteringFactorTable()
#       """
#       return
#
#   def test_setScatteringFactorTableByType(self):
#       """check DebyePDFCalculator.setScatteringFactorTableByType()
#       """
#       return
#
#   def test_value(self):
#       """check DebyePDFCalculator.value()
#       """
#       return

# End of class TestDebyePDFCalculator


if __name__ == '__main__':
    unittest.main()

# End of file
