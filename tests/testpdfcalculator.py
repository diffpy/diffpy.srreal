#!/usr/bin/env python

"""Unit tests for pdfcalculator.py
"""

# version
__id__ = '$Id$'

import os
import unittest

import numpy
from diffpy.srreal.pdfcalculator import PDFCalculator

# useful variables
thisfile = locals().get('__file__', 'file.py')
tests_dir = os.path.dirname(os.path.abspath(thisfile))
testdata_dir = os.path.join(tests_dir, 'testdata')

# helper functions

def _loadTestStructure(basefilename):
    from diffpy.Structure import Structure
    fullpath = os.path.join(testdata_dir, basefilename)
    stru = Structure(filename=fullpath)
    return stru


def _maxNormDiff(yobs, ycalc):
    '''Returned maximum difference normalized by RMS of the yobs
    '''
    yobsa = numpy.array(yobs)
    obsmax = numpy.max(numpy.fabs(yobsa)) or 1
    ynmdiff = (yobsa - ycalc) / obsmax
    rv = max(numpy.fabs(ynmdiff))
    return rv


##############################################################################
class TestPDFCalculator(unittest.TestCase):

    nickel = None
    tio2rutile = None

    def setUp(self):
        self.pdfcalc = PDFCalculator()
        if not TestPDFCalculator.nickel:
            TestPDFCalculator.nickel = _loadTestStructure('Ni.stru')
        if not TestPDFCalculator.tio2rutile:
            TestPDFCalculator.tio2rutile = \
                    _loadTestStructure('TiO2_rutile-fit.stru')
        return

    def tearDown(self):
        return

#   def test___init__(self):
#       """check PDFCalculator.__init__()
#       """
#       return

    def test__getDoubleAttr(self):
        """check PDFCalculator._getDoubleAttr()
        """
        gdba = self.pdfcalc._getDoubleAttr
        self.assertEqual(1.0, gdba('scale'))
        self.assertEqual(0.0, gdba('qdamp'))
        self.assertRaises(Exception, gdba, 'notanattribute')
        return

    def test__hasDoubleAttr(self):
        """check PDFCalculator._hasDoubleAttr()
        """
        self.failUnless(self.pdfcalc._hasDoubleAttr('scale'))
        self.failIf(self.pdfcalc._hasDoubleAttr('notanattribute'))
        return

    def test__namesOfDoubleAttributes(self):
        """check PDFCalculator._namesOfDoubleAttributes()
        """
        self.failUnless(type(self.pdfcalc._namesOfDoubleAttributes()) is set)
        self.failUnless('qmax' in self.pdfcalc._namesOfDoubleAttributes())
        return

    def test__setDoubleAttr(self):
        """check PDFCalculator._setDoubleAttr()
        """
        gdba = self.pdfcalc._getDoubleAttr
        sdba = self.pdfcalc._setDoubleAttr
        self.assertEqual(0.0, gdba('rmin'))
        sdba('rmin', 3.0)
        self.assertEqual(3.0, gdba('rmin'))
        return

    def test_eval_nickel(self):
        """check PDFCalculator.eval() on simple Nickel data
        """
        fnipf2 = os.path.join(testdata_dir, 'Ni-fit.fgr')
        gpf2 = numpy.loadtxt(fnipf2, usecols=(1,))
        self.pdfcalc._setDoubleAttr('rmax', 10.0001)
        self.pdfcalc.eval(self.nickel)
        gcalc = self.pdfcalc.getPDF()
        self.failUnless(_maxNormDiff(gpf2, gcalc) < 0.0091)
        return

    def test_eval_rutile(self):
        """check PDFCalculator.eval() on anisotropic rutile data
        """
        frutile = os.path.join(testdata_dir, 'TiO2_rutile-fit.fgr')
        gpf2 = numpy.loadtxt(frutile, usecols=(1,))
        # configure calculator according to testdata/TiO2_ruitile-fit.fgr
        self.pdfcalc.qmax = 26
        self.pdfcalc.qdamp = 0.0665649
        dscale = 0.655857
        self.pdfcalc.rmin = 1
        self.pdfcalc.rmax = 30.0001
        # apply data scale
        self.pdfcalc(self.tio2rutile)
        self.pdfcalc.scale *= dscale
        gcalc = self.pdfcalc.getPDF()
        # termination at rmin is poorly cut in PDFfit2
        mxnd = _maxNormDiff(gpf2, gcalc)
        self.failUnless(mxnd < 0.057)
        # more accurate from 1.5
        mxnd1 = _maxNormDiff(gpf2[:500], gcalc[:500])
        mxnd2 = _maxNormDiff(gpf2[500:], gcalc[500:])
        self.failUnless(mxnd1 < 0.056)
        self.failUnless(mxnd2 < 0.020)
        return


#   def test_getPDF(self):
#       """check PDFCalculator.getPDF()
#       """
#       return

#   def test_getRDF(self):
#       """check PDFCalculator.getRDF()
#       """
#       return

#   def test_getRadiationType(self):
#       """check PDFCalculator.getRadiationType()
#       """
#       return

#   def test_getRgrid(self):
#       """check PDFCalculator.getRgrid()
#       """
#       return

#   def test_setScatteringFactorTable(self):
#       """check PDFCalculator.setScatteringFactorTable()
#       """
#       return

# End of class TestPDFCalculator


if __name__ == '__main__':
    unittest.main()

# End of file
