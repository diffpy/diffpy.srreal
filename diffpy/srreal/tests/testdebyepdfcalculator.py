#!/usr/bin/env python

"""Unit tests for pdfcalculator.py
"""


import unittest
import cPickle
import numpy

from diffpy.srreal.pdfcalculator import DebyePDFCalculator, PDFCalculator
from diffpy.srreal.tests.testutils import loadDiffPyStructure
from testpdfcalculator import _maxNormDiff

##############################################################################
class TestDebyePDFCalculator(unittest.TestCase):

    bucky = None
    tio2rutile = None

    def setUp(self):
        self.dpdfc = DebyePDFCalculator()
        if not TestDebyePDFCalculator.bucky:
            TestDebyePDFCalculator.bucky = (
                    loadDiffPyStructure('C60bucky.stru'))
        if not TestDebyePDFCalculator.tio2rutile:
            TestDebyePDFCalculator.tio2rutile = (
                    loadDiffPyStructure('TiO2_rutile-fit.stru'))
        return

#   def tearDown(self):
#       return
#
#   def test___call__(self):
#       """check DebyePDFCalculator.__call__()
#       """
#       return
#
#   def test___init__(self):
#       """check DebyePDFCalculator.__init__()
#       """
#       return

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
        self.assertTrue(self.dpdfc._hasDoubleAttr('scale'))
        self.assertFalse(self.dpdfc._hasDoubleAttr('notanattribute'))
        return

    def test__namesOfDoubleAttributes(self):
        """check DebyePDFCalculator._namesOfDoubleAttributes()
        """
        self.assertTrue(type(self.dpdfc._namesOfDoubleAttributes()) is set)
        self.assertTrue('qmax' in self.dpdfc._namesOfDoubleAttributes())
        return

    def test__setDoubleAttr(self):
        """check DebyePDFCalculator._setDoubleAttr()
        """
        gdba = self.dpdfc._getDoubleAttr
        sdba = self.dpdfc._setDoubleAttr
        self.assertEqual(0.0, gdba('rmin'))
        sdba('rmin', 3.0)
        self.assertEqual(3.0, gdba('rmin'))
        return

    def test_PDF_C60bucky(self):
        """check DebyePDFCalculator.pdf for C60 Bucky ball.
        """
        qmax = self.dpdfc.qmax
        r0, g0 = PDFCalculator(qmax=qmax)(self.bucky)
        r1, g1 = self.dpdfc(self.bucky)
        mxnd = _maxNormDiff(g0, g1)
        self.assertTrue(mxnd < 0.0006)
        return

    def test_partial_pdfs(self):
        """Check calculation of partial PDFs.
        """
        dpdfc = self.dpdfc
        dpdfc.qmin = 1.0
        rutile = self.tio2rutile
        r0, g0 = dpdfc(rutile)
        # Ti-Ti
        dpdfc.maskAllPairs(False)
        dpdfc.setTypeMask("Ti", "Ti", True)
        r1, g1 = dpdfc(rutile)
        self.assertTrue(numpy.array_equal(r0, r1))
        dpdfc.invertMask()
        r1i, g1i = dpdfc(rutile)
        self.assertTrue(numpy.array_equal(r0, r1i))
        self.assertTrue(numpy.allclose(g0, g1 + g1i))
        # Ti-O
        dpdfc.maskAllPairs(False)
        dpdfc.setTypeMask('all', 'ALL', True)
        dpdfc.setTypeMask('Ti', 'Ti', False)
        dpdfc.setTypeMask('O', 'O', False)
        r2, g2 = dpdfc(rutile)
        self.assertTrue(numpy.array_equal(r0, r2))
        dpdfc.invertMask()
        r2i, g2i = dpdfc(rutile)
        self.assertTrue(numpy.allclose(g0, g2 + g2i))
        # Ti-O from type mask
        dpdfc.maskAllPairs(True)
        dpdfc.setTypeMask("Ti", "Ti", False)
        dpdfc.setTypeMask("O", "O", False)
        r2t, g2t = dpdfc(rutile)
        self.assertTrue(numpy.array_equal(r0, r2t))
        self.assertTrue(numpy.array_equal(g2, g2t))
        dpdfc.invertMask()
        r2ti, g2ti = dpdfc(rutile)
        self.assertTrue(numpy.array_equal(g2i, g2ti))
        # O-O
        dpdfc.maskAllPairs(False)
        dpdfc.setTypeMask('O', 'O', True)
        r3, g3 = dpdfc(rutile)
        dpdfc.invertMask()
        r3i, g3i = dpdfc(rutile)
        self.assertTrue(numpy.allclose(g0, g3 + g3i))
        # check the sum of all partials
        self.assertTrue(numpy.allclose(g0, g1 + g2 + g3))
        return

    def test_pickling(self):
        '''check pickling and unpickling of PDFCalculator.
        '''
        dpdfc = self.dpdfc
        dpdfc.setScatteringFactorTableByType('N')
        dpdfc.scatteringfactortable.setCustomAs('Na', 'Na', 7)
        dpdfc.addEnvelope('sphericalshape')
        dpdfc.debyeprecision = 0.001
        dpdfc.delta1 = 0.2
        dpdfc.delta2 = 0.3
        dpdfc.maxextension = 10.1
        dpdfc.qbroad = 0.01
        dpdfc.qdamp = 0.05
        dpdfc.qmax = 10
        dpdfc.qmin = 0.5
        dpdfc.rmax = 10.0
        dpdfc.rmin = 0.02
        dpdfc.rstep = 0.02
        dpdfc.scale = 1.1
        dpdfc.spdiameter = 13.3
        dpdfc.foobar = 'asdf'
        spkl = cPickle.dumps(dpdfc)
        dpdfc1 = cPickle.loads(spkl)
        self.assertFalse(dpdfc is dpdfc1)
        sft = dpdfc.scatteringfactortable
        sft1 = dpdfc1.scatteringfactortable
        self.assertEqual(sft.type(), sft1.type())
        self.assertEqual(7.0, sft1.lookup('Na'))
        for a in dpdfc._namesOfDoubleAttributes():
            self.assertEqual(getattr(dpdfc, a), getattr(dpdfc1, a))
        self.assertEqual(13.3,
                dpdfc1.getEnvelope('sphericalshape').spdiameter)
        self.assertEqual(dpdfc._namesOfDoubleAttributes(),
                dpdfc1._namesOfDoubleAttributes())
        self.assertEqual(dpdfc.usedenvelopetypes, dpdfc1.usedenvelopetypes)
        self.assertEqual('asdf', dpdfc1.foobar)
        return


    def test_mask_pickling(self):
        '''Check if mask gets properly pickled and restored.
        '''
        self.dpdfc.maskAllPairs(False)
        self.dpdfc.setPairMask(0, 1, True)
        self.dpdfc.setTypeMask("Na", "Cl", True)
        self.assertTrue(False is self.dpdfc.getPairMask(0, 0))
        self.assertTrue(True is self.dpdfc.getPairMask(0, 1))
        self.assertTrue(True is self.dpdfc.getTypeMask("Cl", "Na"))
        dpdfc1 = cPickle.loads(cPickle.dumps(self.dpdfc))
        self.assertTrue(False is dpdfc1.getPairMask(0, 0))
        self.assertTrue(True is dpdfc1.getPairMask(0, 1))
        self.assertTrue(True is self.dpdfc.getTypeMask("Cl", "Na"))
        return


    def test_pickling_derived_structure(self):
        '''check pickling of DebyePDFCalculator with DerivedStructureAdapter.
        '''
        from diffpy.srreal.tests.testutils import DerivedStructureAdapter
        dpdfc = self.dpdfc
        stru0 = DerivedStructureAdapter()
        dpdfc.setStructure(stru0)
        self.assertEqual(1, stru0.cpqcount)
        spkl = cPickle.dumps(dpdfc)
        dpdfc1 = cPickle.loads(spkl)
        self.assertTrue(stru0 is dpdfc.getStructure())
        stru1 = dpdfc1.getStructure()
        self.assertTrue(type(stru1) is DerivedStructureAdapter)
        self.assertFalse(stru1 is stru0)
        self.assertEqual(1, stru1.cpqcount)
        return


#   def test_getPeakWidthModel(self):
#       """check DebyePDFCalculator.getPeakWidthModel()
#       """
#       return
#
#   def test_qgrid(self):
#       """check DebyePDFCalculator.qgrid
#       """
#       return
#
#   def test_getRadiationType(self):
#       """check DebyePDFCalculator.getRadiationType()
#       """
#       return
#
#   def test_rgrid(self):
#       """check DebyePDFCalculator.rgrid
#       """
#       return
#
#   def test_scatteringfactortable(self):
#       """check DebyePDFCalculator.scatteringfactortable property
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
#   def test_peakwidthmodel(self):
#       """check DebyePDFCalculator.setPeakWidthModel()
#       """
#       return
#
#   def test_value(self):
#       """check DebyePDFCalculator.value
#       """
#       return

# End of class TestDebyePDFCalculator


if __name__ == '__main__':
    unittest.main()

# End of file
