#!/usr/bin/env python

"""Unit tests for diffpy.srreal.pdfcalculator
"""


import unittest
import cPickle

import numpy
from diffpy.srreal.tests.testutils import loadDiffPyStructure
from diffpy.srreal.tests.testutils import datafile
from diffpy.srreal.pdfcalculator import PDFCalculator
from diffpy.srreal.pdfcalculator import fftgtof, fftftog

# helper functions

def _maxNormDiff(yobs, ycalc):
    '''Returned maximum difference normalized by RMS of the yobs
    '''
    yobsa = numpy.array(yobs)
    obsmax = numpy.max(numpy.fabs(yobsa)) or 1
    ynmdiff = (yobsa - ycalc) / obsmax
    rv = max(numpy.fabs(ynmdiff))
    return rv

# ----------------------------------------------------------------------------

class TestPDFCalculator(unittest.TestCase):

    nickel = None
    tio2rutile = None

    def setUp(self):
        self.pdfcalc = PDFCalculator()
        if not self.nickel:
            type(self).nickel = loadDiffPyStructure('Ni.stru')
        if not self.tio2rutile:
            type(self).tio2rutile = (
                    loadDiffPyStructure('TiO2_rutile-fit.stru'))
        return

    def tearDown(self):
        return

    def test___init__(self):
        """check PDFCalculator.__init__()
        """
        pdfc = PDFCalculator(qmin=13, rmin=4, rmax=99)
        self.assertEqual(13, pdfc.qmin)
        self.assertEqual(4, pdfc.rmin)
        self.assertEqual(99, pdfc.rmax)
        self.assertEqual(99, pdfc._getDoubleAttr('rmax'))
        return

    def test___call__(self):
        """Check PDFCalculator.__call__()
        """
        from diffpy.Structure import Structure
        r0, g0 = self.pdfcalc(self.tio2rutile, rmin=2)
        self.assertEqual(2.0, r0[0])
        r1, g1 = self.pdfcalc(self.tio2rutile, scale=7)
        self.assertAlmostEqual(7.0, g1[0] / g0[0])
        # check application of spdiameter
        rutile2 = Structure(self.tio2rutile)
        # work around Structure bug of shared pdffit dictionary
        rutile2.pdffit = dict(self.tio2rutile.pdffit)
        rutile2.pdffit['spdiameter'] = 5.0
        r3, g3 = self.pdfcalc(rutile2)
        self.assertEqual(0.0, sum(g3[r3 >= 5] ** 2))
        r4, g4 = self.pdfcalc(rutile2, scale=1, spdiameter=0)
        self.assertTrue(numpy.all(r4 == r0))
        self.assertTrue(numpy.all(g4 == g0))
        return

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
        self.assertTrue(self.pdfcalc._hasDoubleAttr('scale'))
        self.assertFalse(self.pdfcalc._hasDoubleAttr('notanattribute'))
        return

    def test__namesOfDoubleAttributes(self):
        """check PDFCalculator._namesOfDoubleAttributes()
        """
        self.assertTrue(type(self.pdfcalc._namesOfDoubleAttributes()) is set)
        self.assertTrue('qmax' in self.pdfcalc._namesOfDoubleAttributes())
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
        fnipf2 = datafile('Ni-fit.fgr')
        gpf2 = numpy.loadtxt(fnipf2, usecols=(1,))
        self.pdfcalc._setDoubleAttr('rmax', 10.0001)
        self.pdfcalc.eval(self.nickel)
        gcalc = self.pdfcalc.pdf
        self.assertTrue(_maxNormDiff(gpf2, gcalc) < 0.0091)
        return

    def test_eval_rutile(self):
        """check PDFCalculator.eval() on anisotropic rutile data
        """
        frutile = datafile('TiO2_rutile-fit.fgr')
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
        gcalc = self.pdfcalc.pdf
        # termination at rmin is poorly cut in PDFfit2
        mxnd = _maxNormDiff(gpf2, gcalc)
        self.assertTrue(mxnd < 0.057)
        # more accurate from 1.5
        mxnd1 = _maxNormDiff(gpf2[:500], gcalc[:500])
        mxnd2 = _maxNormDiff(gpf2[500:], gcalc[500:])
        self.assertTrue(mxnd1 < 0.056)
        self.assertTrue(mxnd2 < 0.020)
        return

    def test_partial_pdfs(self):
        """Check calculation of partial PDFs.
        """
        pdfc = self.pdfcalc
        pdfc.rstep = 0.1
        rutile = self.tio2rutile
        atomtypes = [a.element for a in rutile]
        r0, g0 = pdfc(rutile)
        rdf0 = pdfc.rdf
        # Ti-Ti
        pdfc.maskAllPairs(False)
        pdfc.setTypeMask("Ti", "Ti", True)
        r1, g1 = pdfc(rutile)
        rdf1 = pdfc.rdf
        self.assertTrue(numpy.array_equal(r0, r1))
        pdfc.invertMask()
        r1i, g1i = pdfc(rutile)
        rdf1i = pdfc.rdf
        self.assertTrue(numpy.array_equal(r0, r1i))
        self.assertTrue(numpy.allclose(g0, g1 + g1i))
        self.assertTrue(numpy.allclose(rdf0, rdf1 + rdf1i))
        # Ti-O
        pdfc.maskAllPairs(True)
        pdfc.setPairMask(range(2), range(2), False)
        pdfc.setPairMask(range(2, 6), range(2, 6), False)
        r2, g2 = pdfc(rutile)
        rdf2 = pdfc.rdf
        self.assertTrue(numpy.array_equal(r0, r2))
        pdfc.invertMask()
        r2i, g2i = pdfc(rutile)
        rdf2i = pdfc.rdf
        self.assertTrue(numpy.allclose(g0, g2 + g2i))
        self.assertTrue(numpy.allclose(rdf0, rdf2 + rdf2i))
        # Ti-O using type mask
        pdfc.maskAllPairs(True)
        pdfc.setTypeMask("Ti", "Ti", False)
        pdfc.setTypeMask("O", "O", False)
        r2t, g2t = pdfc(rutile)
        rdf2t = pdfc.rdf
        self.assertTrue(numpy.array_equal(r0, r2t))
        self.assertTrue(numpy.array_equal(g2, g2t))
        self.assertTrue(numpy.array_equal(rdf2, rdf2t))
        pdfc.invertMask()
        r2ti, g2ti = pdfc(rutile)
        rdf2ti = pdfc.rdf
        self.assertTrue(numpy.array_equal(g2i, g2ti))
        self.assertTrue(numpy.array_equal(rdf2i, rdf2ti))
        # O-O
        pdfc.maskAllPairs(False)
        for i, smbli in enumerate(atomtypes):
            for j, smblj in enumerate(atomtypes):
                if smbli == smblj == "O":
                    pdfc.setPairMask(i, j, True)
        r3, g3 = pdfc(rutile)
        rdf3 = pdfc.rdf
        pdfc.invertMask()
        r3i, g3i = pdfc(rutile)
        rdf3i = pdfc.rdf
        self.assertTrue(numpy.allclose(g0, g3 + g3i))
        self.assertTrue(numpy.allclose(rdf0, rdf3 + rdf3i))
        # check the sum of all partials
        self.assertTrue(numpy.allclose(g0, g1 + g2 + g3))
        self.assertTrue(numpy.allclose(rdf0, rdf1 + rdf2 + rdf3))
        return

    def test_full_mask(self):
        '''Test PDFCalculator for a fully masked structure.
        '''
        pdfc = self.pdfcalc
        pdfc.rstep = 0.1
        rutile = self.tio2rutile
        pdfc.maskAllPairs(True)
        r0, g0 = pdfc(rutile)
        pdfc.maskAllPairs(False)
        r1, g1 = pdfc(rutile)
        self.assertEqual(0.0, numpy.dot(g1, g1))
        indices = range(len(rutile))
        pdfc.setPairMask(indices, indices, True)
        r2, g2 = pdfc(rutile)
        self.assertTrue(numpy.array_equal(g0, g2))
        return

    def test_zero_mask(self):
        '''Test PDFCalculator with a totally masked out structure.
        '''
        pdfc = self.pdfcalc
        pdfc.rstep = 0.1
        rutile = self.tio2rutile
        indices = range(len(rutile))
        for i in indices:
            for j in indices:
                pdfc.setPairMask(i, j, False)
        r, g = pdfc(rutile)
        self.assertEqual(0.0, numpy.dot(g, g))
        rdf = pdfc.rdf
        self.assertEqual(0.0, numpy.dot(rdf, rdf))
        return

    def test_pickling(self):
        '''check pickling and unpickling of PDFCalculator.
        '''
        pdfc = self.pdfcalc
        pdfc.scatteringfactortable = 'N'
        pdfc.scatteringfactortable.setCustomAs('Na', 'Na', 7)
        pdfc.addEnvelope('sphericalshape')
        pdfc.delta1 = 0.2
        pdfc.delta2 = 0.3
        pdfc.maxextension = 10.1
        pdfc.peakprecision = 5e-06
        pdfc.qbroad = 0.01
        pdfc.qdamp = 0.05
        pdfc.qmax = 10
        pdfc.qmin = 0.5
        pdfc.rmax = 10.0
        pdfc.rmin = 0.02
        pdfc.rstep = 0.02
        pdfc.scale = 1.1
        pdfc.slope = 0.1
        pdfc.spdiameter = 13.3
        pdfc.foobar = 'asdf'
        spkl = cPickle.dumps(pdfc)
        pdfc1 = cPickle.loads(spkl)
        sft = pdfc.scatteringfactortable
        sft1 = pdfc1.scatteringfactortable
        self.assertEqual(sft.type(), sft1.type())
        self.assertEqual(7.0, sft1.lookup('Na'))
        for a in pdfc._namesOfDoubleAttributes():
            self.assertEqual(getattr(pdfc, a), getattr(pdfc1, a))
        self.assertEqual(13.3,
                pdfc1.getEnvelope('sphericalshape').spdiameter)
        self.assertEqual(pdfc._namesOfDoubleAttributes(),
                pdfc1._namesOfDoubleAttributes())
        self.assertEqual(pdfc.usedenvelopetypes, pdfc1.usedenvelopetypes)
        self.assertEqual('asdf', pdfc1.foobar)
        return

    def test_mask_pickling(self):
        '''Check if mask gets properly pickled and restored.
        '''
        self.pdfcalc.maskAllPairs(False)
        self.pdfcalc.setPairMask(0, 1, True)
        self.assertTrue(False is self.pdfcalc.getPairMask(0, 0))
        self.assertTrue(True is self.pdfcalc.getPairMask(0, 1))
        pdfcalc1 = cPickle.loads(cPickle.dumps(self.pdfcalc))
        self.assertTrue(False is pdfcalc1.getPairMask(0, 0))
        self.assertTrue(True is pdfcalc1.getPairMask(0, 1))
        return

    def test_pickling_derived_structure(self):
        '''check pickling of PDFCalculator with DerivedStructureAdapter.
        '''
        from diffpy.srreal.tests.testutils import DerivedStructureAdapter
        pdfc = self.pdfcalc
        stru0 = DerivedStructureAdapter()
        pdfc.setStructure(stru0)
        self.assertEqual(1, stru0.cpqcount)
        spkl = cPickle.dumps(pdfc)
        pdfc1 = cPickle.loads(spkl)
        self.assertTrue(stru0 is pdfc.getStructure())
        stru1 = pdfc1.getStructure()
        self.assertTrue(type(stru1) is DerivedStructureAdapter)
        self.assertFalse(stru1 is stru0)
        self.assertEqual(1, stru1.cpqcount)
        return

    def test_envelopes(self):
        '''Check the envelopes property.
        '''
        from diffpy.srreal.pdfenvelope import PDFEnvelope
        pc = self.pdfcalc
        self.assertTrue(len(pc.envelopes) > 0)
        pc.clearEnvelopes()
        self.assertEqual(0, len(pc.envelopes))
        pc.addEnvelope(PDFEnvelope.createByType('scale'))
        self.assertEqual(1, len(pc.envelopes))
        self.assertEqual('scale', pc.envelopes[0].type())
        pc.envelopes += ('qresolution',)
        self.assertEqual(('qresolution', 'scale'), pc.usedenvelopetypes)
        self.assertTrue(all([isinstance(e, PDFEnvelope)
            for e in pc.envelopes]))
        return


#   def test_pdf(self):
#       """check PDFCalculator.pdf
#       """
#       return

#   def test_getRDF(self):
#       """check PDFCalculator.rdf
#       """
#       return

#   def test_getRadiationType(self):
#       """check PDFCalculator.getRadiationType()
#       """
#       return

#   def test_rgrid(self):
#       """check PDFCalculator.rgrid
#       """
#       return

#   def test_setScatteringFactorTable(self):
#       """check PDFCalculator.setScatteringFactorTable()
#       """
#       return

# End of class TestPDFCalculator

# ----------------------------------------------------------------------------

class TestFFTRoutines(unittest.TestCase):

    def test_fft_conversions(self):
        """Verify conversions of arguments in fftgtof function.
        """
        fnipf2 = datafile('Ni-fit.fgr')
        data = numpy.loadtxt(fnipf2)
        dr = 0.01
        fq0, dq0 = fftgtof(data[:,1], dr)
        fq1, dq1 = fftgtof(data[:,1].copy(), dr)
        fq2, dq2 = fftgtof(list(data[:,1]), dr)
        self.assertTrue(numpy.array_equal(fq0, fq1))
        self.assertTrue(numpy.array_equal(fq0, fq2))
        self.assertEqual(dq0, dq1)
        self.assertEqual(dq0, dq2)
        return


    def test_fft_roundtrip(self):
        """Check if forward and inverse transformation recover the input.
        """
        fnipf2 = datafile('Ni-fit.fgr')
        g0 = numpy.loadtxt(fnipf2, usecols=(1,))
        dr0 = 0.01
        fq, dq = fftgtof(g0, dr0)
        g1, dr1 = fftftog(fq, dq)
        self.assertAlmostEqual(dr0, dr1, 12)
        self.assertTrue(numpy.allclose(g0, g1[:g0.size]))
        return

# End of class TestFFTRoutines

# ----------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()

# End of file
