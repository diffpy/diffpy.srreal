#!/usr/bin/env python

"""Unit tests for the PeakProfile classes from diffpy.srreal.peakprofile
"""


import unittest
import cPickle

from diffpy.srreal.peakprofile import PeakProfile
from diffpy.srreal.pdfcalculator import PDFCalculator

##############################################################################
class TestPeakProfile(unittest.TestCase):

    tio2stru = None
    tio2adpt = None

    def setUp(self):
        self.pkgauss = PeakProfile.createByType('gaussian')
        self.pkcropped = PeakProfile.createByType('croppedgaussian')
        return


    def tearDown(self):
        return


    def test___init__(self):
        """check PeakProfile.__init__()
        """
        self.assertNotEqual(0.0, self.pkgauss.peakprecision)
        self.assertEqual(self.pkgauss.peakprecision,
                self.pkcropped.peakprecision)
        self.pkgauss._setDoubleAttr('peakprecision', 0.01)
        self.assertEqual(0.01, self.pkgauss.peakprecision)
        return


    def test_create(self):
        """check PeakProfile.create
        """
        # this is a virtual method in the base class
        self.assertRaises(RuntimeError, PeakProfile().create)
        self.assertEqual('gaussian', self.pkgauss.create().type())
        self.pkgauss.peakprecision = 0.007
        self.assertNotEqual(0.007, self.pkgauss.create().peakprecision)
        return


    def test_clone(self):
        """check PeakProfile.clone
        """
        # this is a virtual method in the base class
        self.assertRaises(RuntimeError, PeakProfile().clone)
        self.pkgauss.peakprecision = 0.0003
        pkg2 = self.pkgauss.clone()
        self.assertEqual('gaussian', pkg2.type())
        self.assertEqual(0.0003, pkg2.peakprecision)
        self.assertEqual(0.0003, pkg2._getDoubleAttr('peakprecision'))
        return


    def test_type(self):
        """check PeakProfile.type
        """
        # this is a virtual method in the base class
        self.assertRaises(RuntimeError, PeakProfile().type)
        self.assertEqual('croppedgaussian', self.pkcropped.type())
        return


    def test___call__(self):
        """check PeakProfile.__call__()
        """
        ymx = self.pkgauss(0.0, 1)
        yhalflo = self.pkgauss(-0.5, 1)
        yhalfhi = self.pkgauss(-0.5, 1)
        self.assertAlmostEqual(ymx, 2 * yhalflo)
        self.assertAlmostEqual(ymx, 2 * yhalfhi)
        self.assertEqual(0, self.pkcropped(10, 1))
        self.assertNotEqual(0, self.pkgauss(10, 1))
        return


    def test_ticker(self):
        """check PeakProfile.ticker()
        """
        from diffpy.srreal.eventticker import EventTicker
        et0 = EventTicker(self.pkgauss.ticker())
        self.pkgauss.peakprecision = 0.003
        et1 = self.pkgauss.ticker()
        self.assertNotEqual(et0, et1)
        self.assertTrue(et0 < et1)
        return


    def test_ticker_override(self):
        """check method override for PeakProfile.ticker in a derived class.
        """
        pkf = MySawTooth()
        self.assertEqual(0, pkf.tcnt)
        et0 = pkf.ticker()
        self.assertEqual(1, pkf.tcnt)
        et1 = PeakProfile.ticker(pkf)
        self.assertEqual(1, pkf.tcnt)
        self.assertEqual(et0, et1)
        et0.click()
        self.assertEqual(et0, et1)
        # check that implicit ticker call from PDFCalculator is
        # handled by the Python ticker override.
        pc = PDFCalculator()
        pc.peakprofile = pkf
        pc.ticker()
        self.assertEqual(2, pkf.tcnt)
        return


    def test_getRegisteredTypes(self):
        """check PeakProfile.getRegisteredTypes
        """
        regtypes = PeakProfile.getRegisteredTypes()
        self.assertTrue(2 <= len(regtypes))
        self.assertTrue(regtypes.issuperset(
            ['gaussian', 'croppedgaussian']))
        return


    def test_pickling(self):
        '''check pickling and unpickling of PeakProfile.
        '''
        pkg = self.pkgauss
        pkg.peakprecision = 0.0011
        pkg2 = cPickle.loads(cPickle.dumps(pkg))
        self.assertEqual('gaussian', pkg2.type())
        self.assertEqual(0.0011, pkg2.peakprecision)
        self.assertEqual(0.0011, pkg2._getDoubleAttr('peakprecision'))
        return

# ----------------------------------------------------------------------------

class MySawTooth(PeakProfile):
    "Helper class for testing PeakProfile."

    def type(self):
        return "mysawtooth"

    def create(self):
        return MySawTooth()

    def clone(self):
        import copy
        return copy.copy(self)

    tcnt = 0
    def ticker(self):
        self.tcnt += 1
        return PeakProfile.ticker(self)

    def __call__(self, x, fwhm):
        w = 1.0 * fwhm
        rv = (1 - abs(x) / w) / (1.0 * w)
        if rv < 0:  rv = 0
        return rv

MySawTooth()._registerThisType()

# End of class MySawTooth

class TestPeakProfileOwner(unittest.TestCase):

    def setUp(self):
        self.pc = PDFCalculator()
        self.pkf = MySawTooth()
        self.pkf.peakprecision = 0.0017
        self.pc.peakprofile = self.pkf
        return


    def test_pkftype(self):
        '''Check type of the owned PeakProfile instance.
        '''
        self.assertEqual('mysawtooth', self.pc.peakprofile.type())
        return


    def test_pickling(self):
        '''Check pickling of an owned PeakProfile instance.
        '''
        pc1 = cPickle.loads(cPickle.dumps(self.pc))
        self.pkf.peakprecision = 0.0003
        pc2 = cPickle.loads(cPickle.dumps(self.pc))
        self.assertEqual('mysawtooth', pc1.peakprofile.type())
        self.assertEqual(0.0017, pc1.peakprofile.peakprecision)
        self.assertEqual(0.0017, pc1.peakprecision)
        self.assertEqual('mysawtooth', pc2.peakprofile.type())
        self.assertEqual(0.0003, pc2.peakprofile.peakprecision)
        self.assertEqual(0.0003, pc2.peakprecision)
        return

# ----------------------------------------------------------------------------


if __name__ == '__main__':
    unittest.main()

# End of file
