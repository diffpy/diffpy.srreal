#!/usr/bin/env python

"""Unit tests for pairhistogram.py
"""

# version
__id__ = '$Id$'

import os
import unittest
import math
import copy
import numpy

# useful variables
thisfile = locals().get('__file__', 'file.py')
tests_dir = os.path.dirname(os.path.abspath(thisfile))
testdata_dir = os.path.join(tests_dir, 'testdata')

from diffpy.srreal.pairhistogram import PairHistogram

##############################################################################
class TestPairHistogram(unittest.TestCase):

    _data_loaded = False
    rmax = 10
    rutile = None
    silicon = None
    sphalerite = None

    def setUp(self):
        """load test structures.
        """
        datanames = ("rutile", "silicon", "sphalerite")
        # load pair histograms
        if not self._data_loaded:
            from diffpy.Structure import Structure
            for name in datanames:
                basename = name + ".cif"
                stru = Structure(filename=testdata(basename))
                ph = PairHistogram(stru, self.rmax)
                setattr(TestPairHistogram, name, stru)
                setattr(TestPairHistogram, "ph_" + name, ph)
            TestPairHistogram._data_loaded = True
        # reset pair histograms to default values
        for name in datanames:
            ph = getattr(self, "ph_" + name)
            ph.setRmax(self.rmax)
            ph.setRadiationType('X')
        return

    def tearDown(self):
        return

    def test___init__(self):
        """check PairHistogram.__init__()
        """
        ph = PairHistogram(self.rutile, 15.0)
        self.assertEqual(len(self.rutile), len(ph.getStructure()))
        self.assertEqual(15.0, ph.getRmax())
        # check defaults
        self.assertEqual(True, ph.getPBC())
        self.assertEqual(PairHistogram.resolution, ph.getResolution())
        self.assertEqual("X", ph.getRadiationType())
        return
 
    def test_copy(self):
        """check shallow copy of PairHistogram
        """
        phsi = copy.copy(self.ph_silicon)
        self.assert_(phsi._structure is self.ph_silicon._structure)
        self.assert_(phsi._site_coloring is self.ph_silicon._site_coloring)
        self.assert_(phsi._x is self.ph_silicon._x)
        self.assert_(phsi._y is self.ph_silicon._y)
        return
 
    def test_x(self):
        """check PairHistogram.x()
        """
        x_rutile = self.ph_rutile.x()
        x_silicon = self.ph_silicon.x()
        x_sphalerite = self.ph_sphalerite.x()
        nnsi = math.sqrt(3)/4 * self.silicon.lattice.a
        self.assertAlmostEqual(nnsi, x_silicon[0])
        self.assert_(0 < x_rutile[0])
        self.assert_(0 < x_silicon[0])
        self.assert_(0 < x_sphalerite[0])
        self.assert_(x_rutile[-1] <= self.rmax)
        self.assert_(x_silicon[-1] <= self.rmax)
        self.assert_(x_sphalerite[-1] <= self.rmax)
        return
 
    def test_y(self):
        """check PairHistogram.y()
        """
        self.assertAlmostEqual(4, self.ph_silicon.y()[0], 6)
        fZn = self.ph_sphalerite.nmsf('Zn')
        fS = self.ph_sphalerite.nmsf('S')
        self.assertAlmostEqual(4*fZn*fS, self.ph_sphalerite.y()[0], 6)
        self.assert_(0 < min(self.ph_rutile.y()))
        self.assert_(0 < min(self.ph_silicon.y()))
        self.assert_(0 < min(self.ph_sphalerite.y()))
        return
 
    def test_countBars(self):
        """check PairHistogram.countBars()
        """
        phsi = self.ph_silicon
        self.assertEqual(13, phsi.countBars())
        phsi.setRmax(5)
        self.assertEqual(3, phsi.countBars())
        phsi.setRmax(self.rmax)
        self.assertEqual(13, phsi.countBars())
        return
 
    def test_countAtoms(self):
        """check PairHistogram.countAtoms()
        """
        self.assertEqual(6, self.ph_rutile.countAtoms())
        self.assertEqual(8, self.ph_silicon.countAtoms())
        self.assertEqual(8, self.ph_sphalerite.countAtoms())
        return
 
    def test_nmsf(self):
        """check PairHistogram.nmsf()
        """
        self.assertAlmostEqual(1.0, self.ph_silicon.nmsf('Si'), 6)
        fZn = self.ph_sphalerite.nmsf('Zn')
        fS = self.ph_sphalerite.nmsf('S')
        self.assert_(fS < fZn)
        self.assertAlmostEqual(1.0, (fZn + fS)/2, 6)
        phzns = copy.copy(self.ph_sphalerite)
        phzns.setSiteColoring(phzns.countAtoms() * ["C"])
        self.assertAlmostEqual(1.0, phzns.nmsf("C"))
        self.assertRaises(ValueError, phzns.nmsf, "Zn")
        # restore coloring
        phzns.setSiteColoring(self.ph_sphalerite.getSiteColoring())
        self.assertAlmostEqual(fZn, phzns.nmsf('Zn'))
        self.assertRaises(ValueError, phzns.nmsf, "C")
        return
 
    def test_meansf(self):
        """check PairHistogram.meansf()
        """
        self.assertAlmostEqual(14, self.ph_silicon.meansf(), 4)
        self.assertAlmostEqual((30 + 16)/2.0, self.ph_sphalerite.meansf(), 4)
        phsi = copy.copy(self.ph_silicon)
        phsi.setRadiationType('N')
        self.assertAlmostEqual(4.1507, phsi.meansf(), 2)
        # original should be untouched
        self.assertAlmostEqual(14, self.ph_silicon.meansf(), 4)
        return

    def test_setScatteringFactors(self,):
        """check PairHistogram.setScatteringFactors()
        """
        ph = copy.copy(self.ph_sphalerite)
        self.assertEqual(self.ph_sphalerite.y(), ph.y())
        fZn = self.ph_sphalerite.nmsf('Zn')
        fS = self.ph_sphalerite.nmsf('S')
        ph.setScatteringFactors({'Zn' : 0.0, 'S' : 1.0})
        self.assertEqual(0.0, ph.y()[0])
        self.assertEqual(0.0, ph.nmsf('Zn'))
        self.assertEqual(2.0, ph.nmsf('S'))
        # reset back to the defaults
        ph.setScatteringFactors({})
        self.assertEqual(self.ph_sphalerite.y(), ph.y())
        self.assertEqual(fZn, ph.nmsf('Zn'))
        self.assertEqual(fS, ph.nmsf('S'))
        return
 
    def test_setStructure(self):
        """check PairHistogram.setStructure()
        """
        ph = copy.copy(self.ph_rutile)
        ph.setStructure(self.silicon)
        self.assertEqual(self.ph_silicon.x(), ph.x())
        self.assertEqual(self.ph_silicon.y(), ph.y())
        ph.setStructure(self.sphalerite)
        self.assertEqual(self.ph_sphalerite.x(), ph.x())
        self.assertEqual(self.ph_sphalerite.y(), ph.y())
        ph.setStructure(self.rutile)
        self.assertEqual(self.ph_rutile.x(), ph.x())
        self.assertEqual(self.ph_rutile.y(), ph.y())
        return
 
    def test_getStructure(self):
        """check PairHistogram.getStructure()
        """
        si = self.ph_silicon.getStructure()
        self.failIf(si is self.ph_silicon._structure)
        self.assertEqual(len(self.silicon), len(si))
        self.assert_( numpy.all(self.silicon[0].xyz == si[0].xyz) )
        return
 
    def test_setSiteColoring(self):
        """check PairHistogram.setSiteColoring()
        """
        # argument checking
        self.assertRaises(ValueError,
                self.ph_sphalerite.setSiteColoring, [])
        self.assertRaises(ValueError,
                self.ph_sphalerite.setSiteColoring, 100*['C'])
        # returned values
        x0 = self.ph_sphalerite.x()
        y0 = self.ph_sphalerite.y()
        c0 = self.ph_sphalerite.getSiteColoring()
        ph = copy.copy(self.ph_sphalerite)
        self.assertEqual(x0, ph.x())
        self.assertEqual(y0, ph.y())
        c1 = 4 * ['Zn', 'S']
        ph.setSiteColoring(c1)
        self.assertEqual(c1, ph.getSiteColoring())
        yres = numpy.linalg.norm(numpy.array(y0) - ph.y())
        self.assert_(yres > 0.1)
        ph.setSiteColoring(c0)
        self.assertEqual(c0, ph.getSiteColoring())
        yres = numpy.linalg.norm(numpy.array(y0) - ph.y())
        self.assertAlmostEqual(0.0, yres, 8)
        return
 
    def test_getSiteColoring(self):
        """check PairHistogram.getSiteColoring()
        """
        self.assertEqual(2*['Ti'] + 4*['O'], self.ph_rutile.getSiteColoring())
        self.assertEqual(8*['Si'], self.ph_silicon.getSiteColoring())
        return
 
    def test_flipSiteColoring(self):
        """check PairHistogram.flipSiteColoring()
        """
        y0 = self.ph_sphalerite.y()
        c0 = self.ph_sphalerite.getSiteColoring()
        ph = copy.copy(self.ph_sphalerite)
        ph.flipSiteColoring(0, 1)
        self.assertEqual(c0, ph.getSiteColoring())
        self.assertEqual(y0, ph.y())
        ph.flipSiteColoring(0, 7)
        self.assertNotEqual(c0, ph.getSiteColoring())
        yres = numpy.linalg.norm(numpy.array(y0) - ph.y())
        self.assert_(yres > 0.1)
        ph.flipSiteColoring(0, 7)
        self.assertEqual(c0, ph.getSiteColoring())
        yres = numpy.linalg.norm(numpy.array(y0) - ph.y())
        self.assertAlmostEqual(0.0, yres)
        return
 
    def test_setRmax(self):
        """check PairHistogram.setRmax()
        """
        phsi = self.ph_silicon
        x0 = phsi.x()
        y0 = phsi.y()
        phsi.setRmax(0.01)
        self.assert_(len(x0) > 0)
        self.assertEqual(0, phsi.countBars())
        self.assertEqual([], phsi.x())
        self.assertEqual([], phsi.y())
        phsi.setRmax(self.rmax)
        self.assertEqual(x0, phsi.x())
        yres = numpy.linalg.norm(numpy.array(y0) - phsi.y())
        self.assertAlmostEqual(0.0, yres, 8)
        return
 
    def test_getRmax(self):
        """check PairHistogram.getRmax()
        """
        self.assertEqual(self.rmax, self.ph_silicon.getRmax())
        return
 
    def test_setPBC(self):
        """check PairHistogram.setPBC()
        """
        phsi = copy.copy(self.ph_silicon)
        x0 = phsi.x()
        y0 = phsi.y()
        phsi.setPBC(False)
        self.assertNotEqual(x0, phsi.x())
        self.assertNotEqual(y0, phsi.y())
        self.assertAlmostEqual(phsi.countAtoms() - 1, sum(phsi.y()), 8)
        phsi.setPBC(True)
        self.assertEqual(x0, phsi.x())
        self.assertEqual(y0, phsi.y())
        return
 
    def test_getPBC(self):
        """check PairHistogram.getPBC()
        """
        phsi = copy.copy(self.ph_silicon)
        self.assert_(True is phsi.getPBC())
        phsi.setPBC(False)
        self.assert_(False is phsi.getPBC())
        return
 
    def test_setResolution(self):
        """check PairHistogram.setResolution()
        """
        phsi = copy.copy(self.ph_silicon)
        x0 = phsi.x()
        y0 = phsi.y()
        phsi.setResolution(100)
        self.assertEqual(1, phsi.countBars())
        self.assertAlmostEqual(sum(y0), phsi.y()[0], 8)
        phsi.setResolution(PairHistogram.resolution)
        self.assertEqual(x0, phsi.x())
        self.assertEqual(y0, phsi.y())
        self.assertRaises(ValueError, phsi.setResolution, -10)
        return
 
    def test_getResolution(self):
        """check PairHistogram.getResolution()
        """
        phsi = copy.copy(self.ph_silicon)
        self.assertEqual(PairHistogram.resolution, phsi.getResolution())
        return
 
    def test_setRadiationType(self):
        """check PairHistogram.setRadiationType()
        """
        phru = copy.copy(self.ph_rutile)
        x0 = phru.x()
        y0 = phru.y()
        self.assert_(0 < min(phru.y()))
        phru.setRadiationType('N')
        self.assertEqual(x0, phru.x())
        self.assert_(0 > min(phru.y()))
        phru.setRadiationType('X')
        self.assertEqual(x0, phru.x())
        self.assertEqual(y0, phru.y())
        self.assertRaises(ValueError, phru.setRadiationType, 'z')
        return
 
    def test_getRadiationType(self):
        """check PairHistogram.getRadiationType()
        """
        phru = copy.copy(self.ph_rutile)
        self.assertEqual("X", phru.getRadiationType())
        phru.setRadiationType("N")
        self.assertEqual("N", phru.getRadiationType())
        return
 
#   def test__update_x(self):
#       """check PairHistogram._update_x()
#       """
#       return
#
#   def test__update_y(self):
#       """check PairHistogram._update_y()
#       """
#       return
#
#   def test__update_sf(self):
#       """check PairHistogram._update_sf()
#       """
#       return
#
#   def test__uncache(self):
#       """check PairHistogram._uncache()
#       """
#       return
#
#   def test__allPairsCluster(self):
#       """check PairHistogram._allPairsCluster()
#       """
#       return
#
#   def test__allPairsCrystal(self):
#       """check PairHistogram._allPairsCrystal()
#       """
#       return

# End of class TestPairHistogram


##############################################################################
# helper routines


def testdata(basename):
    """Prepend testdata_dir to the basename.
    """
    filename = os.path.join(testdata_dir, basename)
    return filename


if __name__ == '__main__':
    unittest.main()

# End of file
