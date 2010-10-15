#!/usr/bin/env python

"""Unit tests for diffpy.srreal.structureadapter
"""

# version
__id__ = '$Id$'

import os
import unittest
import numpy
from diffpy.Structure import Structure
from diffpy.srreal.pdfcalculator import PDFCalculator

# useful variables
thisfile = locals().get('__file__', 'file.py')
tests_dir = os.path.dirname(os.path.abspath(thisfile))
testdata_dir = os.path.join(tests_dir, 'testdata')
nickel_stru = os.path.join(testdata_dir, 'Ni.stru')
nickel = Structure(filename=nickel_stru)

from diffpy.srreal.structureadapter import *

##############################################################################
class TestRoutines(unittest.TestCase):

    def test_createStructureAdapter(self):
        """check createStructureAdapter() routine.
        """
        adpt = createStructureAdapter(nickel)
        self.assertEqual(4, adpt.countSites())
        self.failUnless(False is adpt.siteAnisotropy(0))
        self.failUnless(StructureAdapter is type(adpt))
        adpt1 = createStructureAdapter(adpt)
        self.failUnless(adpt is adpt1)
        return

# End of class TestStructureAdapter


##############################################################################
class TestNoMeta(unittest.TestCase):

    def test_nometa(self):
        '''check NoMetaStructureAdapter.
        '''
        r0, g0 = PDFCalculator()(nickel)
        ni1 = Structure(nickel)
        ni1.pdffit['scale'] = 2.0
        r1, g1 = PDFCalculator()(ni1)
        self.failUnless(numpy.array_equal(r0, r1))
        self.failUnless(numpy.allclose(2 * g0, g1))
        ni1nm = nometa(ni1)
        self.failUnless(ni1nm is nometa(ni1nm))
        r1nm, g1nm = PDFCalculator()(ni1nm)
        self.failUnless(numpy.array_equal(r0, r1nm))
        self.failUnless(numpy.allclose(g0, g1nm))
        ni2 = Structure(nickel)
        ni2.pdffit['delta2'] = 4
        r2, g2 = PDFCalculator()(ni2)
        r2, g2nm = PDFCalculator()(nometa(ni2))
        self.failIf(numpy.allclose(g0, g2))
        self.failUnless(numpy.allclose(g0, g2nm))
        adpt2 = createStructureAdapter(ni2)
        ra2, ga2 = PDFCalculator()(adpt2)
        ra2, ga2nm = PDFCalculator()(nometa(adpt2))
        self.failUnless(numpy.allclose(g2, ga2))
        self.failUnless(numpy.allclose(g0, ga2nm))
        return
 
# End of class TestNoMeta


##############################################################################
class TestNoSymmetry(unittest.TestCase):

    def test_nosymmetry(self):
        '''check NoSymmetryStructureAdapter.
        '''
        pdfc0 = PDFCalculator()
        r0, g0 = pdfc0(nickel)
        rdf0 = pdfc0.getRDF()
        niuc = nosymmetry(nickel)
        self.failUnless(niuc is nosymmetry(niuc))
        pdfc1 = PDFCalculator()
        r1, g1 = pdfc1(niuc)
        self.failUnless(numpy.array_equal(r0, r1))
        self.failIf(numpy.allclose(g0, g1))
        tail = (r0 > 5.0)
        self.failUnless(numpy.allclose(0.0 * g1[tail], g1[tail]))
        rdf0 = pdfc0.getRDF()
        rdf1 = pdfc1.getRDF()
        head = r0 < 3.0
        self.assertAlmostEqual(12.0, numpy.sum(rdf0[head] * pdfc0.rstep), 5)
        self.assertAlmostEqual(3.0, numpy.sum(rdf1[head] * pdfc1.rstep), 5)
        adpt0 = createStructureAdapter(nickel)
        ra2, ga2 = PDFCalculator()(nosymmetry(adpt0))
        self.failUnless(numpy.array_equal(r0, ra2))
        self.failUnless(numpy.allclose(g1, ga2))
        return
 
# End of class TestNoSymmetry


##############################################################################
# class TestStructureAdapter(unittest.TestCase):
#
#   def setUp(self):
#       return
#
#   def tearDown(self):
#       return
#
#   def test___init__(self):
#       """check StructureAdapter.__init__()
#       """
#       return
#
#   def test___reduce__(self):
#       """check StructureAdapter.__reduce__()
#       """
#       return
#
#   def test__customPQConfig(self):
#       """check StructureAdapter._customPQConfig()
#       """
#       return
#
#   def test_countSites(self):
#       """check StructureAdapter.countSites()
#       """
#       return
#
#   def test_createBondGenerator(self):
#       """check StructureAdapter.createBondGenerator()
#       """
#       return
#
#   def test_numberDensity(self):
#       """check StructureAdapter.numberDensity()
#       """
#       return
#
#   def test_siteAnisotropy(self):
#       """check StructureAdapter.siteAnisotropy()
#       """
#       return
#
#   def test_siteAtomType(self):
#       """check StructureAdapter.siteAtomType()
#       """
#       return
#
#   def test_siteCartesianPosition(self):
#       """check StructureAdapter.siteCartesianPosition()
#       """
#       return
#
#   def test_siteCartesianUij(self):
#       """check StructureAdapter.siteCartesianUij()
#       """
#       return
#
#   def test_siteMultiplicity(self):
#       """check StructureAdapter.siteMultiplicity()
#       """
#       return
#
#   def test_siteOccupancy(self):
#       """check StructureAdapter.siteOccupancy()
#       """
#       return
#
#   def test_totalOccupancy(self):
#       """check StructureAdapter.totalOccupancy()
#       """
#       return

# End of class TestStructureAdapter

if __name__ == '__main__':
    unittest.main()

# End of file
