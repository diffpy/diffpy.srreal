#!/usr/bin/env python

"""Unit tests for diffpy.srreal.structureadapter
"""

# version
__id__ = '$Id$'

import os
import unittest
import cPickle
import numpy
from diffpy.Structure import Structure
from diffpy.srreal.pdfcalculator import PDFCalculator
from srrealtestutils import TestCaseObjCrystOptional, loadObjCrystCrystal
from srrealtestutils import loadDiffPyStructure, resolveDataFile
from diffpy.srreal.structureadapter import *


# useful variables
nickel = loadDiffPyStructure('Ni.stru')
rutile_cif = resolveDataFile('TiO2_rutile-fit.cif')

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
        self.assertRaises(TypeError, createStructureAdapter, 77)
        self.assertRaises(TypeError, createStructureAdapter, range(8))
        self.assertRaises(TypeError, createStructureAdapter, None)
        self.assertRaises(TypeError, createStructureAdapter, {})
        return

    def test_pickling(self):
        '''check pickling of StructureAdapter instances.
        '''
        adpt = createStructureAdapter(nickel)
        adpt1 = cPickle.loads(cPickle.dumps(adpt))
        self.failIf(adpt is adpt1)
        self.assertEqual(adpt.countSites(), adpt1.countSites())
        self.assertEqual(adpt.totalOccupancy(), adpt1.totalOccupancy())
        self.assertEqual(adpt.siteAtomType(1), adpt1.siteAtomType(1))
        self.failUnless(numpy.array_equal(
            adpt.siteCartesianPosition(1), adpt1.siteCartesianPosition(1)))
        self.assertEqual(adpt.siteMultiplicity(1), adpt1.siteMultiplicity(1))
        self.assertEqual(adpt.siteOccupancy(1), adpt1.siteOccupancy(1))
        self.failUnless(adpt.siteAnisotropy(1) is adpt1.siteAnisotropy(1))
        self.failUnless(numpy.array_equal(
            adpt.siteCartesianUij(1), adpt1.siteCartesianUij(1)))
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

    def test_nometa_pickling(self):
        '''check pickling of the NoMetaStructureAdapter wrapper.
        '''
        r0, g0 = PDFCalculator()(nickel)
        ni1 = Structure(nickel)
        ni1.pdffit['scale'] = 2.0
        ni1nm = cPickle.loads(cPickle.dumps(nometa(ni1)))
        self.failIf(ni1nm is ni1)
        r1nm, g1nm = PDFCalculator()(ni1nm)
        self.failUnless(numpy.array_equal(r0, r1nm))
        self.failUnless(numpy.array_equal(g0, g1nm))
        return

# End of class TestNoMeta


##############################################################################
class TestNoSymmetry(unittest.TestCase):

    def test_nosymmetry(self):
        '''check NoSymmetryStructureAdapter.
        '''
        pdfc0 = PDFCalculator()
        r0, g0 = pdfc0(nickel)
        rdf0 = pdfc0.rdf
        niuc = nosymmetry(nickel)
        self.failUnless(niuc is nosymmetry(niuc))
        pdfc1 = PDFCalculator()
        r1, g1 = pdfc1(niuc)
        self.failUnless(numpy.array_equal(r0, r1))
        self.failIf(numpy.allclose(g0, g1))
        tail = (r0 > 5.0)
        self.failUnless(numpy.allclose(0.0 * g1[tail], g1[tail]))
        rdf0 = pdfc0.rdf
        rdf1 = pdfc1.rdf
        head = r0 < 3.0
        self.assertAlmostEqual(12.0, numpy.sum(rdf0[head] * pdfc0.rstep), 5)
        self.assertAlmostEqual(3.0, numpy.sum(rdf1[head] * pdfc1.rstep), 5)
        adpt0 = createStructureAdapter(nickel)
        ra2, ga2 = PDFCalculator()(nosymmetry(adpt0))
        self.failUnless(numpy.array_equal(r0, ra2))
        self.failUnless(numpy.allclose(g1, ga2))
        return

    def test_nosymmetry_pickling(self):
        '''check pickling of the NoSymmetryStructureAdapter wrapper.
        '''
        ni1ns = nosymmetry(nickel)
        r1, g1 = PDFCalculator()(ni1ns)
        ni2ns = cPickle.loads(cPickle.dumps(ni1ns))
        self.failIf(ni1ns is ni2ns)
        r2, g2 = PDFCalculator()(ni2ns)
        self.failUnless(numpy.array_equal(r1, r2))
        self.failUnless(numpy.array_equal(g1, g2))
        return

# End of class TestNoSymmetry


##############################################################################
class TestPyObjCrystAdapter(TestCaseObjCrystOptional):

    def setUp(self):
        rutile_crystal = loadObjCrystCrystal(rutile_cif)
        self.rutile = createStructureAdapter(rutile_crystal)
        return

    def test_objcryst_adapter(self):
        '''check ObjCrystStructureAdapter for rutile.
        '''
        self.assertEqual(2, self.rutile.countSites())
        self.assertEqual(6, self.rutile.totalOccupancy())
        self.assertEqual("Ti", self.rutile.siteAtomType(0))
        self.assertEqual("O", self.rutile.siteAtomType(1))
        self.failUnless(True is self.rutile.siteAnisotropy(0))
        self.failUnless(True is self.rutile.siteAnisotropy(1))
        self.failUnless(numpy.allclose(
            numpy.diag([0.008698, 0.008698, 0.005492]),
            self.rutile.siteCartesianUij(0)))
        self.failUnless(numpy.allclose(
            numpy.diag([0.021733, 0.021733, 0.007707]),
            self.rutile.siteCartesianUij(1)))
        return

    def test_objcryst_pickling(self):
        '''check pickling of the NoSymmetryStructureAdapter wrapper.
        '''
        r0, g0 = PDFCalculator()(self.rutile)
        rutile1 = cPickle.loads(cPickle.dumps(self.rutile))
        self.failIf(self.rutile is rutile1)
        r1, g1 = PDFCalculator()(rutile1)
        self.failUnless(numpy.array_equal(r0, r1))
        self.failUnless(numpy.array_equal(g0, g1))
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
