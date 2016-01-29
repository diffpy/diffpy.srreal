#!/usr/bin/env python

"""Unit tests for diffpy.srreal.structureadapter
"""


import unittest
import cPickle
import numpy
from diffpy.Structure import Structure
from diffpy.srreal.pdfcalculator import PDFCalculator
from diffpy.srreal.tests.testutils import TestCaseObjCrystOptional
from diffpy.srreal.tests.testutils import loadObjCrystCrystal
from diffpy.srreal.tests.testutils import loadDiffPyStructure
from diffpy.srreal.structureadapter import (
        createStructureAdapter, nometa, nosymmetry, StructureAdapter,
        AtomicStructureAdapter, Atom, PeriodicStructureAdapter)
import diffpy.srreal.tests.testutils as testutils


# useful variables
nickel = loadDiffPyStructure('Ni.stru')
rutile_cif = 'TiO2_rutile-fit.cif'

##############################################################################
class TestRoutines(unittest.TestCase):

    def test_createStructureAdapter(self):
        """check createStructureAdapter() routine.
        """
        adpt = createStructureAdapter(nickel)
        self.assertEqual(4, adpt.countSites())
        self.failUnless(False is adpt.siteAnisotropy(0))
        self.failUnless(isinstance(adpt, StructureAdapter))
        adpt1 = createStructureAdapter(adpt)
        self.failUnless(adpt is adpt1)
        self.assertRaises(TypeError, createStructureAdapter, 77)
        self.assertRaises(TypeError, createStructureAdapter, range(8))
        self.assertRaises(TypeError, createStructureAdapter, None)
        self.assertRaises(TypeError, createStructureAdapter, {})
        return

    def test_createStructureAdapterTypes(self):
        '''Check types returned by conversion from diffpy.Structure.
        '''
        from diffpy.srreal.structureconverters import (
            DiffPyStructureAtomicAdapter,
            DiffPyStructurePeriodicAdapter)
        adpt = createStructureAdapter(nickel)
        self.failUnless(type(adpt) is DiffPyStructurePeriodicAdapter)
        nickel.pdffit = None
        adpt1 = createStructureAdapter(nickel)
        self.failUnless(type(adpt1) is PeriodicStructureAdapter)
        nickel.lattice.setLatPar(1, 1, 1, 90, 90, 90)
        adpt2 = createStructureAdapter(nickel)
        self.failUnless(type(adpt2) is AtomicStructureAdapter)
        nickel.pdffit = dict(scale=1)
        adpt3 = createStructureAdapter(nickel)
        self.failUnless(type(adpt3) is DiffPyStructureAtomicAdapter)
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

    def test_pickle_nonwrapped(self):
        '''Check if pickling works for non-wrapped C++ object.
        '''
        from diffpy.srreal.structureadapter import EMPTY as e0
        spkl = cPickle.dumps(e0)
        e1 = cPickle.loads(spkl)
        self.assertEqual(0, e1.countSites())
        return

# End of class TestStructureAdapter


##############################################################################
class TestDerivedAdapter(unittest.TestCase):
    'Check functionality in a Python-derived StructureAdapter class.'

    DerivedCls = testutils.DerivedStructureAdapter

    def setUp(self):
        self.adpt = self.DerivedCls()
        return

    def test__customPQConfig(self):
        """check if DerivedCls._customPQConfig gets called.
        """
        self.assertEqual(0, self.adpt.cpqcount)
        pc = PDFCalculator()
        pc.setStructure(self.adpt)
        self.assertEqual(1, self.adpt.cpqcount)
        pc(self.adpt)
        self.assertEqual(2, self.adpt.cpqcount)
        return

    def test_pickling(self):
        '''check pickling of DerivedCls instances.
        '''
        self.adpt.cpqcount = 1
        adpt1 = cPickle.loads(cPickle.dumps(self.adpt))
        self.failUnless(self.DerivedCls is type(adpt1))
        self.failIf(self.adpt is adpt1)
        self.assertEqual(1, adpt1.cpqcount)
        pc = PDFCalculator()
        pc.setStructure(adpt1)
        self.assertEqual(2, adpt1.cpqcount)
        pc(adpt1)
        self.assertEqual(3, adpt1.cpqcount)
        return

# End of class TestDerivedAdapter

class TestDerivedAtomicAdapter(TestDerivedAdapter):
    DerivedCls = testutils.DerivedAtomicStructureAdapter

class TestDerivedPeriodicAdapter(TestDerivedAdapter):
    DerivedCls = testutils.DerivedPeriodicStructureAdapter

class TestDerivedCrystalAdapter(TestDerivedAdapter):
    DerivedCls = testutils.DerivedCrystalStructureAdapter


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

    def test_nometa_twice(self):
        '''check that second call of nometa returns the same object.
        '''
        adpt1 = nometa(nickel)
        adpt2 = nometa(adpt1)
        self.failUnless(adpt1 is adpt2)

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

    def test_nosymmetry_twice(self):
        '''check that second call of nosymmetry returns the same object.
        '''
        adpt1 = nosymmetry(nickel)
        adpt2 = nosymmetry(adpt1)
        self.failUnless(adpt1 is adpt2)

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

##############################################################################
class TestAtom(unittest.TestCase):

    def setUp(self):
        self.a = Atom()
        return

    def test___init__copy(self):
        '''check Atom copy constructor.
        '''
        self.a.xyz_cartn = (1, 2, 3)
        a1 = Atom(self.a)
        self.assertEqual(self.a, a1)
        self.assertNotEqual(self.a, Atom())
        return

    def test_equality(self):
        '''check Atom equal and not equal operators.
        '''
        self.assertEqual(self.a, Atom())
        self.assertFalse(self.a != Atom())
        a1 = Atom()
        a1.atomtype = 'Na'
        self.assertNotEqual(self.a, a1)
        return

    def test_pickling(self):
        '''check pickling of Atom instances.
        '''
        self.a.atomtype = "Na"
        a1 = cPickle.loads(cPickle.dumps(self.a))
        self.assertEqual("Na", a1.atomtype)
        self.assertEqual(self.a, a1)
        self.failIf(self.a is a1)
        return

    def test_xyz_cartn(self):
        '''check Atom.xyz_cartn.
        '''
        a = self.a
        a.xyz_cartn = 4, 5, 6
        self.assertTrue(numpy.array_equal([4, 5, 6], a.xyz_cartn))
        self.assertEqual(4.0, a.xc)
        self.assertEqual(5.0, a.yc)
        self.assertEqual(6.0, a.zc)
        return

    def test_uij_cartn(self):
        '''check Atom.uij_cartn
        '''
        a = self.a
        a.uij_cartn = numpy.identity(3) * 0.01
        a.uc12 = 0.012
        a.uc13 = 0.013
        a.uc23 = 0.023
        self.assertTrue(numpy.array_equal(a.uij_cartn, [
            [0.01, 0.012, 0.013],
            [0.012, 0.01, 0.023],
            [0.013, 0.023, 0.01]]))
        self.assertEqual(0.01, a.uc11)
        self.assertEqual(0.01, a.uc22)
        self.assertEqual(0.01, a.uc33)
        self.assertEqual(0.012, a.uc12)
        self.assertEqual(0.013, a.uc13)
        self.assertEqual(0.023, a.uc23)
        return

# End of class TestAtom

if __name__ == '__main__':
    unittest.main()

# End of file
