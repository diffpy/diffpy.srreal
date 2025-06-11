#!/usr/bin/env python

"""Unit tests for diffpy.srreal.structureadapter."""


import pickle
import unittest

import numpy

import diffpy.srreal.tests.testutils as testutils
from diffpy.srreal.pdfcalculator import PDFCalculator
from diffpy.srreal.structureadapter import (
    Atom,
    AtomicStructureAdapter,
    CrystalStructureAdapter,
    PeriodicStructureAdapter,
    StructureAdapter,
    createStructureAdapter,
    nometa,
    nosymmetry,
)
from diffpy.srreal.tests.testutils import (
    _msg_nopyobjcryst,
    has_pyobjcryst,
    loadCrystalStructureAdapter,
    loadDiffPyStructure,
    loadObjCrystCrystal,
)

# ----------------------------------------------------------------------------


class TestRoutines(unittest.TestCase):

    def setUp(self):
        self.nickel = loadDiffPyStructure("Ni.stru")
        return

    def test_createStructureAdapter(self):
        """Check createStructureAdapter() routine."""
        adpt = createStructureAdapter(self.nickel)
        self.assertEqual(4, adpt.countSites())
        self.assertTrue(False is adpt.siteAnisotropy(0))
        self.assertTrue(isinstance(adpt, StructureAdapter))
        adpt1 = createStructureAdapter(adpt)
        self.assertTrue(adpt is adpt1)
        self.assertRaises(TypeError, createStructureAdapter, 77)
        self.assertRaises(TypeError, createStructureAdapter, list(range(8)))
        self.assertRaises(TypeError, createStructureAdapter, None)
        self.assertRaises(TypeError, createStructureAdapter, {})
        return

    def test_createStructureAdapterTypes(self):
        """Check types returned by conversion from diffpy.structure."""
        from diffpy.srreal.structureconverters import (
            DiffPyStructureAtomicAdapter,
            DiffPyStructurePeriodicAdapter,
        )

        adpt = createStructureAdapter(self.nickel)
        self.assertTrue(type(adpt) is DiffPyStructurePeriodicAdapter)
        self.nickel.pdffit = None
        adpt1 = createStructureAdapter(self.nickel)
        self.assertTrue(type(adpt1) is PeriodicStructureAdapter)
        self.nickel.lattice.setLatPar(1, 1, 1, 90, 90, 90)
        adpt2 = createStructureAdapter(self.nickel)
        self.assertTrue(type(adpt2) is AtomicStructureAdapter)
        self.nickel.pdffit = dict(scale=1)
        adpt3 = createStructureAdapter(self.nickel)
        self.assertTrue(type(adpt3) is DiffPyStructureAtomicAdapter)
        return

    def test_createStructureAdapter_int64_occupancy(self):
        """Check Structure conversion when occupany is of numpy.int64 type."""
        self.nickel[0].occupancy = numpy.int64(0)
        self.nickel[1].occupancy = numpy.int64(1)
        adpt = createStructureAdapter(self.nickel)
        self.assertEqual(0.0, adpt.siteOccupancy(0))
        self.assertEqual(1.0, adpt.siteOccupancy(1))
        return

    def test_pickling(self):
        """Check pickling of StructureAdapter instances."""
        adpt = createStructureAdapter(self.nickel)
        adpt1 = pickle.loads(pickle.dumps(adpt))
        self.assertFalse(adpt is adpt1)
        self.assertEqual(adpt.countSites(), adpt1.countSites())
        self.assertEqual(adpt.totalOccupancy(), adpt1.totalOccupancy())
        self.assertEqual(adpt.siteAtomType(1), adpt1.siteAtomType(1))
        self.assertTrue(
            numpy.array_equal(
                adpt.siteCartesianPosition(1), adpt1.siteCartesianPosition(1)
            )
        )
        self.assertEqual(adpt.siteMultiplicity(1), adpt1.siteMultiplicity(1))
        self.assertEqual(adpt.siteOccupancy(1), adpt1.siteOccupancy(1))
        self.assertTrue(adpt.siteAnisotropy(1) is adpt1.siteAnisotropy(1))
        self.assertTrue(
            numpy.array_equal(adpt.siteCartesianUij(1), adpt1.siteCartesianUij(1))
        )
        return

    def test_pickle_nonwrapped(self):
        """Check if pickling works for non-wrapped C++ object."""
        from diffpy.srreal.structureadapter import EMPTY as e0

        spkl = pickle.dumps(e0)
        e1 = pickle.loads(spkl)
        self.assertEqual(0, e1.countSites())
        return


# End of class TestStructureAdapter

# ----------------------------------------------------------------------------


class TestDerivedAdapter(unittest.TestCase):
    "Check functionality in a Python-derived StructureAdapter class."

    DerivedCls = testutils.DerivedStructureAdapter

    def setUp(self):
        self.adpt = self.DerivedCls()
        return

    def test__customPQConfig(self):
        """Check if DerivedCls._customPQConfig gets called."""
        self.assertEqual(0, self.adpt.cpqcount)
        pc = PDFCalculator()
        pc.setStructure(self.adpt)
        self.assertEqual(1, self.adpt.cpqcount)
        pc(self.adpt)
        self.assertEqual(2, self.adpt.cpqcount)
        return

    def test_pickling(self):
        """Check pickling of DerivedCls instances."""
        self.adpt.cpqcount = 1
        adpt1 = pickle.loads(pickle.dumps(self.adpt))
        self.assertTrue(self.DerivedCls is type(adpt1))
        self.assertFalse(self.adpt is adpt1)
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


# ----------------------------------------------------------------------------


class TestNoMeta(unittest.TestCase):

    def setUp(self):
        self.nickel = loadDiffPyStructure("Ni.stru")
        return

    def test_nometa(self):
        """Check NoMetaStructureAdapter."""
        r0, g0 = PDFCalculator()(self.nickel)
        ni1 = self.nickel.copy()
        ni1.pdffit["scale"] = 2.0
        r1, g1 = PDFCalculator()(ni1)
        self.assertTrue(numpy.array_equal(r0, r1))
        self.assertTrue(numpy.allclose(2 * g0, g1))
        ni1nm = nometa(ni1)
        self.assertTrue(ni1nm is nometa(ni1nm))
        r1nm, g1nm = PDFCalculator()(ni1nm)
        self.assertTrue(numpy.array_equal(r0, r1nm))
        self.assertTrue(numpy.allclose(g0, g1nm))
        ni2 = self.nickel.copy()
        ni2.pdffit["delta2"] = 4
        r2, g2 = PDFCalculator()(ni2)
        r2, g2nm = PDFCalculator()(nometa(ni2))
        self.assertFalse(numpy.allclose(g0, g2))
        self.assertTrue(numpy.allclose(g0, g2nm))
        adpt2 = createStructureAdapter(ni2)
        ra2, ga2 = PDFCalculator()(adpt2)
        ra2, ga2nm = PDFCalculator()(nometa(adpt2))
        self.assertTrue(numpy.allclose(g2, ga2))
        self.assertTrue(numpy.allclose(g0, ga2nm))
        return

    def test_nometa_pickling(self):
        """Check pickling of the NoMetaStructureAdapter wrapper."""
        r0, g0 = PDFCalculator()(self.nickel)
        ni1 = self.nickel.copy()
        ni1.pdffit["scale"] = 2.0
        ni1nm = pickle.loads(pickle.dumps(nometa(ni1)))
        self.assertFalse(ni1nm is ni1)
        r1nm, g1nm = PDFCalculator()(ni1nm)
        self.assertTrue(numpy.array_equal(r0, r1nm))
        self.assertTrue(numpy.array_equal(g0, g1nm))
        return

    def test_nometa_twice(self):
        """Check that second call of nometa returns the same object."""
        adpt1 = nometa(self.nickel)
        adpt2 = nometa(adpt1)
        self.assertTrue(adpt1 is adpt2)


# End of class TestNoMeta

# ----------------------------------------------------------------------------


class TestNoSymmetry(unittest.TestCase):

    def setUp(self):
        self.nickel = loadDiffPyStructure("Ni.stru")
        return

    def test_nosymmetry(self):
        """Check NoSymmetryStructureAdapter."""
        pdfc0 = PDFCalculator()
        r0, g0 = pdfc0(self.nickel)
        rdf0 = pdfc0.rdf
        niuc = nosymmetry(self.nickel)
        self.assertTrue(niuc is nosymmetry(niuc))
        pdfc1 = PDFCalculator()
        r1, g1 = pdfc1(niuc)
        self.assertTrue(numpy.array_equal(r0, r1))
        self.assertFalse(numpy.allclose(g0, g1))
        tail = r0 > 5.0
        self.assertTrue(numpy.allclose(0.0 * g1[tail], g1[tail]))
        rdf0 = pdfc0.rdf
        rdf1 = pdfc1.rdf
        head = r0 < 3.0
        self.assertAlmostEqual(12.0, numpy.sum(rdf0[head] * pdfc0.rstep), 5)
        self.assertAlmostEqual(3.0, numpy.sum(rdf1[head] * pdfc1.rstep), 5)
        adpt0 = createStructureAdapter(self.nickel)
        ra2, ga2 = PDFCalculator()(nosymmetry(adpt0))
        self.assertTrue(numpy.array_equal(r0, ra2))
        self.assertTrue(numpy.allclose(g1, ga2))
        return

    def test_nosymmetry_twice(self):
        """Check that second call of nosymmetry returns the same object."""
        adpt1 = nosymmetry(self.nickel)
        adpt2 = nosymmetry(adpt1)
        self.assertTrue(adpt1 is adpt2)

    def test_nosymmetry_pickling(self):
        """Check pickling of the NoSymmetryStructureAdapter wrapper."""
        ni1ns = nosymmetry(self.nickel)
        r1, g1 = PDFCalculator()(ni1ns)
        ni2ns = pickle.loads(pickle.dumps(ni1ns))
        self.assertFalse(ni1ns is ni2ns)
        r2, g2 = PDFCalculator()(ni2ns)
        self.assertTrue(numpy.array_equal(r1, r2))
        self.assertTrue(numpy.array_equal(g1, g2))
        return


# End of class TestNoSymmetry

# ----------------------------------------------------------------------------


@unittest.skipUnless(has_pyobjcryst, _msg_nopyobjcryst)
class TestPyObjCrystAdapter(unittest.TestCase):

    def setUp(self):
        rutile_crystal = loadObjCrystCrystal("TiO2_rutile-fit.cif")
        self.rutile = createStructureAdapter(rutile_crystal)
        return

    def test_objcryst_adapter(self):
        """Check ObjCrystStructureAdapter for rutile."""
        self.assertEqual(2, self.rutile.countSites())
        self.assertEqual(6, self.rutile.totalOccupancy())
        self.assertEqual("Ti", self.rutile.siteAtomType(0))
        self.assertEqual("O", self.rutile.siteAtomType(1))
        self.assertTrue(True is self.rutile.siteAnisotropy(0))
        self.assertTrue(True is self.rutile.siteAnisotropy(1))
        self.assertTrue(
            numpy.allclose(
                numpy.diag([0.008698, 0.008698, 0.005492]),
                self.rutile.siteCartesianUij(0),
            )
        )
        self.assertTrue(
            numpy.allclose(
                numpy.diag([0.021733, 0.021733, 0.007707]),
                self.rutile.siteCartesianUij(1),
            )
        )
        return

    def test_objcryst_pickling(self):
        """Check pickling of the NoSymmetryStructureAdapter wrapper."""
        r0, g0 = PDFCalculator()(self.rutile)
        rutile1 = pickle.loads(pickle.dumps(self.rutile))
        self.assertFalse(self.rutile is rutile1)
        r1, g1 = PDFCalculator()(rutile1)
        self.assertTrue(numpy.array_equal(r0, r1))
        self.assertTrue(numpy.array_equal(g0, g1))
        return


# End of class TestPyObjCrystAdapter

# ----------------------------------------------------------------------------


class IndexRangeTests(object):
    "Check error handling for site index arguments."

    AdptClass = None

    def setUp(self):
        self.adpt = self.AdptClass()
        return

    def test_siteAtomTypeIndex(self):
        """Check out-of-range arguments in AdptClass.siteAtomType."""
        cnt = self.adpt.countSites()
        self.assertRaises(IndexError, self.adpt.siteAtomType, cnt)
        self.assertRaises(IndexError, self.adpt.siteAtomType, -1)
        return

    def test_siteCartesianPositionIndex(self):
        """Check out-of-range arguments in AdptClass.siteCartesianPosition."""
        cnt = self.adpt.countSites()
        self.assertRaises(IndexError, self.adpt.siteCartesianPosition, cnt)
        self.assertRaises(IndexError, self.adpt.siteCartesianPosition, -1)
        return

    def test_siteMultiplicityIndex(self):
        """Check out-of-range arguments in AdptClass.siteMultiplicity."""
        cnt = self.adpt.countSites()
        self.assertRaises(IndexError, self.adpt.siteMultiplicity, cnt)
        self.assertRaises(IndexError, self.adpt.siteMultiplicity, -1)
        return

    def test_siteOccupancyIndex(self):
        """Check out-of-range arguments in AdptClass.siteOccupancy."""
        cnt = self.adpt.countSites()
        self.assertRaises(IndexError, self.adpt.siteOccupancy, cnt)
        self.assertRaises(IndexError, self.adpt.siteOccupancy, -1)
        return

    def test_siteAnisotropyIndex(self):
        """Check out-of-range arguments in AdptClass.siteAnisotropy."""
        cnt = self.adpt.countSites()
        self.assertRaises(IndexError, self.adpt.siteAnisotropy, cnt)
        self.assertRaises(IndexError, self.adpt.siteAnisotropy, -1)
        return

    def test_siteCartesianUijIndex(self):
        """Check out-of-range arguments in AdptClass.siteCartesianUij."""
        cnt = self.adpt.countSites()
        self.assertRaises(IndexError, self.adpt.siteCartesianUij, cnt)
        self.assertRaises(IndexError, self.adpt.siteCartesianUij, -1)
        return


# End of class IndexRangeTests

TestCase = unittest.TestCase

# test index bounds for C++ classes


class TestAtomicAdapterIndexRange(IndexRangeTests, TestCase):
    AdptClass = AtomicStructureAdapter


# No need to do index tests for PeriodicStructureAdapter as it does not
# override any of AtomicStructureAdapter site-access methods.
# CrystalStructureAdapter overrides siteMultiplicity so we'll
# test it here as well.


class TestCrystalAdapter(IndexRangeTests, TestCase):
    AdptClass = CrystalStructureAdapter

    def test_expandLatticeAtom(self):
        """Check CrystalStructureAdapter.expandLatticeAtom."""
        cdse = loadCrystalStructureAdapter("CdSe_cadmoselite.cif")
        a = Atom()
        a.xyz_cartn = (0.1, 0.13, 0.17)
        asymsites = cdse.expandLatticeAtom(a)
        self.assertTrue(type(asymsites) is list)
        self.assertEqual(12, len(asymsites))
        self.assertEqual(12, cdse.countSymOps())
        return

    def test_getEquivalentAtoms(self):
        """Check CrystalStructureAdapter.getEquivalentAtoms."""
        cdse = loadCrystalStructureAdapter("CdSe_cadmoselite.cif")
        eqatoms0 = cdse.getEquivalentAtoms(0)
        eqatoms1 = cdse.getEquivalentAtoms(1)
        self.assertTrue(type(eqatoms1) is list)
        self.assertEqual(2, len(eqatoms0))
        self.assertEqual(2, len(eqatoms1))
        self.assertTrue(all(a.atomtype == "Cd" for a in eqatoms0))
        self.assertTrue(all(a.atomtype == "Se" for a in eqatoms1))
        return


# ----------------------------------------------------------------------------


class TestDerivedStructureAdapter(IndexRangeTests, TestCase):
    AdptClass = testutils.DerivedStructureAdapter

    def setUp(self):
        IndexRangeTests.setUp(self)
        self.adpt1 = self.adpt.clone()
        self.adpt1.positions.append(numpy.array([1.0, 2.0, 3.0]))
        return

    def test_siteAtomType_valid(self):
        """Check DerivedStructureAdapter.siteAtomType."""
        adpt1 = self.adpt1
        self.assertEqual("Cu", adpt1.siteAtomType(0))
        self.assertEqual("", StructureAdapter.siteAtomType(adpt1, 0))
        return

    def test_siteCartesianPosition_valid(self):
        """Check DerivedStructureAdapter.siteCartesianPosition."""
        adpt1 = self.adpt1
        xyz0 = adpt1.siteCartesianPosition(0)
        self.assertTrue(numpy.array_equal([1, 2, 3], xyz0))
        return

    def test_siteMultiplicity_valid(self):
        """Check DerivedStructureAdapter.siteMultiplicity."""
        adpt1 = self.adpt1
        self.assertEqual(2, adpt1.siteMultiplicity(0))
        self.assertEqual(1, StructureAdapter.siteMultiplicity(adpt1, 0))
        return

    def test_siteOccupancy_valid(self):
        """Check DerivedStructureAdapter.siteOccupancy."""
        adpt1 = self.adpt1
        self.assertEqual(0.5, adpt1.siteOccupancy(0))
        self.assertEqual(1.0, StructureAdapter.siteOccupancy(adpt1, 0))
        return

    def test_siteAnisotropy_valid(self):
        """Check DerivedStructureAdapter.siteAnisotropy."""
        adpt1 = self.adpt1
        self.assertFalse(adpt1.siteAnisotropy(0))
        return

    def test_siteCartesianUij_valid(self):
        """Check DerivedStructureAdapter.siteCartesianUij."""
        adpt1 = self.adpt1
        uiso = 0.005 * numpy.identity(3)
        self.assertTrue(numpy.array_equal(uiso, adpt1.siteCartesianUij(0)))
        self.assertRaises(IndexError, adpt1.siteCartesianUij, 1)
        return


# End of class TestDerivedStructureAdapter

# ----------------------------------------------------------------------------


class TestStructureAdapter(unittest.TestCase):

    def setUp(self):
        self.adpt = StructureAdapter()
        return

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

    def test_siteAtomType(self):
        """Check StructureAdapter.siteAtomType()"""
        self.assertEqual("", self.adpt.siteAtomType(0))
        return

    def test_siteCartesianPosition(self):
        """Check StructureAdapter.siteCartesianPosition()"""
        self.assertRaises(RuntimeError, self.adpt.siteAnisotropy, 0)
        return

    def test_siteMultiplicity(self):
        """Check StructureAdapter.siteMultiplicity()"""
        self.assertEqual(1, self.adpt.siteMultiplicity(0))
        return

    def test_siteOccupancy(self):
        """Check StructureAdapter.siteOccupancy()"""
        # check if we use the C++ method that alwasy return 1.
        self.assertEqual(1.0, self.adpt.siteOccupancy(0))
        self.assertEqual(1.0, self.adpt.siteOccupancy(99))
        return

    def test_siteAnisotropy(self):
        """Check StructureAdapter.siteAnisotropy()"""
        self.assertRaises(RuntimeError, self.adpt.siteAnisotropy, 0)
        return

    def test_siteCartesianUij(self):
        """Check StructureAdapter.siteCartesianUij()"""
        self.assertRaises(RuntimeError, self.adpt.siteCartesianUij, 0)
        return


#   def test_totalOccupancy(self):
#       """check StructureAdapter.totalOccupancy()
#       """
#       return

# End of class TestStructureAdapter

# ----------------------------------------------------------------------------


class TestAtom(unittest.TestCase):

    def setUp(self):
        self.a = Atom()
        return

    def test___init__copy(self):
        """Check Atom copy constructor."""
        self.a.xyz_cartn = (1, 2, 3)
        a1 = Atom(self.a)
        self.assertEqual(self.a, a1)
        self.assertNotEqual(self.a, Atom())
        return

    def test_equality(self):
        """Check Atom equal and not equal operators."""
        self.assertEqual(self.a, Atom())
        self.assertFalse(self.a != Atom())
        a1 = Atom()
        a1.atomtype = "Na"
        self.assertNotEqual(self.a, a1)
        return

    def test_pickling(self):
        """Check pickling of Atom instances."""
        self.a.atomtype = "Na"
        a1 = pickle.loads(pickle.dumps(self.a))
        self.assertEqual("Na", a1.atomtype)
        self.assertEqual(self.a, a1)
        self.assertFalse(self.a is a1)
        return

    def test_xyz_cartn(self):
        """Check Atom.xyz_cartn."""
        a = self.a
        a.xyz_cartn = 4, 5, 6
        self.assertTrue(numpy.array_equal([4, 5, 6], a.xyz_cartn))
        self.assertEqual(4.0, a.xc)
        self.assertEqual(5.0, a.yc)
        self.assertEqual(6.0, a.zc)
        return

    def test_uij_cartn(self):
        """Check Atom.uij_cartn."""
        a = self.a
        a.uij_cartn = numpy.identity(3) * 0.01
        a.uc12 = 0.012
        a.uc13 = 0.013
        a.uc23 = 0.023
        self.assertTrue(
            numpy.array_equal(
                a.uij_cartn,
                [[0.01, 0.012, 0.013], [0.012, 0.01, 0.023], [0.013, 0.023, 0.01]],
            )
        )
        self.assertEqual(0.01, a.uc11)
        self.assertEqual(0.01, a.uc22)
        self.assertEqual(0.01, a.uc33)
        self.assertEqual(0.012, a.uc12)
        self.assertEqual(0.013, a.uc13)
        self.assertEqual(0.023, a.uc23)
        return

    def test_xc_yc_zc(self):
        "check Atom properties xc, yc, zc."
        a = self.a
        a.xc, a.yc, a.zc = numpy.arange(1, 4)
        self.assertEqual(1.0, a.xc)
        self.assertEqual(2.0, a.yc)
        self.assertEqual(3.0, a.zc)
        return

    def test_occupancy(self):
        "check Atom.occupancy"
        a = self.a
        (a.occupancy,) = numpy.arange(1)
        self.assertEqual(0.0, a.occupancy)
        a.occupancy = numpy.float32(0.5)
        self.assertEqual(0.5, a.occupancy)
        return

    def test_anisotropy(self):
        "check Atom.anisotropy"
        a = self.a
        nptrue, npfalse = numpy.arange(2) < 1
        a.anisotropy = nptrue
        self.assertTrue(a.anisotropy)
        a.anisotropy = npfalse
        self.assertFalse(a.anisotropy)
        return

    def test_ucij(self):
        "check Atom attributes u11, u22, etc."
        a = self.a
        a.uc11, a.uc22, a.uc33, a.uc12, a.uc13, a.uc23 = numpy.arange(1, 7)
        uijexp = [[1, 4, 5], [4, 2, 6], [5, 6, 3]]
        self.assertTrue(numpy.array_equal(uijexp, a.uij_cartn))
        return


# End of class TestAtom

# ----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()

# End of file
