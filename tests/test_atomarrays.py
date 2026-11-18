#!/usr/bin/env python
"""Unit tests for the atom data of structure adapters as arrays."""

import gc
import weakref

import numpy
import pytest
from testutils import loadDiffPyStructure

from diffpy.srreal.structureadapter import (
    Atom,
    AtomicStructureAdapter,
    CrystalStructureAdapter,
    PeriodicStructureAdapter,
    createStructureAdapter,
)
from diffpy.structure import Atom as StructureAtom
from diffpy.structure import Lattice, Structure

TRICLINIC = (4.1, 5.2, 6.3, 81.0, 97.0, 115.0)
ADAPTERS = [
    AtomicStructureAdapter,
    PeriodicStructureAdapter,
    CrystalStructureAdapter,
]


def make_arrays(n=7):
    """Return arrays of n atoms, every third one anisotropic."""
    rng = numpy.random.default_rng(n)
    atomtypes = [("C", "Ti4+", "O2-")[i % 3] for i in range(n)]
    xyz = rng.uniform(-1.0, 2.0, (n, 3))
    occupancy = rng.uniform(0.1, 1.0, n)
    anisotropy = numpy.arange(n) % 3 == 1
    uiso = rng.uniform(0.002, 0.02, n)
    uij = uiso[:, None, None] * numpy.eye(3)
    for i in numpy.flatnonzero(anisotropy):
        m = rng.uniform(-0.03, 0.03, (3, 3))
        uij[i] = 0.5 * (m + m.T) + 0.1 * numpy.eye(3)
    return atomtypes, xyz, occupancy, anisotropy, uij


def append_atoms(adpt, arrays, lattice=False, isotropic_cartesian=False):
    """Fill the adapter atom by atom, as the converters once did."""
    for atomtype, xyz, occupancy, anisotropy, uij in zip(*arrays):
        atom = Atom()
        atom.atomtype = atomtype
        atom.xyz_cartn = xyz
        atom.occupancy = occupancy
        atom.anisotropy = anisotropy
        atom.uij_cartn = uij
        if lattice:
            adpt.toCartesian(atom)
            if isotropic_cartesian and not anisotropy:
                atom.uij_cartn = uij
        adpt.append(atom)
    return adpt


@pytest.mark.parametrize("cls", ADAPTERS)
def test_set_atom_arrays(cls):
    arrays = make_arrays()
    expected = append_atoms(cls(), arrays)
    adpt = cls()
    adpt.setAtomArrays(*arrays)
    assert expected == adpt
    assert 7 == adpt.countSites()
    # the adapter copies the arrays
    arrays[1][:] = 0.0
    assert expected == adpt
    # other sequences give the same atoms
    atomtypes, xyz, occupancy, anisotropy, uij = make_arrays()
    adpt.setAtomArrays(
        tuple(atomtypes),
        xyz.tolist(),
        list(occupancy),
        [bool(f) for f in anisotropy],
        numpy.asfortranarray(uij),
    )
    assert expected == adpt
    # new arrays replace all atoms
    adpt.setAtomArrays(*make_arrays(3))
    assert append_atoms(cls(), make_arrays(3)) == adpt
    adpt.setAtomArrays([], [], [], [], [])
    assert cls() == adpt
    return


@pytest.mark.parametrize("cls", ADAPTERS)
def test_get_atom_arrays(cls):
    arrays = make_arrays()
    adpt = append_atoms(cls(), arrays)
    atomtypes, xyz, occupancy, anisotropy, uij = adpt.getAtomArrays()
    assert arrays[0] == atomtypes
    assert numpy.array_equal(arrays[1], xyz)
    assert numpy.array_equal(arrays[2], occupancy)
    assert numpy.array_equal(arrays[3], anisotropy)
    assert bool == anisotropy.dtype
    assert numpy.array_equal(arrays[4], uij)
    assert (7, 3) == xyz.shape
    assert (7, 3, 3) == uij.shape
    # the arrays are copies
    xyz[:] = 0.0
    assert numpy.array_equal(arrays[1], adpt.getAtomArrays()[1])
    # an empty adapter has empty arrays of the same dimensions
    atomtypes, xyz, occupancy, anisotropy, uij = cls().getAtomArrays()
    assert [] == atomtypes
    assert (0, 3) == xyz.shape
    assert (0,) == occupancy.shape
    assert (0,) == anisotropy.shape
    assert (0, 3, 3) == uij.shape
    return


def test_atom_array_boolean_storage():
    atomtypes, xyz, occ, _, uij = make_arrays(3)
    adpt = AtomicStructureAdapter()
    # Nonzero bytes in a read-only NumPy bool view represent True.
    flags = numpy.array([0, 2, 255], dtype=numpy.uint8).view(bool)
    flags.setflags(write=False)
    for view in (flags, flags[::-1]):
        adpt.setAtomArrays(atomtypes, xyz, occ, view, uij)
        out = adpt.getAtomArrays()[3]
        assert out.flags.owndata
        numpy.testing.assert_array_equal(out, view)
        numpy.testing.assert_array_equal(out.view(numpy.uint8), view != 0)


def test_set_atom_arrays_invalid():
    adpt = AtomicStructureAdapter()
    adpt.setAtomArrays(*make_arrays())
    expected = adpt.clone()
    atomtypes, xyz, occupancy, anisotropy, uij = make_arrays()
    invalid = [
        (atomtypes[:-1], xyz, occupancy, anisotropy, uij),
        (atomtypes, xyz[:-1], occupancy, anisotropy, uij),
        (atomtypes, xyz[:, :2], occupancy, anisotropy, uij),
        (atomtypes, xyz.ravel(), occupancy, anisotropy, uij),
        (atomtypes, xyz, occupancy[:-1], anisotropy, uij),
        (atomtypes, xyz, occupancy[:, None], anisotropy, uij),
        (atomtypes, xyz, occupancy, anisotropy[1:], uij),
        (atomtypes, xyz, occupancy, anisotropy, uij[:-1]),
        (atomtypes, xyz, occupancy, anisotropy, uij[:, :, :2]),
        (atomtypes, xyz, occupancy, anisotropy, uij.reshape(-1, 9)),
    ]
    for arrays in invalid:
        with pytest.raises(ValueError):
            adpt.setAtomArrays(*arrays)
    with pytest.raises(TypeError):
        adpt.setAtomArrays([1] * 7, xyz, occupancy, anisotropy, uij)
    with pytest.raises(TypeError):
        adpt.setAtomArrays(atomtypes, xyz, occupancy, anisotropy)
    # a failed call leaves the atoms as they were
    assert expected == adpt
    return


def test_set_atom_arrays_detaches_atoms():
    adpt = AtomicStructureAdapter()
    adpt.setAtomArrays(*make_arrays())
    first = adpt[0]
    xyz_view = first.xyz_cartn
    xyz0 = xyz_view.copy()
    adpt.setAtomArrays(*make_arrays(3))
    numpy.testing.assert_array_equal(xyz_view, xyz0)
    xyz_view[:] = 17.0
    assert not numpy.shares_memory(xyz_view, adpt[0].xyz_cartn)
    # the atom keeps its values and no longer changes the adapter
    assert numpy.array_equal(xyz0, first.xyz_cartn)
    first.occupancy = 0.123
    assert 0.123 != adpt[0].occupancy
    assert 3 == len(adpt)
    return


@pytest.mark.parametrize("cls", ADAPTERS[1:])
def test_set_lattice_atom_arrays(cls):
    arrays = make_arrays()
    expected = cls()
    expected.setLatPar(*TRICLINIC)
    append_atoms(expected, arrays, lattice=True)
    adpt = cls()
    adpt.setLatPar(*TRICLINIC)
    adpt._setLatticeAtomArrays(*arrays)
    assert expected == adpt
    assert not numpy.allclose(arrays[4], adpt.getAtomArrays()[4])
    # Cartesian isotropic tensors stay as given, anisotropic ones change
    expected = cls()
    expected.setLatPar(*TRICLINIC)
    append_atoms(expected, arrays, lattice=True, isotropic_cartesian=True)
    adpt._setLatticeAtomArrays(*arrays, isotropic_cartesian=True)
    assert expected == adpt
    uij = adpt.getAtomArrays()[4]
    anisotropy = arrays[3]
    assert numpy.array_equal(arrays[4][~anisotropy], uij[~anisotropy])
    assert not numpy.allclose(arrays[4][anisotropy], uij[anisotropy])
    return


def test_set_symops():
    adpt = CrystalStructureAdapter()
    adpt.setLatPar(*TRICLINIC)
    adpt._setLatticeAtomArrays(*make_arrays())
    rotations = numpy.array([numpy.eye(3), -numpy.eye(3)])
    translations = numpy.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.5]])
    adpt._setSymOps(rotations, translations)
    assert 2 == adpt.countSymOps()
    for i in range(2):
        R, t = adpt.getSymOp(i)
        assert numpy.array_equal(rotations[i], R)
        assert numpy.array_equal(translations[i], t)
    expected = CrystalStructureAdapter()
    expected.setLatPar(*TRICLINIC)
    expected._setLatticeAtomArrays(*make_arrays())
    for R, t in zip(rotations, translations):
        expected.addSymOp(R, t)
    assert expected == adpt
    assert 2 == adpt.siteMultiplicity(0)
    # the same operations again keep the adapter as it is
    adpt._setSymOps(rotations, translations)
    assert expected == adpt
    adpt._setSymOps(rotations[:1], translations[:1])
    assert 1 == adpt.countSymOps()
    assert 1 == adpt.siteMultiplicity(0)
    with pytest.raises(ValueError):
        adpt._setSymOps(rotations, translations[:1])
    with pytest.raises(ValueError):
        adpt._setSymOps(rotations[:, :2], translations)
    assert 1 == adpt.countSymOps()
    return


def reference_adapter(stru):
    """Convert a diffpy.structure object atom by atom."""
    periodic = (1, 1, 1, 90, 90, 90) != stru.lattice.cell_parms()
    adpt = PeriodicStructureAdapter() if periodic else AtomicStructureAdapter()
    if periodic:
        adpt.setLatPar(*stru.lattice.cell_parms())
    for a in stru:
        atom = Atom()
        atom.atomtype = a.element
        atom.occupancy = a.occupancy
        atom.anisotropy = a.anisotropy
        atom.xyz_cartn = a.xyz
        atom.uij_cartn = a.U
        if periodic:
            adpt.toCartesian(atom)
        adpt.append(atom)
    return adpt


def check_same_atoms(expected, adpt):
    """Check that the adapters have the same atoms to the last bit."""
    assert type(expected) is type(adpt)
    assert expected == adpt
    for x0, x1 in zip(expected.getAtomArrays(), adpt.getAtomArrays()):
        assert numpy.array_equal(x0, x1)
    return


@pytest.mark.parametrize(
    "filename",
    ["Ni.stru", "TiO2_rutile-fit.stru", "C60bucky.stru", "silicon.cif"],
)
def test_diffpy_structure_files(filename):
    stru = loadDiffPyStructure(filename)
    adpt = createStructureAdapter(stru)
    check_same_atoms(reference_adapter(stru), adpt)
    if hasattr(adpt, "getLatPar"):
        assert stru.lattice.cell_parms() == adpt.getLatPar()
    return


@pytest.mark.parametrize("cell", [TRICLINIC, (1, 1, 1, 90, 90, 90)])
def test_diffpy_structure_mixed_displacements(cell):
    stru = Structure(lattice=Lattice(*cell))
    stru.append(StructureAtom("Na", [0.1, 0.2, 0.3], Uisoequiv=0.004))
    stru.append(
        StructureAtom(
            "Cl",
            [0.6, 0.5, 0.9],
            occupancy=0.7,
            anisotropy=True,
            U=[[0.01, 0.002, 0.0], [0.002, 0.02, -0.003], [0.0, -0.003, 0.03]],
        )
    )
    stru.append(StructureAtom("Na", [0.4, 0.8, 0.2]))
    stru.append(StructureAtom("Cl", [0.3, 0.1, 0.7], Uisoequiv=0.011))
    adpt = createStructureAdapter(stru)
    check_same_atoms(reference_adapter(stru), adpt)
    assert [False, True, False, False] == list(adpt.getAtomArrays()[3])
    # isotropic atoms have Uisoequiv on the diagonal
    uij = adpt.getAtomArrays()[4]
    assert numpy.allclose(0.004 * numpy.eye(3), uij[0], rtol=0, atol=1e-17)
    assert numpy.array_equal(numpy.zeros((3, 3)), uij[2])
    # all atoms anisotropic, and none
    for a in stru:
        a.anisotropy = True
    check_same_atoms(reference_adapter(stru), createStructureAdapter(stru))
    check_same_atoms(
        reference_adapter(stru[:0]), createStructureAdapter(stru[:0])
    )
    return


@pytest.mark.parametrize(
    "cell",
    [(1, 1, 1, 90, 90, 90), (4, 5, 6, 90, 90, 90)],
    ids=["atomic", "periodic"],
)
def test_diffpy_structure_shared_atom_lattice(cell):
    stru = Structure(lattice=Lattice(*cell))
    stru.append(StructureAtom("Na", [0.1, 0.2, 0.3], Uisoequiv=0.02))
    stru.append(StructureAtom("Cl", [0.4, 0.5, 0.6], Uisoequiv=0.01))
    subset = stru[:1]
    subset.lattice = Lattice(*TRICLINIC)
    assert stru[0].lattice is subset.lattice
    check_same_atoms(reference_adapter(stru), createStructureAdapter(stru))


@pytest.mark.parametrize("occupancy", [None, "0.7"])
def test_diffpy_structure_invalid_occupancy(occupancy):
    stru = Structure([StructureAtom("Ni"), StructureAtom("O")])
    stru[1].occupancy = occupancy
    with pytest.raises(TypeError):
        createStructureAdapter(stru)


@pytest.mark.parametrize("xyz", [0.25, [0.25]])
def test_diffpy_structure_broadcast_coordinates(xyz):
    stru = Structure([StructureAtom("Ni"), StructureAtom("O", [1, 2, 3])])
    stru[0].xyz = xyz
    check_same_atoms(reference_adapter(stru), createStructureAdapter(stru))


def test_diffpy_structure_atom_u_override():
    class DisorderedAtom(StructureAtom):
        @property
        def U(self):
            return super().U + 0.02 * self.lattice.isotropicunit

    stru = Structure([StructureAtom("Ni", Uisoequiv=0.01)])
    stru.append(DisorderedAtom("O", Uisoequiv=0.01), copy=False)
    check_same_atoms(reference_adapter(stru), createStructureAdapter(stru))


def test_converter_accepts_unhashable_class():
    class Unhashable(type):
        __hash__ = None

    class MyStructure(Structure, metaclass=Unhashable):
        pass

    stru = MyStructure([StructureAtom("C")])
    assert createStructureAdapter(stru).countSites() == 1


def test_converter_releases_dynamic_class():
    class MyStructure(Structure):
        pass

    ref = weakref.ref(MyStructure)
    createStructureAdapter(MyStructure())
    del MyStructure
    gc.collect()
    assert ref() is None


def test_converter_of_new_type():
    """Check that a converter registered later is used at once."""
    from diffpy.srreal.structureadapter import (
        RegisterStructureAdapter,
        _adapter_converters_registry,
    )

    class MyStructure(Structure):
        pass

    stru = MyStructure(loadDiffPyStructure("Ni.stru"))
    assert 4 == createStructureAdapter(stru).countSites()
    fqname = str(MyStructure).split("'")[1]
    try:
        RegisterStructureAdapter(fqname, lambda s: AtomicStructureAdapter())
        assert 0 == createStructureAdapter(stru).countSites()
    finally:
        del _adapter_converters_registry[fqname]
    return


@pytest.mark.parametrize("cls", ADAPTERS)
def test_fast_updates_of_python_adapter_class(cls):
    """Check that a class derived in Python keeps the fast updates.

    The calculator keeps a copy of the last structure, which is of the
    built-in class.
    """
    from diffpy.srreal.pdfcalculator import PDFCalculator

    class DerivedAdapter(cls):
        pass

    def make(shift):
        adpt = DerivedAdapter()
        if hasattr(adpt, "setLatPar"):
            adpt.setLatPar(11.0, 12.0, 13.0, 90, 90, 90)
        atomtypes, xyz, occupancy, anisotropy, uij = make_arrays()
        xyz[3, 0] += shift
        adpt.setAtomArrays(atomtypes, xyz, occupancy, anisotropy, uij)
        if hasattr(adpt, "addSymOp"):
            adpt.addSymOp(numpy.eye(3), numpy.zeros(3))
        return adpt

    calc = PDFCalculator(rmax=8)
    basic = PDFCalculator(rmax=8, evaluatortype="BASIC")
    r, g0 = calc(make(0.0))
    assert "BASIC" == calc.evaluatortypeused
    for shift in (0.0, 0.2):
        r, g = calc(make(shift))
        assert "OPTIMIZED" == calc.evaluatortypeused
        r, gb = basic(make(shift))
        assert numpy.allclose(gb, g, rtol=0, atol=1e-10)
    assert not numpy.allclose(g0, g)
    return
