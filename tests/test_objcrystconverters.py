"""Integration tests for copying structures through pyobjcryst's Python
API."""

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, wait
from threading import Event

import numpy as np
import pytest

from diffpy.srreal.structureadapter import (
    CrystalStructureAdapter,
    createStructureAdapter,
)
from diffpy.srreal.structureconverters import (
    convertObjCrystCrystal,
    convertObjCrystMolecule,
)


@pytest.fixture
def objcryst():
    return pytest.importorskip("pyobjcryst._pyobjcryst")


@pytest.mark.parametrize(
    "spacegroup", ["P1", "P -1", "F m -3 m", "F d -3 m:2"]
)
def test_crystal_symmetry_and_isotropic_u(objcryst, spacegroup):
    crystal = objcryst.Crystal(4, 4, 4, spacegroup)
    sp = objcryst.ScatteringPowerAtom("C", "C")
    sp.SetBiso(0.5)
    source = objcryst.Atom(0.137, 0.231, 0.319, "C1", sp)
    source.Occupancy = 0.7
    crystal.AddScatteringPower(sp)
    crystal.AddScatterer(source)
    adapter = createStructureAdapter(crystal)

    sg = crystal.GetSpaceGroup()
    assert adapter.countSites() == 1
    assert adapter.countSymOps() == sg.GetNbSymmetrics()
    assert adapter.totalOccupancy() == pytest.approx(
        0.7 * len(sg.GetAllSymmetrics(source.X, source.Y, source.Z))
    )
    np.testing.assert_allclose(adapter[0].xyz_cartn, [0.548, 0.924, 1.276])
    np.testing.assert_allclose(
        adapter[0].uij_cartn, np.eye(3) * 0.5 / (8 * np.pi**2)
    )
    # Compare actual positions, not just counts, including origin-2 inversion.
    expected = sg.GetAllSymmetrics(source.X, source.Y, source.Z) % 1
    for atom in adapter.getEquivalentAtoms(0):
        adapter.toFractional(atom)
        assert (
            np.min(np.linalg.norm(expected - atom.xyz_cartn % 1, axis=1))
            < 1e-8
        )
    # The adapter owns a snapshot; later upstream changes cannot invalidate it.
    source.X = 0.5
    sp.SetBiso(2)
    np.testing.assert_allclose(adapter[0].xyz_cartn, [0.548, 0.924, 1.276])
    assert adapter[0].uij_cartn[0, 0] == pytest.approx(0.5 / (8 * np.pi**2))


def test_molecule_coordinates_occupancy_anisotropy_and_dummy(objcryst):
    crystal = objcryst.Crystal(10, 10, 10, "P1")
    molecule = objcryst.Molecule(crystal, "molecule")
    sp = objcryst.ScatteringPowerAtom("C", "C")
    sp.B11, sp.B22, sp.B33 = 0.4, 0.5, 0.6
    sp.B12, sp.B13, sp.B23 = 0.01, 0.02, 0.03
    crystal.AddScatteringPower(sp)
    molecule.AddAtom(1, 2, 3, sp, "C1")
    molecule.GetAtom(0).Occupancy = 0.5
    molecule.AddAtom(0, 0, 0, None, "dummy")
    adapter = convertObjCrystMolecule(molecule)
    assert len(adapter) == 1
    assert adapter[0].atomtype == "C"
    assert adapter[0].occupancy == 0.5
    assert adapter[0].anisotropy
    np.testing.assert_allclose(adapter[0].xyz_cartn, [1, 2, 3])
    np.testing.assert_allclose(
        adapter[0].uij_cartn,
        np.array([[0.4, 0.01, 0.02], [0.01, 0.5, 0.03], [0.02, 0.03, 0.6]])
        / (8 * np.pi**2),
        rtol=0,
        atol=1e-16,
    )

    oxygen = objcryst.ScatteringPowerAtom("O", "O")
    oxygen.SetBiso(0.7)
    crystal.AddScatteringPower(oxygen)
    molecule.AddAtom(-1, 0.5, 2, oxygen, "O1")
    molecule.AddAtom(4, 5, 6, sp, "C2")
    molecule.GetAtom(3).Occupancy = 0.25
    adapter = convertObjCrystMolecule(molecule)
    atomtypes, xyz, occupancy, anisotropy, uij = adapter.getAtomArrays()
    assert ["C", "O", "C"] == atomtypes
    np.testing.assert_array_equal([[1, 2, 3], [-1, 0.5, 2], [4, 5, 6]], xyz)
    np.testing.assert_array_equal([0.5, 1, 0.25], occupancy)
    np.testing.assert_array_equal([True, False, True], anisotropy)
    np.testing.assert_allclose(
        uij[1], np.eye(3) * 0.7 / (8 * np.pi**2), rtol=0, atol=1e-16
    )
    assert 0 == len(convertObjCrystMolecule(objcryst.Molecule(crystal, "m")))


def test_wrong_structure_type(objcryst):
    with pytest.raises(TypeError):
        convertObjCrystCrystal(object())
    with pytest.raises(TypeError):
        convertObjCrystMolecule(objcryst.Crystal())


def test_structureadapter_can_be_imported_first():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from diffpy.srreal.structureadapter "
            "import createStructureAdapter; "
            "from diffpy.structure import Structure; "
            "assert createStructureAdapter(Structure()).countSites() == 0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def reference_crystal_adapter(crystal):
    """Convert a crystal atom by atom from its list of components."""
    from diffpy.srreal.structureadapter import Atom

    adapter = CrystalStructureAdapter()
    lattice = [crystal.GetLatticePar(i) for i in range(6)]
    lattice[3:] = np.degrees(lattice[3:])
    adapter.setLatPar(*lattice)
    for source in crystal.GetScatteringComponentList():
        sp = source.mpScattPow
        if sp is None:
            continue
        atom = Atom()
        atom.atomtype = sp.GetSymbol()
        atom.anisotropy = not sp.IsIsotropic()
        if atom.anisotropy:
            bij = np.array(
                [[sp.GetBij(i, j) for j in range(1, 4)] for i in range(1, 4)]
            )
        else:
            bij = np.eye(3) * sp.GetBiso()
        uij = bij / (8 * np.pi**2)
        atom.uij_cartn = uij
        atom.xyz_cartn = [source.mX, source.mY, source.mZ]
        atom.occupancy = source.mOccupancy
        adapter.toCartesian(atom)
        if not atom.anisotropy:
            atom.uij_cartn = uij
        adapter.append(atom)
    sg = crystal.GetSpaceGroup()
    # For each operation f(x) = R x + t, f(0) gives its translation
    # and f(e_i) - f(0) gives column i of its rotation.
    translations = sg.GetAllSymmetrics(0, 0, 0)
    rotations = np.stack(
        [sg.GetAllSymmetrics(*axis) - translations for axis in np.eye(3)],
        axis=-1,
    )
    for rotation, translation in zip(rotations, translations):
        adapter.addSymOp(rotation, translation)
    return adapter


def make_crystal(objcryst, spacegroup="P n m a"):
    """Return a crystal with isotropic, anisotropic, partial atoms."""
    crystal = objcryst.Crystal(5.3, 7.7, 5.4, spacegroup)
    spca = objcryst.ScatteringPowerAtom("Ca", "Ca")
    spca.SetBiso(0.6)
    spti = objcryst.ScatteringPowerAtom("Ti", "Ti")
    spti.B11, spti.B22, spti.B33 = 0.4, 0.5, 0.6
    spti.B12, spti.B13, spti.B23 = 0.01, 0.02, 0.03
    spo = objcryst.ScatteringPowerAtom("O", "O")
    spo.SetBiso(0.9)
    for sp in (spca, spti, spo):
        crystal.AddScatteringPower(sp)
    crystal.AddScatterer(objcryst.Atom(0.03, 0.25, 0.99, "Ca1", spca))
    crystal.AddScatterer(objcryst.Atom(0.0, 0.0, 0.5, "Ti1", spti))
    crystal.AddScatterer(objcryst.Atom(0.48, 0.25, 0.07, "O1", spo))
    crystal.AddScatterer(objcryst.Atom(0.29, 0.04, 0.71, "O2", spo, 0.8))
    return crystal


def check_as_reference(crystal):
    """Check the converted crystal and its symmetry positions."""
    adapter = createStructureAdapter(crystal)
    expected = reference_crystal_adapter(crystal)
    assert expected.countSites() == adapter.countSites()
    assert expected.countSymOps() == adapter.countSymOps()
    assert expected.getLatPar() == adapter.getLatPar()
    a0, a1 = expected.getAtomArrays(), adapter.getAtomArrays()
    assert a0[0] == a1[0]
    np.testing.assert_allclose(a0[1], a1[1], rtol=0, atol=1e-13)
    np.testing.assert_array_equal(a0[2], a1[2])
    np.testing.assert_array_equal(a0[3], a1[3])
    np.testing.assert_allclose(a0[4], a1[4], rtol=0, atol=1e-15)
    for i in range(expected.countSymOps()):
        for x0, x1 in zip(expected.getSymOp(i), adapter.getSymOp(i)):
            np.testing.assert_array_equal(x0, x1)
    for i in range(expected.countSites()):
        e0 = expected.getEquivalentAtoms(i)
        e1 = adapter.getEquivalentAtoms(i)
        assert len(e0) == len(e1)
        for atom0, atom1 in zip(e0, e1):
            np.testing.assert_allclose(
                atom0.xyz_cartn, atom1.xyz_cartn, rtol=0, atol=1e-13
            )
            np.testing.assert_allclose(
                atom0.uij_cartn, atom1.uij_cartn, rtol=0, atol=1e-15
            )
    return adapter


@pytest.mark.parametrize("spacegroup", ["P1", "P n m a", "F m -3 m"])
def test_crystal_as_atom_by_atom(objcryst, spacegroup):
    check_as_reference(make_crystal(objcryst, spacegroup))


def test_crystal_without_occupancy_correction(objcryst):
    """Check that the crystal keeps its setting of the correction of
    occupancies, which the conversion does not use."""
    crystal = make_crystal(objcryst)
    assert crystal.GetUseDynPopCorr()
    check_as_reference(crystal)
    crystal.GetScatterer(3).X = 0.31
    check_as_reference(crystal)
    assert crystal.GetUseDynPopCorr()


def test_crystal_changes_after_conversion(objcryst):
    """Check repeated conversion of one crystal after each change."""
    crystal = make_crystal(objcryst)
    first = check_as_reference(crystal)
    again = createStructureAdapter(crystal)
    assert first == again
    assert first is not again
    # a changed copy does not show in the next conversion
    again[0].xyz_cartn = [1.0, 2.0, 3.0]
    again.setLatPar(4, 5, 6, 90, 90, 90)
    assert first == check_as_reference(crystal)
    ca, ti, o1, o2 = (crystal.GetScatterer(i) for i in range(4))
    spti = ti.GetScatteringPower()
    spo = o1.GetScatteringPower()
    changes = [
        lambda: setattr(o2, "X", 0.33),
        lambda: setattr(ca, "Occupancy", 0.5),
        lambda: spo.SetBiso(1.4),
        lambda: setattr(spti, "B11", 0.7),
        lambda: setattr(spti, "B23", -0.02),
        lambda: setattr(crystal, "a", 5.5),
        lambda: setattr(crystal, "c", 5.2),
        # to a special position and away from it
        lambda: setattr(o2, "Y", 0.25),
        lambda: setattr(o2, "Y", 0.1),
        lambda: crystal.ChangeSpaceGroup("P 21/c"),
        lambda: crystal.GetSpaceGroup().ChangeSpaceGroup("P -1"),
        lambda: crystal.RemoveScatterer(o1),
        lambda: crystal.AddScatterer(
            objcryst.Atom(0.1, 0.6, 0.2, "O3", spo, 0.4)
        ),
    ]
    last = first
    for change in changes:
        change()
        adapter = check_as_reference(crystal)
        assert last != adapter
        # an unchanged crystal converts to an equal adapter
        assert adapter == createStructureAdapter(crystal)
        last = adapter


def test_renamed_sphere_changes_atom_type(objcryst):
    crystal = objcryst.Crystal(5, 5, 5, "P1")
    power = objcryst.ScatteringPowerSphere("C", 1)
    crystal.AddScatteringPower(power)
    atom = objcryst.Atom(0.1, 0.2, 0.3, "site", power)
    crystal.AddScatterer(atom)
    original = createStructureAdapter(crystal)
    power.SetName("Ni")
    assert createStructureAdapter(crystal)[0].atomtype == "Ni"
    assert original[0].atomtype == "C"


def test_refinement_parameter_with_independent_clock(objcryst):
    from pyobjcryst.refinableobj import RefinableObjClock

    clock = RefinableObjClock()
    crystal = make_crystal(objcryst, "P1")
    power = crystal.GetScatterer(0).GetScatteringPower()
    biso = power.GetPar("Biso")
    biso.AssignClock(clock)
    original = createStructureAdapter(crystal)
    biso.SetValue(2.4)
    updated = createStructureAdapter(crystal)
    assert updated[0].uij_cartn[0, 0] == pytest.approx(2.4 / (8 * np.pi**2))
    assert original[0].uij_cartn[0, 0] == pytest.approx(0.6 / (8 * np.pi**2))


def test_change_symbol_and_remove_power_in_use(objcryst):
    crystal = make_crystal(objcryst, "P1")
    power = crystal.GetScatterer(0).GetScatteringPower()
    original = createStructureAdapter(crystal)
    power.SetSymbol("Mg")
    assert createStructureAdapter(crystal)[0].atomtype == "Mg"
    crystal.RemoveScatteringPower(power)
    power.SetBiso(1.7)
    atom = createStructureAdapter(crystal)[0]
    assert atom.atomtype == "Mg"
    assert atom.uij_cartn[0, 0] == pytest.approx(1.7 / (8 * np.pi**2))
    assert original[0].atomtype == "Ca"


def test_concurrent_crystal_snapshots(objcryst, monkeypatch):
    crystal = make_crystal(objcryst, "P1")
    original = createStructureAdapter(crystal)
    site = crystal.GetScatterer(0)
    site.Occupancy = 0.25
    read_first = Event()
    resume = Event()
    set_symops = CrystalStructureAdapter._setSymOps

    def set_symops_after_read(adapter, rotations, translations):
        # All source reads are complete before the main thread changes it.
        if len(rotations) == 1:
            read_first.set()
            resume.wait()
        set_symops(adapter, rotations, translations)

    monkeypatch.setattr(
        CrystalStructureAdapter, "_setSymOps", set_symops_after_read
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(createStructureAdapter, crystal)
        try:
            assert read_first.wait(timeout=5)
            site.Occupancy = 0.75
            crystal.ChangeSpaceGroup("P -1")
            second = pool.submit(createStructureAdapter, crystal)
            # A converter that serializes calls can finish after resume.
            wait([second], timeout=1)
        finally:
            resume.set()
    first, second = first.result(), second.result()
    assert (first[0].occupancy, first.countSymOps()) == (0.25, 1)
    assert (second[0].occupancy, second.countSymOps()) == (0.75, 2)
    assert original[0].occupancy == 1


@pytest.mark.parametrize("previously_registered", [False, True])
def test_molecule_with_unregistered_power(objcryst, previously_registered):
    from pyobjcryst.crystal import Crystal

    crystal = Crystal(5, 5, 5, "P1")
    molecule = objcryst.Molecule(crystal, "molecule")
    crystal.AddScatterer(molecule)
    registered = objcryst.ScatteringPowerAtom("C", "C")
    other = objcryst.ScatteringPowerAtom("C", "C")
    crystal.AddScatteringPower(registered)
    crystal.AddScatteringPower(objcryst.ScatteringPowerAtom("O", "O"))
    if previously_registered:
        crystal.AddScatteringPower(other)
        crystal.RemoveScatteringPower(other)
    molecule.AddAtom(0, 0, 0, registered, "C1")
    molecule.AddAtom(1, 0, 0, other, "C2")
    other.SetBiso(1)
    first = createStructureAdapter(crystal)
    other.SetBiso(3)
    second = createStructureAdapter(crystal)
    assert first[1].uij_cartn[0, 0] == pytest.approx(1 / (8 * np.pi**2))
    assert second[1].uij_cartn[0, 0] == pytest.approx(3 / (8 * np.pi**2))


def test_crystal_component_list_override(objcryst):
    class SelectedSiteCrystal(objcryst.Crystal):
        selected = 0

        def GetScatteringComponentList(self):
            scatterer = self.GetScatterer(self.selected)
            return iter(scatterer.GetScatteringComponentList())

    crystal = SelectedSiteCrystal(5, 5, 5, "P1")
    for i, symbol in enumerate(("C", "O")):
        power = objcryst.ScatteringPowerAtom(symbol, symbol)
        crystal.AddScatteringPower(power)
        crystal.AddScatterer(objcryst.Atom(0.2 * i, 0, 0, symbol, power))
    for i, symbol in enumerate(("C", "O")):
        crystal.selected = i
        assert createStructureAdapter(crystal).getAtomArrays()[0] == [symbol]


def test_crystal_lattice_override_after_conversion(objcryst):
    class ScaledCrystal(objcryst.Crystal):
        scale = 1.0

        def GetLatticePar(self, index):
            value = super().GetLatticePar(index)
            return value * self.scale if index < 3 else value

    crystal = ScaledCrystal(5, 5, 5, "P1")
    first = createStructureAdapter(crystal)
    crystal.scale = 1.2
    assert createStructureAdapter(crystal).getLatPar()[0] == 6
    assert first.getLatPar()[0] == 5


def test_crystal_component_order_with_scatterer_override(objcryst):
    class ReversedCrystal(objcryst.Crystal):
        def GetScatterer(self, index):
            return super().GetScatterer(self.GetNbScatterer() - 1 - index)

    crystal = ReversedCrystal(5, 5, 5, "P1")
    for i, symbol in enumerate(("C", "O")):
        power = objcryst.ScatteringPowerAtom(symbol, symbol)
        crystal.AddScatteringPower(power)
        crystal.AddScatterer(objcryst.Atom(0.2 * i, 0, 0, symbol, power))
    assert createStructureAdapter(crystal).getAtomArrays()[0] == ["C", "O"]


def test_crystals_converted_in_turn(objcryst):
    crystals = [make_crystal(objcryst, sg) for sg in ("P1", "P n m a")]
    for i in range(3):
        for crystal in crystals:
            crystal.GetScatterer(0).X += 0.01
            check_as_reference(crystal)
    # the adapter of a deleted crystal is not used for another one
    del crystals[:], crystal
    for sg in ("P -1", "P1", "P m m m"):
        check_as_reference(make_crystal(objcryst, sg))
