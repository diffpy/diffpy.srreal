"""Integration tests for copying structures through pyobjcryst's Python
API."""

import subprocess
import sys

import numpy as np
import pytest

from diffpy.srreal.structureadapter import createStructureAdapter
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
    )


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
