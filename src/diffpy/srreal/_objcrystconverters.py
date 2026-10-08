"""Copy ObjCryst structures without crossing a C++ library ABI boundary.

Use only pyobjcryst's Python API so both its Boost.Python releases and
its nanobind migration (with a private, static ObjCryst++ library) can
be used. Keep the numerical conventions of libdiffpy's
ObjCrystStructureAdapter.
"""

import numpy as np

from diffpy.srreal.srreal_ext import (
    Atom,
    AtomicStructureAdapter,
    CrystalStructureAdapter,
)


def _check_type(obj, name):
    # Import lazily: pyobjcryst is an optional runtime dependency.
    from pyobjcryst import _pyobjcryst

    if not isinstance(obj, getattr(_pyobjcryst, name)):
        raise TypeError(f"Expected a pyobjcryst {name} object.")


def _atom_from_scattering_power(sp):
    atom = Atom()
    atom.atomtype = sp.GetSymbol()
    atom.anisotropy = not sp.IsIsotropic()
    if atom.anisotropy:
        bij = np.array(
            [[sp.GetBij(i, j) for j in range(1, 4)] for i in range(1, 4)]
        )
    else:
        bij = np.eye(3) * sp.GetBiso()
    atom.uij_cartn = bij / (8 * np.pi**2)
    return atom


def convertObjCrystMolecule(molecule):
    """Copy molecule atoms in local Cartesian coordinates, omitting
    dummies."""
    _check_type(molecule, "Molecule")
    adapter = AtomicStructureAdapter()
    adapter.reserve(molecule.GetNbComponent())
    for i in range(molecule.GetNbComponent()):
        source = molecule.GetAtom(i)
        if source.IsDummy():
            continue
        atom = _atom_from_scattering_power(source.GetScatteringPower())
        atom.xyz_cartn = [source.X, source.Y, source.Z]
        atom.occupancy = source.Occupancy
        adapter.append(atom)
    return adapter


def convertObjCrystCrystal(crystal):
    """Copy the asymmetric unit and complete space-group symmetry
    operations."""
    _check_type(crystal, "Crystal")
    adapter = CrystalStructureAdapter()
    lattice = [crystal.GetLatticePar(i) for i in range(6)]
    lattice[3:] = np.degrees(lattice[3:])
    adapter.setLatPar(*lattice)
    components = crystal.GetScatteringComponentList()
    adapter.reserve(len(components))
    for source in components:
        sp = source.mpScattPow
        if sp is None:
            continue
        atom = _atom_from_scattering_power(sp)
        atom.xyz_cartn = [source.mX, source.mY, source.mZ]
        atom.occupancy = source.mOccupancy
        # Isotropic U is already Cartesian. Only anisotropic U follows the
        # lattice transformation; coordinates always need that transformation.
        uij = atom.uij_cartn.copy()
        adapter.toCartesian(atom)
        if not atom.anisotropy:
            atom.uij_cartn = uij
        adapter.append(atom)

    sg = crystal.GetSpaceGroup()
    operations = [
        (np.asarray(rotation), np.asarray(translation) + offset)
        for offset in sg.GetTranslationVectors()
        for translation, rotation in sg.GetSymmetryOperations()
    ]
    if len(operations) < sg.GetNbSymmetrics():
        inversion = 2 * np.asarray(sg.GetInversionCenter())
        operations += [(-r, inversion - t) for r, t in operations]
    if len(operations) != sg.GetNbSymmetrics():
        raise ValueError("Inconsistent pyobjcryst space-group operations.")
    for rotation, translation in operations:
        adapter.addSymOp(rotation, translation)
    adapter.updateSymmetryPositions()
    return adapter
