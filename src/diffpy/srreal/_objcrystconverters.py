"""Copy ObjCryst structures without crossing a C++ library ABI boundary.

Use only pyobjcryst's Python API so both its Boost.Python releases and
its nanobind migration (with a private, static ObjCryst++ library) can
be used. Keep the numerical conventions of libdiffpy's
ObjCrystStructureAdapter.
"""

import numpy as np

from diffpy.srreal.srreal_ext import (
    AtomicStructureAdapter,
    CrystalStructureAdapter,
)


def _check_type(obj, name):
    # Import lazily: pyobjcryst is an optional runtime dependency.
    from pyobjcryst import _pyobjcryst

    if not isinstance(obj, getattr(_pyobjcryst, name)):
        raise TypeError(f"Expected a pyobjcryst {name} object.")


def _power_arrays(powers):
    """Return atom types, anisotropy flags and U tensors of scattering
    powers, in pyobjcryst coordinates."""
    n = len(powers)
    atomtypes = [sp.GetSymbol() for sp in powers]
    anisotropy = np.fromiter((not sp.IsIsotropic() for sp in powers), bool, n)
    biso = np.fromiter((sp.GetBiso() for sp in powers), float, n)
    bij = np.zeros((n, 3, 3))
    for k in range(3):
        bij[:, k, k] = biso
    for i in np.flatnonzero(anisotropy):
        sp = powers[i]
        bij[i] = [[sp.GetBij(j, k) for k in (1, 2, 3)] for j in (1, 2, 3)]
    return atomtypes, anisotropy, bij / (8 * np.pi**2)


def convertObjCrystMolecule(molecule):
    """Copy molecule atoms in local Cartesian coordinates, omitting
    dummies."""
    _check_type(molecule, "Molecule")
    adapter = AtomicStructureAdapter()
    powers, xyz, occupancy = [], [], []
    for i in range(molecule.GetNbComponent()):
        source = molecule.GetAtom(i)
        if source.IsDummy():
            continue
        powers.append(source.GetScatteringPower())
        xyz.append((source.X, source.Y, source.Z))
        occupancy.append(source.Occupancy)
    atomtypes, anisotropy, uij = _power_arrays(powers)
    adapter.setAtomArrays(atomtypes, xyz, occupancy, anisotropy, uij)
    return adapter


def _crystal_components(crystal):
    """Read the asymmetric unit in crystal order."""
    from pyobjcryst._pyobjcryst import Crystal

    for name in (
        "GetScatteringComponentList",
        "GetNbScatterer",
        "GetScatterer",
    ):
        method = getattr(crystal, name)
        if (
            getattr(method, "__self__", None) is not crystal
            or getattr(method, "__func__", None) is not getattr(Crystal, name)
        ):
            return crystal.GetScatteringComponentList()
    return [
        component
        for i in range(crystal.GetNbScatterer())
        for component in crystal.GetScatterer(i).GetScatteringComponentList()
    ]


def convertObjCrystCrystal(crystal):
    """Copy the asymmetric unit and complete space-group symmetry
    operations."""
    _check_type(crystal, "Crystal")
    # The master clock does not track all changes to the crystal's children.
    adapter = CrystalStructureAdapter()
    lattice = [crystal.GetLatticePar(i) for i in range(6)]
    lattice[3:] = np.degrees(lattice[3:])
    adapter.setLatPar(*lattice)
    powers, xyz, occupancy = [], [], []
    for source in _crystal_components(crystal):
        sp = source.mpScattPow
        if sp is None:
            continue
        powers.append(sp)
        xyz.append((source.mX, source.mY, source.mZ))
        occupancy.append(source.mOccupancy)
    atomtypes, anisotropy, uij = _power_arrays(powers)
    # isotropic U is already Cartesian
    adapter._setLatticeAtomArrays(
        atomtypes, xyz, occupancy, anisotropy, uij, isotropic_cartesian=True
    )

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
    adapter._setSymOps([r for r, t in operations], [t for r, t in operations])
    return adapter
