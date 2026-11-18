"""Copy ObjCryst structures without crossing a C++ library ABI boundary.

Use only pyobjcryst's Python API so both its Boost.Python releases and
its nanobind migration (with a private, static ObjCryst++ library) can
be used. Keep the numerical conventions of libdiffpy's
ObjCrystStructureAdapter.
"""

import weakref

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


def _update_crystal_adapter(adapter, crystal, components):
    """Copy lattice, asymmetric unit and symmetry operations to the
    adapter and return the scattering powers of the sites."""
    lattice = [crystal.GetLatticePar(i) for i in range(6)]
    lattice[3:] = np.degrees(lattice[3:])
    adapter.setLatPar(*lattice)
    powers, xyz, occupancy = [], [], []
    for source in components:
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
    adapter.updateSymmetryPositions()
    return powers


def _scatterer_components(crystal):
    """Return Crystal.GetScatteringComponentList without its slow
    occupancy correction for overlapping atoms."""
    rv = []
    for i in range(crystal.GetNbScatterer()):
        rv.extend(crystal.GetScatterer(i).GetScatteringComponentList())
    return rv


def _master_tracks_powers(crystal, powers):
    from pyobjcryst._pyobjcryst import ScatteringPowerAtom

    # SetName changes the symbol of a ScatteringPowerSphere, but not the
    # master clock.  Other power classes may behave alike.
    if any(type(sp) is not ScatteringPowerAtom for sp in powers):
        return False
    # Molecule atoms may use powers never added to the crystal.  Compare
    # native pointers, Python wrappers and names are not unique.
    used = {sp.int_ptr(): sp for sp in powers}
    registry = crystal.GetScatteringPowerRegistry()
    if len(used) > len(registry):
        return False
    for ptr, sp in used.items():
        # only AddScatteringPower connects the power to the crystal clock
        if sp.GetClientRegistry().Find(crystal) < 0:
            return False
        name = sp.GetName()
        if registry.Find(name, "ScatteringPowerAtom", True) < 0:
            return False
        if crystal.GetScatteringPower(name).int_ptr() != ptr:
            return False
    return True


# Adapters of converted crystals by id, refinements convert them again
# after every parameter change.
_crystal_adapters = {}


def _clock_snapshot(clock):
    """Return a copy of a pyobjcryst clock, or None if it cannot be
    copied."""
    try:
        from pyobjcryst.refinableobj import RefinableObjClock

        rv = RefinableObjClock()
        rv.SetEqual(clock)
    except (ImportError, AttributeError, TypeError):
        rv = None
    return rv


def convertObjCrystCrystal(crystal):
    """Copy the asymmetric unit and complete space-group symmetry
    operations."""
    from pyobjcryst._pyobjcryst import Crystal

    _check_type(crystal, "Crystal")
    # Only native accessors follow the ObjCryst clocks and component order.
    for name in (
        "GetScatteringComponentList",
        "GetLatticePar",
        "GetSpaceGroup",
        "GetNbScatterer",
        "GetScatterer",
        "GetClockMaster",
        "GetScatteringPowerRegistry",
        "GetScatteringPower",
    ):
        method = getattr(crystal, name, None)
        native = getattr(Crystal, name, None)
        if (
            getattr(method, "__self__", None) is not crystal
            or getattr(method, "__func__", None) is not native
        ):
            adapter = CrystalStructureAdapter()
            components = crystal.GetScatteringComponentList()
            _update_crystal_adapter(adapter, crystal, components)
            return adapter
    key = id(crystal)
    entry = _crystal_adapters.get(key)
    if entry is not None and entry[0]() is crystal:
        ref, clock, adapter = entry
        if clock is not None:
            master = crystal.GetClockMaster()
            if not (clock < master or master < clock):
                return adapter.clone()
    else:
        adapter = CrystalStructureAdapter()
        try:
            ref = weakref.ref(
                crystal, lambda r: _crystal_adapters.pop(key, None)
            )
        except TypeError:
            ref = None
    powers = _update_crystal_adapter(
        adapter, crystal, _scatterer_components(crystal)
    )
    if ref is None:
        return adapter
    # Keep the symmetry cache even when the powers must be read again.
    clock = (
        _clock_snapshot(crystal.GetClockMaster())
        if _master_tracks_powers(crystal, powers)
        else None
    )
    _crystal_adapters[key] = (ref, clock, adapter)
    return adapter.clone()
