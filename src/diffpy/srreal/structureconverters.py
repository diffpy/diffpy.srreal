#!/usr/bin/env python
##############################################################################
#
# diffpy.srreal     Complex Modeling Initiative
#                   (c) 2014 Brookhaven Science Associates,
#                   Brookhaven National Laboratory.
#                   All rights reserved.
#
# File coded by:    Pavol Juhas
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################
"""Converters from other structure representations in Python to
diffpy.srreal StructureAdapter classes."""

from diffpy.srreal.srreal_ext import (
    AtomicStructureAdapter,
    PeriodicStructureAdapter,
    convertObjCrystCrystal,
    convertObjCrystMolecule,
)
from diffpy.srreal.structureadapter import RegisterStructureAdapter

# Converters for Molecule and Crystal from pyobjcryst ------------------------


RegisterStructureAdapter(
    "pyobjcryst._pyobjcryst.Molecule", convertObjCrystMolecule
)
RegisterStructureAdapter(
    "pyobjcryst._pyobjcryst.Crystal", convertObjCrystCrystal
)

# Converter for Structure class from diffpy.structure ------------------------


# TODO prune in version 1.4.
@RegisterStructureAdapter("diffpy.Structure.structure.Structure")
@RegisterStructureAdapter("diffpy.structure.structure.Structure")
def convertDiffPyStructure(stru):
    """Adapt Structure class from diffpy.structure package."""
    haslattice = (1, 1, 1, 90, 90, 90) != stru.lattice.abcABG()
    isperiodic = haslattice
    adpt = (
        PeriodicStructureAdapter() if isperiodic else AtomicStructureAdapter()
    )
    _fetchDiffPyStructureData(adpt, stru)
    return adpt


# Adapter classes and helpers for diffpy.structure class ---------------------


# Keep the historical class paths so saved adapters can still be restored.
# Their pdffit state, if present in a pickle, is inert.
class DiffPyStructureAtomicAdapter(AtomicStructureAdapter):
    """Legacy adapter retained for pickle compatibility."""


class DiffPyStructurePeriodicAdapter(PeriodicStructureAdapter):
    """Legacy adapter retained for pickle compatibility."""


def _fetchDiffPyStructureData(adpt, stru):
    """Copy structure data from diffpy Structure object to this Adapter.

    Attributes
    ----------
    adpt
        instance of AtomicStructureAdapter or PeriodicStructureAdapter
    stru
        instance of Structure class from diffpy.structure

    No return value.
    """
    from diffpy.srreal.srreal_ext import Atom as AdapterAtom

    # copy atoms
    adpt.clear()
    adpt.reserve(len(stru))
    aa = AdapterAtom()
    for a0 in stru:
        aa.atomtype = a0.element
        aa.occupancy = a0.occupancy
        aa.anisotropy = a0.anisotropy
        # copy fractional coordinates
        aa.xyz_cartn = a0.xyz
        aa.uij_cartn = a0.U
        adpt.append(aa)
    if hasattr(adpt, "setLatPar"):
        adpt.setLatPar(*stru.lattice.abcABG())
        for aa in adpt:
            adpt.toCartesian(aa)
    return


# End of file
