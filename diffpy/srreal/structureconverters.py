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


"""\
Converters from other structure representations in Python to diffpy.srreal
StructureAdapter classes.
"""

from diffpy.srreal.structureadapter import RegisterStructureAdapter
from diffpy.srreal.srreal_ext import AtomicStructureAdapter
from diffpy.srreal.srreal_ext import PeriodicStructureAdapter

# Converter for Structure class from diffpy.Structure ------------------------

@RegisterStructureAdapter('diffpy.Structure.structure.Structure')
def convertDiffPyStructure(stru):
    'Adapt Structure class from diffpy.Structure package.'
    haslattice = ((1, 1, 1, 90, 90, 90) != stru.lattice.abcABG())
    isperiodic = haslattice
    hasmeta = _DiffPyStructureMetadata.hasMetadata(stru)
    if hasmeta:
        if isperiodic:
            adpt = DiffPyStructurePeriodicAdapter()
        else:
            adpt = DiffPyStructureAtomicAdapter()
        adpt._fetchMetadata(stru)
    else:
        if isperiodic:
            adpt = PeriodicStructureAdapter()
        else:
            adpt = AtomicStructureAdapter()
    _fetchDiffPyStructureData(adpt, stru)
    return adpt

# Converters for Molecule and Crystal from pyobjcryst ------------------------

from diffpy.srreal.srreal_ext import convertObjCrystMolecule
RegisterStructureAdapter(
        'pyobjcryst._pyobjcryst.Molecule', convertObjCrystMolecule)

from diffpy.srreal.srreal_ext import convertObjCrystCrystal
RegisterStructureAdapter(
        'pyobjcryst._pyobjcryst.Crystal', convertObjCrystCrystal)

# Adapter classes and helpers for diffpy.Structure class ---------------------

class _DiffPyStructureMetadata(object):

    "Base class for handling metadata information in the pdffit attribute."

    pdffit = None

    @staticmethod
    def hasMetadata(stru):
        """True if Structure object carries data in its pdffit attribute.
        """
        rv = hasattr(stru, 'pdffit') and bool(stru.pdffit)
        return rv


    def _customPQConfig(self, pqobj):
        """Apply PDF-related metadata if defined in PDFFit structure format.
        """
        pqname = type(pqobj).__name__
        if not pqname in ('PDFCalculator', 'DebyePDFCalculator'):  return
        if not self.pdffit:  return
        # scale
        envtps = pqobj.usedenvelopetypes
        if 'scale' not in envtps:
            pqobj.addEnvelope('scale')
        pqobj.scale = self.pdffit['scale']
        # spdiameter
        if "spdiameter" in self.pdffit:
            if not 'sphericalshape' in envtps:
                pqobj.addEnvelope('sphericalshape')
            pqobj.spdiameter = self.pdffit['spdiameter']
        # stepcut
        if "stepcut" in self.pdffit:
            if not 'stepcut' in envtps:
                pqobj.addEnvelope('stepcut')
            pqobj.stepcut = self.pdffit['stepcut']
        # delta1, delta2 - set these only when using JeongPeakWidth model
        if pqobj.peakwidthmodel.type() == "jeong":
            pqobj.delta1 = self.pdffit['delta1']
            pqobj.delta2 = self.pdffit['delta2']
        return


    def _fetchMetadata(self, stru):
        """Copy data from the pdffit attribute of diffpy.Structure object

        stru -- instance of Structure class from diffpy.Structure

        No return value.
        """
        # get PDF-related metadata
        self.pdffit = {}
        if self.hasMetadata(stru):
            self.pdffit.update(scale=1.0, delta1=0.0, delta2=0.0)
            self.pdffit.update(stru.pdffit)
        return

# end of class _DiffPyStructureMetadata


class DiffPyStructureAtomicAdapter(
        _DiffPyStructureMetadata, AtomicStructureAdapter):
    pass


class DiffPyStructurePeriodicAdapter(
        _DiffPyStructureMetadata, PeriodicStructureAdapter):
    pass


def _fetchDiffPyStructureData(adpt, stru):
    """Copy structure data from diffpy.Structure object to this Adapter.

    adpt -- instance of AtomicStructureAdapter or PeriodicStructureAdapter
    stru -- instance of Structure class from diffpy.Structure

    No return value.
    """
    from diffpy.srreal.srreal_ext import Atom as AdapterAtom
    # copy atoms
    del adpt[:]
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
    if hasattr(adpt, 'setLatPar'):
        adpt.setLatPar(*stru.lattice.abcABG())
        map(adpt.toCartesian, adpt)
    return

# End of file
