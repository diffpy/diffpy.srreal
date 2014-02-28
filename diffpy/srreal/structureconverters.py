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

import numpy
from diffpy.srreal.structureadapter import RegisterStructureAdapter
from diffpy.srreal.srreal_ext import PeriodicStructureAdapter
from diffpy.srreal.srreal_ext import nosymmetry
from diffpy.srreal.srreal_ext import Atom as AdapterAtom

# Converter for Structure class from diffpy.Structure ------------------------

@RegisterStructureAdapter('diffpy.Structure.structure.Structure')
def convertDiffPyStructure(stru):
    'Adapt Structure class from diffpy.Structure package.'
    rv = DiffPyStructureAdapter()
    rv._fetchStructureData(stru)
    if numpy.allclose(stru.lattice.base, numpy.identity(3)):
        rv = nosymmetry(rv)
    return rv

# Converters for Molecule and Crystal from pyobjcryst ------------------------

from diffpy.srreal.srreal_ext import convertObjCrystMolecule
RegisterStructureAdapter(
        'pyobjcryst._pyobjcryst.Molecule', convertObjCrystMolecule)

from diffpy.srreal.srreal_ext import convertObjCrystCrystal
RegisterStructureAdapter(
        'pyobjcryst._pyobjcryst.Crystal', convertObjCrystCrystal)

# Define adapter class for diffpy.Structure class ----------------------------

class DiffPyStructureAdapter(PeriodicStructureAdapter):

    pdffit = None

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


    def _fetchStructureData(self, stru):
        """Copy structure data from Python class to this Adapter.

        stru -- instance of Structure class from diffpy.Structure

        No return value.
        """
        # get PDF-related metadata
        self.pdffit = {}
        if hasattr(stru, 'pdffit') and stru.pdffit:
            self.pdffit.update(scale=1.0, delta1=0.0, delta2=0.0)
            self.pdffit.update(stru.pdffit)
        # lattice
        self.setLatPar(*stru.lattice.abcABG())
        # copy atoms
        del self[:]
        self.reserve(len(stru))
        aa = AdapterAtom()
        for a0 in stru:
            aa.atomtype = a0.element
            aa.occupancy = a0.occupancy
            aa.anisotropy = a0.anisotropy
            # copy fractional coordinates and then convert to Cartesian
            aa.xyz_cartn = a0.xyz
            aa.uij_cartn = a0.U
            self.toCartesian(aa)
            self.append(aa)
        return

# End of class DiffPyStructureAdapter

# End of file
