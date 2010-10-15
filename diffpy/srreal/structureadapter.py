##############################################################################
#
# diffpy.srreal     by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2010 Trustees of the Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Pavol Juhas
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################


"""class StructureAdapter -- adapter of any structure object to the interface
    expected by srreal PairQuantity calculators

Routines:

createStructureAdapter -- create StructureAdapter from a Python object
nometa       -- create StructureAdapter with disabled _customPQConfig method
                this prevents copying of diffpy.Structure pdffit metadata
                to PDFCalculator object
nosymmetry   -- create StructureAdapter with disabled symmetry expansion.
"""

# module version
__id__ = "$Id$"

from diffpy.srreal.srreal_ext import StructureAdapter, createStructureAdapter
from diffpy.srreal.srreal_ext import nometa, nosymmetry

# End of file
