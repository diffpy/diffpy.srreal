#!/usr/bin/env python
##############################################################################
#
# diffpy.srreal     by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2010 The Trustees of Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Pavol Juhas
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE_DANSE.txt for license information.
#
##############################################################################
"""Class ScatteringFactorTable -- scattering factors for atoms, ions and
isotopes."""


# exported items, these also makes them show in pydoc.
__all__ = [
    "ScatteringFactorTable",
    "SFTXray",
    "SFTElectron",
    "SFTNeutron",
    "SFTElectronNumber",
    "SFAverage",
]

from diffpy.srreal.sfaverage import SFAverage
from diffpy.srreal.srreal_ext import (
    ScatteringFactorTable,
    SFTElectron,
    SFTElectronNumber,
    SFTNeutron,
    SFTXray,
)

# Pickling Support -----------------------------------------------------------

# disable dictionary pickling for wrapped C++ classes

SFTXray.__getstate_manages_dict__ = None
SFTElectron.__getstate_manages_dict__ = None
SFTNeutron.__getstate_manages_dict__ = None
SFTElectronNumber.__getstate_manages_dict__ = None

# End of file
