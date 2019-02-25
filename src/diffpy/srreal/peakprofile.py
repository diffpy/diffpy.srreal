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

"""
Class for configuring PDF profile function:
    PeakProfile
    GaussianProfile, CroppedGaussianProfile
"""


# exported items
__all__ = [
    'PeakProfile',
    'GaussianProfile',
    'CroppedGaussianProfile'
]

from diffpy.srreal import _final_imports
from diffpy.srreal.srreal_ext import PeakProfile
from diffpy.srreal.srreal_ext import GaussianProfile, CroppedGaussianProfile
from diffpy.srreal.wraputils import propertyFromExtDoubleAttr

# class PeakProfile ----------------------------------------------------------

# disable dictionary pickling for wrapped C++ classes

GaussianProfile.__getstate_manages_dict__ = None
CroppedGaussianProfile.__getstate_manages_dict__ = None

# add attribute wrappers for PeakProfile and derived classes

PeakProfile.peakprecision = propertyFromExtDoubleAttr('peakprecision',
    '''Profile amplitude relative to the peak maximum for evaluating peak
    bounds xboundlo and xboundhi. [3.33e-6 unitless]
    ''')

# Import delayed tweaks of the extension classes.

_final_imports.import_now()
del _final_imports

# End of file
