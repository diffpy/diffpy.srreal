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


"""\
Class for configuring PDF profile function:
    PeakProfile
    GaussianProfile, CroppedGaussianProfile
"""


# exported items
__all__ = '''
    PeakProfile
    GaussianProfile
    CroppedGaussianProfile
    '''.split()

from diffpy.srreal import _final_imports
from diffpy.srreal.srreal_ext import PeakProfile
from diffpy.srreal.srreal_ext import GaussianProfile, CroppedGaussianProfile
from diffpy.srreal.wraputils import propertyFromExtDoubleAttr
from diffpy.srreal.wraputils import _pickle_getstate, _pickle_setstate

# class PeakProfile ----------------------------------------------------------

# pickling support

def _peakprofile_create(s):
    from diffpy.srreal.srreal_ext import _PeakProfile_fromstring
    return _PeakProfile_fromstring(s)

def _peakprofile_reduce(self):
    from diffpy.srreal.srreal_ext import _PeakProfile_tostring
    args = (_PeakProfile_tostring(self),)
    rv = (_peakprofile_create, args)
    return rv

def _peakprofile_reduce_with_state(self):
    rv = _peakprofile_reduce(self) + (self.__getstate__(),)
    return rv

# inject pickle methods to the base class

PeakProfile.__reduce__ = _peakprofile_reduce_with_state
PeakProfile.__getstate__ = _pickle_getstate
PeakProfile.__setstate__ = _pickle_setstate

# Derived C++ classes are pickled without dictionary

GaussianProfile.__reduce__ = _peakprofile_reduce
CroppedGaussianProfile.__reduce__ = _peakprofile_reduce

# add attribute wrappers for PeakProfile and derived classes

PeakProfile.peakprecision = propertyFromExtDoubleAttr('peakprecision',
    '''Profile amplitude relative to the peak maximum for evaluating peak
    bounds xboundlo and xboundhi. [3.33e-6 unitless]
    ''')

# Import delayed tweaks of the extension classes.

_final_imports.import_now()
del _final_imports

# End of file
