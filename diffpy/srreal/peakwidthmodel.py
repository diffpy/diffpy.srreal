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
Classes for configuring peak width evaluation in PDF calculations:
    PeakWidthModel,
    ConstantPeakWidth, DebyeWallerPeakWidth, JeongPeakWidth
"""


# exported items
__all__ = '''
    PeakWidthModel
    ConstantPeakWidth
    DebyeWallerPeakWidth
    JeongPeakWidth
    '''.split()

from diffpy.srreal import _final_imports
from diffpy.srreal.srreal_ext import PeakWidthModel, ConstantPeakWidth
from diffpy.srreal.srreal_ext import DebyeWallerPeakWidth, JeongPeakWidth
from diffpy.srreal.wraputils import propertyFromExtDoubleAttr
from diffpy.srreal.wraputils import _pickle_getstate, _pickle_setstate

# class PeakWidthModel -------------------------------------------------------

# pickling support

def _peakwidthmodel_create(s):
    from diffpy.srreal.srreal_ext import _PeakWidthModel_fromstring
    return _PeakWidthModel_fromstring(s)

def _peakwidthmodel_reduce(self):
    from diffpy.srreal.srreal_ext import _PeakWidthModel_tostring
    args = (_PeakWidthModel_tostring(self),)
    rv = (_peakwidthmodel_create, args)
    return rv

def _peakwidthmodel_reduce_with_state(self):
    rv = _peakwidthmodel_reduce(self) + (self.__getstate__(),)
    return rv

# inject pickle methods to the base class

PeakWidthModel.__reduce__ = _peakwidthmodel_reduce_with_state
PeakWidthModel.__getstate__ = _pickle_getstate
PeakWidthModel.__setstate__ = _pickle_setstate

# Derived C++ classes are pickled without dictionary

ConstantPeakWidth.__reduce__ = _peakwidthmodel_reduce
DebyeWallerPeakWidth.__reduce__ = _peakwidthmodel_reduce
JeongPeakWidth.__reduce__ = _peakwidthmodel_reduce

# add attribute wrappers for the derived classes

ConstantPeakWidth.width = propertyFromExtDoubleAttr('width',
    '''Constant FWHM value returned by this model.
    ''')

JeongPeakWidth.delta1 = propertyFromExtDoubleAttr('delta1',
        'Coefficient for (1/r) contribution to the peak sharpening.')

JeongPeakWidth.delta2 = propertyFromExtDoubleAttr('delta2',
        'Coefficient for (1/r**2) contribution to the peak sharpening.')

JeongPeakWidth.qbroad = propertyFromExtDoubleAttr('qbroad',
        'PDF peak broadening from increased intensity noise at high Q.')

# Import delayed tweaks of the extension classes.

_final_imports.import_now()
del _final_imports

# End of file
