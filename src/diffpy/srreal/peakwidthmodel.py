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
Classes for configuring peak width evaluation in PDF calculations:
    PeakWidthModel,
    ConstantPeakWidth, DebyeWallerPeakWidth, JeongPeakWidth
"""


# exported items
__all__ = [
    'PeakWidthModel',
    'ConstantPeakWidth',
    'DebyeWallerPeakWidth',
    'JeongPeakWidth'
]

from diffpy.srreal import _final_imports
from diffpy.srreal.srreal_ext import PeakWidthModel, ConstantPeakWidth
from diffpy.srreal.srreal_ext import DebyeWallerPeakWidth, JeongPeakWidth
from diffpy.srreal.wraputils import propertyFromExtDoubleAttr

# class PeakWidthModel -------------------------------------------------------

# add attribute wrappers for the derived classes

ConstantPeakWidth.width = propertyFromExtDoubleAttr('width',
    '''Constant FWHM value returned by this model.
    ''')

ConstantPeakWidth.bisowidth = propertyFromExtDoubleAttr('bisowidth',
    '''Equivalent uniform Biso for this constant `width`.
    ''')

ConstantPeakWidth.uisowidth = propertyFromExtDoubleAttr('uisowidth',
    '''Equivalent uniform Uiso for this constant `width`.
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
