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

"""class Attributes  -- wrapper to C++ class diffpy::Attributes
    A base to PairQuantity and quite a few other classes.
"""

__all__ = ['Attributes']

from diffpy.srreal.srreal_ext import Attributes

# Inject the __getattr__ and __setattr__ methods to the Attributes class

def _getattr(self, name):
    '''Lookup a C++ double attribute if standard Python lookup fails.

    Raise AttributeError if C++ double attribute does not exist.
    '''
    rv = self._getDoubleAttr(name)
    return rv

Attributes.__getattr__ = _getattr


def _setattr(self, name, value):
    '''Assign to C++ double attribute if Python attribute does not exist.
    '''
    try:
        object.__getattribute__(self, name)
    except AttributeError:
        if self._hasDoubleAttr(name):
            self._setDoubleAttr(name, value)
            return
    object.__setattr__(self, name, value)
    return

Attributes.__setattr__ = _setattr

# Helper accessor functions used by the _registerDoubleAttribute method

def _pyattrgetter(name):
    f = lambda obj: object.__getattribute__(obj, name)
    return f

def _pyattrsetter(name):
    f = lambda obj, value: object.__setattr__(obj, name, value)
    return f

# End of file
