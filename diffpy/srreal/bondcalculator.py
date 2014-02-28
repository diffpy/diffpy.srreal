#!/usr/bin/env python
##############################################################################
#
# diffpy.srreal     by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2011 The Trustees of Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Pavol Juhas
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE_DANSE.txt for license information.
#
##############################################################################


"""class BondCalculator -- distances between atoms in the structure.
"""


# exported items, these also makes them show in pydoc.
__all__ = ['BondCalculator']

from diffpy.srreal.wraputils import propertyFromExtDoubleAttr
from diffpy.srreal.wraputils import setattrFromKeywordArguments
from diffpy.srreal.srreal_ext import BondCalculator

# property wrappers to C++ double attributes

BondCalculator.rmin = propertyFromExtDoubleAttr('rmin',
        '''Lower bound for the bond distances.
        [0 A]''')

BondCalculator.rmax = propertyFromExtDoubleAttr('rmax',
        '''Upper bound for the bond distances.
        [5 A]''')


# method overrides that support keyword arguments

def _init_kwargs(self, **kwargs):
    '''Create a new instance of BondCalculator.
    Keyword arguments can be used to configure
    calculator properties, for example:

    bdc = BondCalculator(rmin=1.5, rmax=2.5)

    Raise ValueError for invalid keyword argument.
    '''
    BondCalculator.__boostpython__init(self)
    setattrFromKeywordArguments(self, **kwargs)
    return


def _call_kwargs(self, structure=None, **kwargs):
    '''Return sorted bond distances in the specified structure.

    structure    -- structure to be evaluated, an instance of diffpy Structure
                    or pyobjcryst Crystal.  Reuse the last structure when None.
    kwargs       -- optional parameter settings for this calculator

    Return a sorted numpy array.
    '''
    setattrFromKeywordArguments(self, **kwargs)
    self.eval(structure)
    return self.distances


BondCalculator.__boostpython__init = BondCalculator.__init__
BondCalculator.__init__ = _init_kwargs
BondCalculator.__call__ = _call_kwargs

# End of file
