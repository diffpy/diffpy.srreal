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


"""class BVSCalculator -- bond valence sums calculator
"""


# exported items
__all__ = ['BVSCalculator']

from diffpy.srreal.srreal_ext import BVSCalculator
from diffpy.srreal.wraputils import propertyFromExtDoubleAttr
from diffpy.srreal.wraputils import setattrFromKeywordArguments

# Property wrappers to C++ double attributes

BVSCalculator.valenceprecision = propertyFromExtDoubleAttr('valenceprecision',
        '''Cutoff value for valence contributions at long distances.
        [1e-5]''')

BVSCalculator.rmaxused = propertyFromExtDoubleAttr('rmaxused',
        '''Effective bound for bond lengths, where valence contributions
        become smaller than valenceprecission, read-only.  Always smaller or
        equal to rmax.  The value depends on ions present in the structure.
        ''')

BVSCalculator.rmin = propertyFromExtDoubleAttr('rmin',
        '''Lower bound for the summed bond lengths.
        [0 A]''')

BVSCalculator.rmax = propertyFromExtDoubleAttr('rmax',
        '''Upper bound for the summed bond lengths.  The calculation is
        actually cut off much earlier when valence contributions get below
        valenceprecission.  See also rmaxused and valenceprecission.
        [1e6 A]''')


# method overrides that support keyword arguments

def _init_kwargs(self, **kwargs):
    '''Create a new instance of BVSCalculator.
    Keyword arguments can be used to configure the calculator properties,
    for example:

    bvscalc = BVSCalculator(valenceprecision=0.001)

    Raise ValueError for invalid keyword argument.
    '''
    BVSCalculator.__boostpython__init(self)
    setattrFromKeywordArguments(self, **kwargs)
    return


def _call_kwargs(self, structure=None, **kwargs):
    '''Return bond valence sums at each atom site in the structure.

    structure    -- structure to be evaluated, an instance of diffpy Structure
                    or pyobjcryst Crystal.  Reuse the last structure when None.
    kwargs       -- optional parameter settings for this calculator

    Return an array of calculated valence sums.
    See valences for the expected values.
    '''
    setattrFromKeywordArguments(self, **kwargs)
    rv = self.eval(structure)
    return rv


BVSCalculator.__boostpython__init = BVSCalculator.__init__
BVSCalculator.__init__ = _init_kwargs
BVSCalculator.__call__ = _call_kwargs

# End of file
