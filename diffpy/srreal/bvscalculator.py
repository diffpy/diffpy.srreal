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


"""class BVSCalculator -- bond valence sums calculator
"""

# module version
__id__ = "$Id$"

# exported items
__all__ = ['BVSCalculator']

from diffpy.srreal.srreal_ext import BVSCalculator_ext
from diffpy.srreal.wraputils import propertyFromExtDoubleAttr
from diffpy.srreal.wraputils import setattrFromKeywordArguments


class BVSCalculator(BVSCalculator_ext):

    # Property wrappers to double attributes of the C++ BVSCalculator_ext

    valenceprecision = propertyFromExtDoubleAttr('valenceprecision',
        '''Cutoff value for valence contributions at long distances.
        [1e-5]''')

    rmaxused = propertyFromExtDoubleAttr('rmaxused',
        '''Effective bound for bond lengths, where valence contributions
        become smaller than valenceprecission, read-only.  Always smaller or
        equal to rmax.  The value depends on ions present in the structure.
        ''')

    rmin = propertyFromExtDoubleAttr('rmin',
        '''Lower bound for the summed bond lengths.
        [0 A]''')

    rmax = propertyFromExtDoubleAttr('rmax',
        '''Upper bound for the summed bond lengths.  The calculation is
        actually cut off much earlier when valence contributions get below
        valenceprecission.  See also rmaxused and valenceprecission.
        [1e6 A]''')

    # Methods

    def __init__(self, **kwargs):
        '''Create a new instance of BVSCalculator.
        Keyword arguments can be used to configure the calculator properties,
        for example:

        bvscalc = BVSCalculator(valenceprecision=0.001)

        Raise ValueError for invalid keyword argument.
        '''
        super(BVSCalculator, self).__init__()
        setattrFromKeywordArguments(self, **kwargs)
        return


    def __call__(self, structure, **kwargs):
        '''Return bond valence sums at each atom site in the structure.

        structure    -- structure to be evaluated, an instance of diffpy Structure
                        or pyobjcryst Crystal
        kwargs       -- optional parameter settings for this calculator

        Return an array of calculated valence sums.
        See valences() for the expected values.
        '''
        setattrFromKeywordArguments(self, **kwargs)
        rv = self.eval(structure)
        return rv


    def value(self):
        '''Return bond valence sums per each atom site in the structure.
        '''
        rv = super(BVSCalculator, self).value()
        return rv

# class BVSCalculator

# BVSCalculator_ext pickling support -----------------------------------------

# End of file
