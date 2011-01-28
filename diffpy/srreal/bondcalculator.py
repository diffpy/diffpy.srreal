##############################################################################
#
# diffpy.srreal     by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2011 Trustees of the Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Pavol Juhas
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################


"""class BondCalculator -- distances between atoms in the structure.
"""

# module version
__id__ = "$Id$"

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

# inject init method that accepts keyword arguments

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

BondCalculator.__boostpython__init = BondCalculator.__init__
BondCalculator.__init__ = _init_kwargs

# End of file
