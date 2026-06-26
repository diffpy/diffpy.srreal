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
"""Class BondCalculator -- distances between atoms in the structure."""


# exported items, these also makes them show in pydoc.
__all__ = ["BondCalculator"]

from diffpy.srreal.srreal_ext import BondCalculator as _BondCalculator
from diffpy.srreal.wraputils import (
    propertyFromExtDoubleAttr,
    setattrFromKeywordArguments,
)


class BondCalculator(_BondCalculator):
    __doc__ = _BondCalculator.__doc__

    def __init__(self, **kwargs):
        """Create a new instance of BondCalculator. Keyword arguments can be
        used to configure calculator properties, for example:

        bdc = BondCalculator(rmin=1.5, rmax=2.5)

        Raise ValueError for invalid keyword argument.
        """
        super().__init__()
        setattrFromKeywordArguments(self, **kwargs)
        return

    def __call__(self, structure=None, **kwargs):
        """Return sorted bond distances in the specified structure.

        Attributes
        ----------
        structure
            structure to be evaluated, an instance of diffpy Structure
            or pyobjcryst Crystal.  Reuse the last structure when None.
        kwargs
            optional parameter settings for this calculator

        Return a sorted numpy array.
        """
        setattrFromKeywordArguments(self, **kwargs)
        self.eval(structure)
        return self.distances


# property wrappers to C++ double attributes

BondCalculator.rmin = propertyFromExtDoubleAttr(
    "rmin",
    """Lower bound for the bond distances.
        [0 A]""",
)

BondCalculator.rmax = propertyFromExtDoubleAttr(
    "rmax",
    """Upper bound for the bond distances.
        [5 A]""",
)


# End of file
