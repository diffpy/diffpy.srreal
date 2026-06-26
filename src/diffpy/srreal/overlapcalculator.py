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
"""Class OverlapCalculator -- calculator of atom overlaps in a
structure."""


# exported items, these also makes them show in pydoc.
__all__ = ["OverlapCalculator"]

from diffpy.srreal.srreal_ext import OverlapCalculator as _OverlapCalculator
from diffpy.srreal.wraputils import (
    propertyFromExtDoubleAttr,
    setattrFromKeywordArguments,
)


class OverlapCalculator(_OverlapCalculator):
    __doc__ = _OverlapCalculator.__doc__

    def __init__(self, **kwargs):
        """Create a new instance of OverlapCalculator. Keyword arguments can
        be used to configure calculator properties, for example:

        olc = OverlapCalculator(rmax=2.5)

        Raise ValueError for invalid keyword argument.
        """
        super().__init__()
        setattrFromKeywordArguments(self, **kwargs)
        return

    def __call__(self, structure=None, **kwargs):
        """Return siteSquareOverlaps per each site of the structure.

        Attributes
        ----------
        structure
            structure to be evaluated, an instance of diffpy Structure
            or pyobjcryst Crystal.  Reuse the last structure when None.
        kwargs
            optional parameter settings for this calculator

        Return a numpy array.
        """
        setattrFromKeywordArguments(self, **kwargs)
        self.eval(structure)
        return self.sitesquareoverlaps


# property wrappers to C++ double attributes

OverlapCalculator.rmin = propertyFromExtDoubleAttr(
    "rmin",
    """Lower bound for the bond distances.
        [0 A]""",
)

OverlapCalculator.rmax = propertyFromExtDoubleAttr(
    "rmax",
    """Upper bound for the bond distances.
        [5 A]""",
)

OverlapCalculator.rmaxused = propertyFromExtDoubleAttr(
    "rmaxused",
    """Effective upper bound for the bond distances.
        rmaxused equals either a double of the maximum atom radius
        in the structure or rmax.
        """,
)

# End of file
