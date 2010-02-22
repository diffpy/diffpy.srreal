########################################################################
#
# diffpy.srreal     by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2008 Trustees of the Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Pavol Juhas
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
########################################################################


"""\
class AtomPair  -- instance of atom pair with other atom in the structure
iter_atom_pairs -- iterator over pairs of a selected atom
"""

# module version
__id__ = "$Id$"


def iter_atom_pairs(stru, atom):
    """Return iterator of all pairs of atom with atoms in stru.
    atom may be present in stru.
    """
    pass


class AtomPair:
    """Instance of atom pair data.

    Public data:

    ri  -- absolute Cartesian coordinates of the first atom
    rj  -- absolute Cartesian coordinates of the second atom
    label_i -- string label of the first atom
    label_j -- string label of the second atom
    count -- number of equivalent pairs in the structure
    """

    # Public methods:

    def rij():
        """Cartesian dissplacement of atom i from atom j.
        """
        pass

    def rijNorm():
        """Distance from atom i to atom j.
        """
        pass

    def atom_i():
        """Reference to the first atom in the pair.
        """
        pass

    def atom_j():
        """Reference to the second atom in the pair.
        """
        pass

# End of AtomPair
