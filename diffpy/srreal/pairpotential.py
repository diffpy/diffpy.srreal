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
class Energy  -- calculator of a total pair potential energy in a structure
class PairPotential  -- calculator of one pair potential
"""

# module version
__id__ = "$Id$"


class Energy:

    def __init__(self, pair_potential=None, structure=None):
        pass

    def __call__(self):
        """Total energy of the structure.
        """
        # slow implementation
        total = 0.0
        stru = copy(self.getStructure()).clear()
        for a in self.getStructure():
            for ap in iter_atom_pairs(stru, a):
                vij = self._pair_potential(ap)
                total += vij
            stru.append(a)
        # TODO: data caching, check for modified atoms,
        # fast update of the total when only few atoms get changes.
        return total

    def derivative(self, varsymbol, order=1):
        """Derivative of the total energy with respect to given variable.

        varsymbol -- symbol of the variable
        order     -- derivative order
        """
        pass

    def setPairPotential(self, pair_potential):
        pass

    def getPairPotential(self, pair_potential):
        pass

    def setStructure(self, stru):
        pass

    def getStructure(self):
        pass

# End of TotalPairEnergy


class PairPotential:

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, atom_pair):
        """Value of pair potential for a given instance of AtomPair.
        """
        pass

    def derivative(self, atom_pair, varsymbol, order=1):
        """Derivative of the pair potential with respect to varsymbol.
        """
        pass

# End of PairPotential
