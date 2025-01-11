#!/usr/bin/env python

"""Demonstration of using PairQuantity class for calculation of Lennard Jones
potential.

Vij = 4 * ( rij ** -12  -  rij ** -6 )
"""

import sys

from diffpy.srreal.pairquantity import PairQuantity
from diffpy.structure import Structure


class LennardJonesCalculator(PairQuantity):

    # Initialization.  The size of the result array is always 1.
    def __init__(self):
        PairQuantity.__init__(self)
        self._resizeValue(1)
        return

    def __call__(self, structure):
        """Return LJ potential for a specified structure."""
        values = self.eval(structure)
        return values[0]

    def _addPairContribution(self, bnds, sumscale):
        """Add Lennard-Jones contribution from a single pair of atoms.

        bnds     -- a BaseBondGenerator instance that contains information
                    about the current pair of atom sites.
        sumscale -- integer multiplicity of the current pair.  It may
                    be negative in case of fast value updates.

        No return value.  Updates _value, the internal results array.
        """
        rij = bnds.distance()
        ljij = 4 * (rij**-12 - rij**-6)
        self._value[0] += sumscale * ljij / 2.0
        return


# class LennardJonesCalculator


def main():
    # load structure from a specified file, by default "lj50.xyz"
    filename = len(sys.argv) > 1 and sys.argv[1] or "lj50.xyz"
    stru = Structure()
    stru.read(filename)
    # create an instance of LennardJonesCalculator
    ljcalc = LennardJonesCalculator()
    # calculate and print the LJ potential.
    print("LJ potential of %s is %g" % (filename, ljcalc(stru)))


if __name__ == "__main__":
    main()
