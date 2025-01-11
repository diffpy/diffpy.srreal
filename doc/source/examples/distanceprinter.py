#!/usr/bin/env python

"""Demonstration of using PairQuantity class for a printout of pair distances
in periodic and non-periodic structures."""

from diffpy.srreal.pairquantity import PairQuantity
from diffpy.structure import Structure


class DistancePrinter(PairQuantity):
    """This PairQuantity class simply prints the visited pair distances and the
    indices of the contributing atoms."""

    def _resetValue(self):
        self.count = 0

    def _addPairContribution(self, bnds, sumscale):
        self.count += bnds.multiplicity() * sumscale / 2.0
        print("%i %g %i %i" % (self.count, bnds.distance(), bnds.site0(), bnds.site1()))
        return


# class DistancePrinter

# define nickel structure data
nickel_discus_data = """
title   Ni
spcgr   P1
cell    3.523870,  3.523870,  3.523870, 90.000000, 90.000000, 90.000000
ncell          1,         1,         1,         4
atoms
NI          0.00000000        0.00000000        0.00000000       0.1000
NI          0.00000000        0.50000000        0.50000000       0.1000
NI          0.50000000        0.00000000        0.50000000       0.1000
NI          0.50000000        0.50000000        0.00000000       0.1000
"""

nickel = Structure()
nickel.readStr(nickel_discus_data, format="discus")

bucky = Structure(filename="datafiles/C60bucky.stru", format="discus")
distprint = DistancePrinter()
distprint._setDoubleAttr("rmax", 10)


def get_pyobjcryst_sphalerite():
    from pyobjcryst import loadCrystal

    crst = loadCrystal("datafiles/sphalerite.cif")
    return crst


def main():
    s = input("Enter rmin: ")
    if s.strip():
        distprint.rmin = float(s)
    print("rmin = %g" % distprint.rmin)
    s = input("Enter rmax: ")
    if s.strip():
        distprint.rmax = float(s)
    print("rmax = %g" % distprint.rmax)
    print()
    linesep = 78 * "-"
    # C60bucky
    print(linesep)
    input("Press enter for distances in C60 molecule.")
    distprint.eval(bucky)
    # nickel
    print(linesep)
    input("Press enter for distances in a nickel crystal.")
    distprint.eval(nickel)
    # objcryst sphalerite
    print(linesep)
    input("Press enter for distances in objcryst loaded sphalerite.cif.")
    crst = get_pyobjcryst_sphalerite()
    distprint.eval(crst)
    print(linesep)
    input("Press enter for distances in diffpy.structure sphalerite.cif.")
    crst = Structure(filename="datafiles/sphalerite.cif")
    distprint.eval(crst)
    return


if __name__ == "__main__":
    main()
