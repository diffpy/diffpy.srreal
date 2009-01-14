#!/usr/bin/env python
"""A small test for the bindings."""

from pyobjcryst import *
from __init__ import *
from numpy import pi

def getNi():

    c = Crystal(3.52, 3.52, 3.52, "225")
    sp = ScatteringPowerAtom("Ni", "Ni")
    sp.SetBiso(8*pi*pi*0.003)
    atomp = Atom(0, 0, 0, "Ni", sp)
    c.AddScatterer(atomp)
    return c

def printBonds():

    c = getNi();
    bi = BondIterator(c, 0, 10)
    getUnitCell(c);

    scl = c.GetScatteringComponentList();
    for sc in scl:
        print sc
        bi.setScatteringComponent(sc)
        bi.rewind()
        while(not bi.finished()):
            bp = bi.getBondPair()
            print bp
            bi.next()

if __name__ == "__main__":

    printBonds()
