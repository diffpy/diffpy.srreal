#!/usr/bin/env python
"""A small test for the bindings."""

from pyobjcryst import *
from __init__ import *
import numpy

def getNi():

    pi = numpy.pi
    c = Crystal(3.52, 3.52, 3.52, "225")
    sp = ScatteringPowerAtom("Ni", "Ni")
    sp.SetBiso(8*pi*pi*0.003)
    atomp = Atom(0, 0, 0, "Ni", sp)
    c.AddScatteringPower(sp)
    c.AddScatterer(atomp)
    return c

def printBonds():

    c = getNi()
    bi = BondIterator(c, 0, 4)
    getUnitCell(c)

    scl = c.GetScatteringComponentList()
    for sc in scl:
        bi.setScatteringComponent(sc)
        bi.rewind()
        while(not bi.finished()):
            bp = bi.getBondPair()
            print bp
            bi.next()
    return

def printPDF():
    
    c = getNi()
    rvals = numpy.arange(0, 10, 0.05)
    biter = BondIterator(c)
    bwcalc = JeongBWCalculator()

    pdfcalc = PDFCalculator(biter, bwcalc)
    pdfcalc.setCalculationPoints(rvals)
    pdf = pdfcalc.getPDF()

    from pylab import plot, show
    plot(rvals, pdf)
    show()

if __name__ == "__main__":

    printPDF()
