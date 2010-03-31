#!/usr/bin/env python

"""Plot C60 PDFs calculated with PDFCalculator and DebyePDFCalculator.
"""

from pylab import plot, show, clf, draw

from diffpy.Structure import Structure
from diffpy.srreal.pdfcalculator import PDFCalculator
from diffpy.srreal.srreal_ext import DebyePDFCalculator_ext

bucky = Structure(filename='C60bucky.stru')
cfg = { 'qmax' : 25,
        'rmin' : 0,
        'rmax' : 10.001,
        'rstep' : 0.05,
}

pc0 = PDFCalculator(**cfg)
r0, g0 = pc0(bucky)

pc1 = DebyePDFCalculator_ext()
for k, v in cfg.items():
    pc1._setDoubleAttr(k, v)
pc1.eval(bucky)
r1 = pc1.getRgrid()
g1 = pc1.getPDF()
gd = g0 - g1

clf()
plot(r0, g0, r1, g1, r0, gd - 1)
show()
