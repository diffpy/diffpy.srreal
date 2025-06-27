#!/usr/bin/env python

"""Plot C60 PDFs calculated with PDFCalculator and DebyePDFCalculator.

The C60 molecule is held in a diffpy Structure object.
"""

import os
import sys

from matplotlib.pyplot import clf, plot, show

from diffpy.srreal.pdfcalculator import DebyePDFCalculator, PDFCalculator
from diffpy.structure import Structure

mydir = os.path.dirname(os.path.abspath(sys.argv[0]))
buckyfile = os.path.join(mydir, "datafiles", "C60bucky.stru")

# load C60 molecule as a diffpy.structure object
bucky = Structure(filename=buckyfile)
cfg = {
    "qmax": 25,
    "rmin": 0,
    "rmax": 10.001,
}

# calculate PDF by real-space summation
pc0 = PDFCalculator(**cfg)
r0, g0 = pc0(bucky)

# calculate PDF using Debye formula
pc1 = DebyePDFCalculator(**cfg)
r1, g1 = pc1(bucky)
gd = g0 - g1

# plot both results and the difference curve
clf()
plot(r0, g0, r1, g1, r0, gd - 1)
show()
