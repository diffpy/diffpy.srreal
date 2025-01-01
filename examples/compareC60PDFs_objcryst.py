#!/usr/bin/env python

"""Plot C60 PDFs calculated with PDFCalculator and DebyePDFCalculator.

The C60 molecule are stored in a pyobjcryst object.
"""

from matplotlib.pyplot import clf, plot, show
from pyobjcryst.crystal import Crystal
from pyobjcryst.molecule import Molecule
from pyobjcryst.scatteringpower import ScatteringPowerAtom

from diffpy.srreal.pdfcalculator import DebyePDFCalculator, PDFCalculator
from diffpy.structure import Structure

# load C60 molecule as a diffpy.structure object
bucky_diffpy = Structure(filename="datafiles/C60bucky.stru")

# convert to an ObjCryst molecule
c60 = Crystal(1, 1, 1, "P1")
mc60 = Molecule(c60, "C60")
c60.AddScatterer(mc60)
# Create the scattering power object for the carbon atoms
sp = ScatteringPowerAtom("C", "C")
sp.SetBiso(bucky_diffpy[0].Bisoequiv)
for i, a in enumerate(bucky_diffpy):
    cname = "C%i" % (i + 1)
    mc60.AddAtom(a.xyz_cartn[0], a.xyz_cartn[1], a.xyz_cartn[2], sp, cname)

# PDF configuration
cfg = {
    "qmax": 25,
    "rmin": 0,
    "rmax": 10.001,
    "rstep": 0.05,
}

# calculate PDF by real-space summation
pc0 = PDFCalculator(**cfg)
r0, g0 = pc0(c60)

# calculate PDF using Debye formula
pc1 = DebyePDFCalculator(**cfg)
r1, g1 = pc1(c60)
gd = g0 - g1

# plot both results and the difference curve
clf()
plot(r0, g0, r1, g1, r0, gd - 1)
show()
