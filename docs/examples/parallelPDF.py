#!/usr/bin/env python

"""Demonstration of parallel PDF calculation using the multiprocessing
package.

A PDF of menthol structure is first calculated on a single core and then
on all computer CPUs.  The script then compares both results and prints
elapsed time per each calculation.
"""

import multiprocessing
import optparse
import os
import sys
import time

from matplotlib.pyplot import clf, plot, show

from diffpy.srreal.parallel import createParallelCalculator
from diffpy.srreal.pdfcalculator import PDFCalculator
from diffpy.structure import Structure

mydir = os.path.dirname(os.path.abspath(__file__))
mentholcif = os.path.join(mydir, "datafiles", "menthol.cif")
Uisodefault = 0.005

# configure options parsing
parser = optparse.OptionParser("%prog [options]\n" + __doc__)
parser.add_option(
    "--pyobjcryst",
    action="store_true",
    help="Use pyobjcryst to load the CIF file.",
)
parser.allow_interspersed_args = True
opts, args = parser.parse_args(sys.argv[1:])

# load menthol structure and make sure Uiso values are non-zero
if opts.pyobjcryst:
    # use pyobjcryst if requested by the user
    from numpy import pi
    from pyobjcryst.crystal import create_crystal_from_cif

    menthol = create_crystal_from_cif(mentholcif)
    for sc in menthol.GetScatteringComponentList():
        sp = sc.mpScattPow
        sp.Biso = sp.Biso or 8 * pi**2 * Uisodefault
else:
    # or use diffpy.structure by default
    menthol = Structure(filename=mentholcif)
    for a in menthol:
        a.Uisoequiv = a.Uisoequiv or Uisodefault

# configuration of a PDF calculator
cfg = {
    "qmax": 25,
    "rmin": 0,
    "rmax": 30,
}

# number of CPUs for parallel calculation
ncpu = multiprocessing.cpu_count()

# calculate PDF on a single core
t0 = time.time()
pc0 = PDFCalculator(**cfg)
r0, g0 = pc0(menthol)
t0 = time.time() - t0
print("Calculation time on 1 CPU: %g" % t0)

# create a pool of workers
pool = multiprocessing.Pool(processes=ncpu)

t1 = time.time()
# create a proxy parallel calculator to PDFCalculator pc0,
# that uses ncpu parallel jobs submitted via pool.imap_unordered
pc1 = createParallelCalculator(pc0, ncpu, pool.imap_unordered)
r1, g1 = pc1(menthol)
t1 = time.time() - t1
print("Calculation time on %i CPUs: %g" % (ncpu, t1))
print("Time ratio: %g" % (t0 / t1))

# plot both results and the difference curve

clf()
gd = g0 - g1
plot(r0, g0, r1, g1, r0, gd - 3)
show()
