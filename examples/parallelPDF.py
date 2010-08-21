#!/usr/bin/env python

"""Demonstration of parallel PDF calculation using the multiprocessing
package.  A PDF of menthol structure is first calculated on a single core
and then on all computer CPUs.  The script then compares both results
and prints elapsed time per each calculation.
"""

# NOTE: This example uses low-level interface to parallel evaluation.
# There should be something more user friendly soon.

import os
import time
import multiprocessing
from diffpy.Structure import Structure
from diffpy.srreal.pdfcalculator import PDFCalculator
from diffpy.srreal.parallel import createParallelCalculator

mydir = os.path.dirname(os.path.abspath(__file__))
mentholcif = os.path.join(mydir, 'datafiles', 'menthol.cif')

# load menthol structure and make sure Uiso values are non-zero
menthol = Structure(filename=mentholcif)
for a in menthol:
    a.Uisoequiv = a.Uisoequiv or 0.005

# configuration of a PDF calculator
cfg = { 'qmax' : 25,
        'rmin' : 0,
        'rmax' : 30,
}

# number of CPUs for parallel calculation
ncpu = multiprocessing.cpu_count()

# calculate PDF on a single core
t0 = time.time()
pc0 = PDFCalculator(**cfg)
r0, g0 = pc0(menthol)
t0 = time.time() - t0
print "Calculation time on 1 CPU: %g" % t0

# create a pool of workers
pool = multiprocessing.Pool(processes=ncpu)

t1 = time.time()
# create a proxy parrallel calculator to PDFCalculator pc0,
# that uses ncpu parallel jobs submitted via pool.imap_unordered
pc1 = createParallelCalculator(pc0, ncpu, pool.imap_unordered)
r1, g1 = pc1(menthol)
t1 = time.time() - t1
print "Calculation time on %i CPUs: %g" % (ncpu, t1)
print "Time ratio: %g" % (t0 / t1)

# plot both results and the difference curve
from pylab import plot, show, clf, draw
clf()
gd = g0 - g1
plot(r0, g0, r1, g1, r0, gd - 3)
show()
