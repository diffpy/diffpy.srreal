#!/usr/bin/env python

from __future__ import print_function
import multiprocessing
import numpy as np
import timeit
from diffpy.Structure import Structure
from diffpy.srreal.pdfcalculator import PDFCalculator
from diffpy.srreal.structureadapter import createStructureAdapter
from diffpy.srreal.parallel import createParallelCalculator

mentholcif = 'menthol.cif'
Uisodefault = 0.01

# or use diffpy.Structure by default
menthol = Structure(filename=mentholcif)
menthol.Uisoequiv = Uisodefault
adpt = createStructureAdapter(menthol)

pdfcstd = PDFCalculator(rmax=30, qmax=25)
pc0 = pdfcstd.copy()

ncpu = multiprocessing.cpu_count()
pool = multiprocessing.Pool(processes=ncpu)
ppc = [createParallelCalculator(pdfcstd.copy(), nn, pool.imap_unordered)
        for nn in range(1, ncpu + 1)]

def timecalculator(pc):
    import __main__
    __main__._thecalculator = pc
    t = timeit.repeat('pc(adpt)',
            setup='from __main__ import _thecalculator as pc, adpt as adpt',
            repeat=1, number=1)
    return np.mean(t)

print("time on a single thread:", timecalculator(pc0))

for ppi in ppc:
    print("timing with %i/%i parallel jobs:" % (ppi.ncpu, ncpu),
            timecalculator(ppi))
