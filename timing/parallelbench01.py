#!/usr/bin/env python

from __future__ import print_function
import multiprocessing
import numpy as np
import time
from diffpy.Structure import Structure
from diffpy.srreal.pdfcalculator import PDFCalculator
from diffpy.srreal.structureadapter import createStructureAdapter
from diffpy.srreal.parallel import createParallelCalculator
from timingutils import fitparalleltimes, timecalculator

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

def timecalculator(pc, repeats=1):
    t0 = time.time()
    for i in range(repeats):
        pc(adpt)
    t1 = time.time()
    return (t1 - t0) / repeats


print("time on a single thread:", timecalculator(pc0))

partimes = []
for nn in range(1, ncpu + 1):
    ppi = createParallelCalculator(pdfcstd.copy(), nn, pool.imap_unordered)
    ti = timecalculator(ppi)
    partimes.append(ti)
    print("timing with %i/%i parallel jobs:" % (ppi.ncpu, ncpu), ti)

fpt = fitparalleltimes(partimes)
print(fpt)
