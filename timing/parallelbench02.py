#!/usr/bin/env python

from __future__ import print_function
import multiprocessing
import numpy as np
import time
from diffpy.Structure import Structure
from diffpy.srreal.pdfcalculator import PDFCalculator
from diffpy.srreal.structureadapter import createStructureAdapter
from diffpy.srreal.parallel import createParallelCalculator
from timingutils import fitparalleltimes

mentholcif = 'menthol.cif'
Uisodefault = 0.01

# or use diffpy.Structure by default
menthol = Structure(filename=mentholcif)
menthol.Uisoequiv = Uisodefault
adpt = createStructureAdapter(menthol)

pdfcstd = PDFCalculator(rmax=30, qmax=25)
pc0 = pdfcstd.copy()

def partialresult((pcmaster, cpuindex, ncpu)):
    pc = pcmaster.copy()
    pc._setupParallelRun(cpuindex, ncpu)
    pc.eval()
    return pc._getParallelData()


def timecalculator(pc, repeats=1):
    t0 = time.time()
    for i in range(repeats):
        pc(adpt)
    t1 = time.time()
    return (t1 - t0) / repeats


from ipyparallel import Client
rc = Client()
dv = rc[:]
ncpu = len(rc)


class ParallelCalculator(object):

    def __init__(self, ncpu):
        self.pcmaster = pdfcstd.copy()
        self.ncpu = ncpu
        return


    def __call__(self, stru):
        self.pcmaster.setStructure(stru)
        arglist = [(self.pcmaster, i, self.ncpu)
                for i in range(self.ncpu)]
        for data in dv.imap(partialresult, arglist):
            self.pcmaster._mergeParallelData(data, self.ncpu)
        rv = (self.pcmaster.rgrid, self.pcmaster.pdf)
        return rv


print("time on a single thread:", timecalculator(pc0))

partimes = []
for nn in range(1, ncpu + 1):
    ppi = ParallelCalculator(nn)
    ti = timecalculator(ppi)
    partimes.append(ti)
    print("timing with %i/%i parallel jobs:" % (ppi.ncpu, ncpu), ti)

fpt = fitparalleltimes(partimes)
print(fpt)
