#!/usr/bin/env python

from __future__ import print_function
import multiprocessing
import numpy as np
import time
import sys
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
        dv.push(dict(pc=self.pcmaster.copy()))
        for i, vv in enumerate(rc):
            vv.execute('pc._setupParallelRun(%i, %i)' %
                    (i, self.ncpu))
        return


    def __call__(self, stru):
        self.pcmaster.setStructure(stru)
        dv.push(dict(stru=stru))
        dv.execute('pc.eval(stru); pdata = pc._getParallelData()')
        for i, data in enumerate(dv.pull('pdata')):
            if i == self.ncpu:  break
            self.pcmaster._mergeParallelData(data, self.ncpu)
        rv = (self.pcmaster.rgrid, self.pcmaster.pdf)
        return rv


r0, g0 = pc0(adpt)
ppc = ParallelCalculator(4)
r1, g1 = ppc(adpt)

print("parallel calculator consistent:", np.allclose(g0, g1))

print("time on a single thread:", timecalculator(pc0))

partimes = []
for nn in range(1, ncpu + 1):
    ppi = ParallelCalculator(nn)
    ti = timecalculator(ppi)
    partimes.append(ti)
    print("timing with %i/%i parallel jobs:" % (ppi.ncpu, ncpu), ti)

fpt = fitparalleltimes(partimes)
print(fpt)
