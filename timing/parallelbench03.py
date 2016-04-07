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
import threading
import numpy

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


ncpu = multiprocessing.cpu_count()

dataList = []

def appendData(data):
    dataList.append(data)

class threadedCalculator(threading.Thread):

    data = []

    def __init__(self, cpuindex, ncpu, workerPc, callback):
        threading.Thread.__init__(self)
        self.callback = callback
        self.pc = workerPc
        self.pc._setupParallelRun(cpuindex, ncpu)

    def run(self):
        self.pc.eval()
        self.callback(self.pc._getParallelData())


class ParallelCalculator(object):

    def __init__(self, ncpu):
        self.pcmaster = pdfcstd.copy()
        self.ncpu = ncpu
        return


    def __call__(self, stru):
        self.pcmaster.setStructure(stru)

        threads = []
        global dataList
        dataList = []

        workerPc = self.pcmaster.copy()

        for i in range(0, self.ncpu):
            worker = threadedCalculator(cpuindex=i, ncpu=self.ncpu, workerPc=workerPc, callback=appendData)
            threads.append(worker)
            worker.start()

        for worker in threads:
            worker.join()

        for data in dataList:
            self.pcmaster._mergeParallelData(data, self.ncpu)

        rv = (self.pcmaster.rgrid, self.pcmaster.pdf)
        return rv


print("time on a single thread:", timecalculator(pc0))

print(pc0.value)

partimes = []
for nn in range(1, ncpu + 1):
    ppi = ParallelCalculator(nn)
    ti = timecalculator(ppi)

    partimes.append(ti)
    print("timing with %i/%i parallel jobs:" % (ppi.ncpu, ncpu), ti)
    print(numpy.allclose(pc0.pdf, ppi.pcmaster.pdf))

fpt = fitparalleltimes(partimes)
print(fpt)
