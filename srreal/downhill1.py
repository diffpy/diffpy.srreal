#!/usr/bin/python
import sys
import os
import copy
import numpy
from diffpy.Structure import Structure
from pairhistogram import PairHistogram
from tmputils import createSuperCell, CostCalculator

rp = {"ph0" : None, "mno" : None, "rmax" : 10, "seed" : None, "cost" : None}

def usage():
    myname = os.path.basename(sys.argv[0])
    print "usage: %s structure m,n,o [rmax] [seed]" % myname
    sys.exit()

def busyDownhill():
    ph1 = copy.copy(rp["ph0"])
    print "bars:", ph1.countBars()
    print "atoms:", ph1.countAtoms()
    cost = rp["cost"]
    if rp["seed"] is not None:
        numpy.random.seed(rp["seed"])
    while True:
        c1 = cost(ph1)
        color1 = ph1.getSiteColoring()
        print "%.6f, %r" % (c1, color1)
        cij2 = [(c1, 0, 0)]
        for i in range(ph1.countAtoms()):
            for j in range(i + 1, ph1.countAtoms()):
                if color1[i] == color1[j]:  continue
                ph2 = copy.copy(ph1)
                ph2.flipSiteColoring(i, j)
                cij2.append((cost(ph2), i, j))
        c2, i2, j2 = min(cij2)
        if not c2 < c1:     break
        ph1.flipSiteColoring(i2, j2)

def processArguments(args):
    rv = dict(rp)
    numargs = len(args)
    if numargs < 3:
        usage()
    rv["mno"] = tuple([int(w) for w in args[2].split(',')])
    if numargs > 3: rv["rmax"] = float(args[3])
    if numargs > 4: rv["seed"] = int(args[4])
    stru = Structure(filename=args[1])
    y0 = PairHistogram(stru, rv["rmax"]).y()
    rv["cost"] = CostCalculator(y0)
    stru_mno = createSuperCell(stru, rv["mno"])
    ph0 = PairHistogram(stru_mno, rv["rmax"])
    newcol = numpy.random.permutation(ph0.getSiteColoring())
    ph0.setSiteColoring(newcol)
    rv["ph0"] = ph0
    return rv

def main():
    rp.update(processArguments(sys.argv))
    busyDownhill()

if __name__ == "__main__":
    main()
