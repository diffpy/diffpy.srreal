#!/usr/bin/python

from downhill1 import *

rp = {}

def usage():
    myname = os.path.basename(sys.argv[0])
    print "usage: %s structure m,n,o [rmax] [seed]" % myname
    sys.exit()

def blitzdownhill(ph1, goodflips_cost):
    cost = rp["cost"]
    c1 = cost(ph1)
    color1 = ph1.getSiteColoring()
    best_cij = (c1, 0, 0)
    print "%.6f, %r" % (c1, color1)
    flipsleft = {}
    for i, j in goodflips_cost:
        if color1[i] == color1[j]:
            continue
        ph2 = copy.copy(ph1)
        ph2.flipSiteColoring(i, j)
        c2 = cost(ph2)
        if c2 < c1:
            flipsleft[(i,j)] = c2
            if (c2, i, j) < best_cij:
                best_cij = (c2, i, j)
    if flipsleft:
        i2, j2 = best_cij[1:]
        ph1.flipSiteColoring(i2, j2)
        for k in range(ph1.countAtoms()):
            i2k = i2 < k and (i2, k) or (k, i2)
            j2k = j2 < k and (j2, k) or (k, j2)
            flipsleft.pop(i2k, None)
            flipsleft.pop(j2k, None)
    return flipsleft


def smartDownhill():
    ph1 = copy.copy(rp["ph0"])
    print "bars:", ph1.countBars()
    print "atoms:", ph1.countAtoms()
    cost = rp["cost"]
    if rp["seed"] is not None:
        numpy.random.seed(rp["seed"])
    allflips = dict.fromkeys([(i, j)
        for i in range(ph1.countAtoms())
            for j in range(i + 1, ph1.countAtoms())])
    while True:
        c1 = cost(ph1)
        print "%.6f, %r" % (c1, ph1.getSiteColoring())
        goodflips_cost = dict(allflips)
        while goodflips_cost:
            goodflips_cost = blitzdownhill(ph1, goodflips_cost)
        if not cost(ph1) < c1: break


def main():
    rp.update(processArguments(sys.argv))
    smartDownhill()

if __name__ == "__main__":
    main()
