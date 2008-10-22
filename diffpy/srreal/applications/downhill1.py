#!/usr/bin/env python

"""downhill1.py     simulate coloring of a structure from ideal peak areas.
Usage: downhill1.py structfile m,n,o [rmax] [seed]

This script calculates ideal pair histogram for the structfile,
then it shuffles atoms over the sites and looks for the original
configuration.  The coloring is optimized by busy downhill, i.e.,
by performing atom swaps, which give the best improvement to match
the ideal peak amplitudes.  Costs of all atom swaps are recalculated
after every iteration.

structfile  -- original structure data
m,n,o       -- size of the supercell, where coloring is performed
rmax        -- cutoff for considered pair distances, by default 10.
seed        -- seed for random number generator.
"""

__id__ = "$Id$"

import sys
import os

# run parameters
glb_rp = {
        "mno" : None,   # supercell size
        "rmax" : 10,    # default distance cutoff
        "seed" : None,  # use random random seed when None
}

class CostCalculator:
    """Functor for evaluating cost of the pair histogram, with
    respect to the reference histogram amplitudes.
    """

    def __init__(self, y0):
        """Create cost calculator and set the reference amplitudes.

        y0  -- list of reference amplitudes that will be compared
               with peak histogram.

        No return value.
        """
        self.y0 = y0[:]
        return

    def __call__(self, pairhist):
        """Cost of the pair histogram.

        pairhist -- instance of PairHistogram.  Amplitudes in pairhist
                    are compared to the reference data.

        Return mean squar difference between histogram and reference
        amplitudes.
        """
        cst = sum([(y0 - y1)**2 for y0, y1 in zip(self.y0, pairhist.y()) ])
        cst /= pairhist.countBars()
        return cst

# End of CostCalculator

# global instance of CostCalculator
glb_cost = None


def usage(brief=False):
    myname = os.path.basename(sys.argv[0])
    msg = __doc__.replace('downhill1.py', myname)
    briefmsg = msg.split('\n')[1] + '\n' + \
            "Try '%s --help' for more information" % myname
    if brief:
        print briefmsg
    else:
        print msg
    return


def busyDownhill(pairhist):
    """Perform busy downhill optimization on pair histogram.

    pairhist -- instance of PairHistogram.

    Return a copy of pairhist with optimized site coloring.
    """
    import copy
    ph1 = copy.copy(pairhist)
    print "bars:", ph1.countBars()
    print "atoms:", ph1.countAtoms()
    while True:
        c1 = glb_cost(ph1)
        color1 = ph1.getSiteColoring()
        print "%.6f, %r" % (c1, color1)
        cij2 = [(c1, 0, 0)]
        for i in range(ph1.countAtoms()):
            for j in range(i + 1, ph1.countAtoms()):
                if color1[i] == color1[j]:  continue
                ph2 = copy.copy(ph1)
                ph2.flipSiteColoring(i, j)
                cij2.append((glb_cost(ph2), i, j))
        c2, i2, j2 = min(cij2)
        if not c2 < c1:     break
        ph1.flipSiteColoring(i2, j2)
    return ph1

def processArguments(args):
    """Process program arguments and set items in the global
    parameter dictionary glb_rp and prepare the default cost
    calculator glb_cost.

    Return PairHistogram instance holding the shuffled
    supercelled structure.
    """
    numargs = len(args)
    if "-h" in args or "--help" in args:
        usage()
        sys.exit()
    elif "-V" in args or "--version" in args:
        print __id__
        sys.exit()
    elif numargs == 1:
        usage(brief=True)
        sys.exit()
    elif numargs < 3:
        emsg = "Insufficient number of arguments."
        raise RuntimeError, emsg
    # delayed imports
    from diffpy.Structure import Structure
    # required arguments
    stru0 = Structure(filename=args[1])
    # strip charge symbols
    for a in stru0:
        a.element = a.element.rstrip('012345678+-')
    glb_rp["mno"] = tuple([int(w) for w in args[2].split(',')])
    # optional arguments:
    if numargs > 3:
        glb_rp["rmax"] = float(args[3])
    if numargs > 4:
        glb_rp["seed"] = int(args[4])
    # take care of random seed
    import numpy
    if glb_rp["seed"] is not None:
        numpy.random.seed(glb_rp["seed"])
    # prepare global cost calculator
    from pairhistogram import PairHistogram
    phofstru0 = PairHistogram(stru0, glb_rp["rmax"])
    y0 = phofstru0.y()
    global glb_cost
    glb_cost = CostCalculator(y0)
    # prepare pair histogram with shuffled sites
    from diffpy.Structure.expansion import supercell
    stru_mno = supercell(stru0, glb_rp["mno"])
    ph0 = PairHistogram(stru_mno, glb_rp["rmax"])
    shuffled = numpy.random.permutation(ph0.getSiteColoring())
    ph0.setSiteColoring(shuffled)
    return ph0

def main():
    try:
        ph0 = processArguments(sys.argv)
        ph1 = busyDownhill(ph0)
    except Exception, errmsg:
        print >> sys.stderr, errmsg
        sys.exit(2)
    return 0

if __name__ == "__main__":
    main()
