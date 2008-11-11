#!/usr/bin/env python

"""SCRIPTNAME   show coordination numbers in given crystal structure
Usage: SCRIPTNAME [options] structfile1 [structfile2]...

Atoms are considered neighbors when d12 < (r1 + r2)*(sqrt(2) + 1)/2.

Options:

  -f, --formula=FORMULA     chemical formula of the unit cell, must have the
                            same number of elements as number of atom sites.
                            Species are identified by a capital letter, thus
                            it is critical to use standard capitalization of
                            element symbols.  Order is important, for example,
                            Na3Cl2NaCl2.
  -l, --latpar=a,b,c,A,B,G  override lattice parameters in structfiles.
  -r, --radia=A1:r1,...     Redefine element radia.  By default use covalent
                            radia from the elements package.
  -v, --verbose             print all neighbours of every atom
      --debug               Enter python debugger after catching an exception.
  -h, --help                display this message
  -V, --version             show script version
"""

__id__ = "$Id$"

import sys
import math

# global variables
gl_doc = __doc__
gl_opts = [
        # short long
        "f:",   "formula=",
        "l:",   "latpar=",
        "r:",   "radia=",
        "v",    "verbose",
        "",     "debug",
        "h",    "help",
        "V",    "version",
]
gl_shortopts = "".join(gl_opts[0::2])
gl_longopts = gl_opts[1::2]


from colorFromOverlap import ColorFromOverlap

class CrystalCoordinationScript(ColorFromOverlap):
    """Class for running coordination number evaluation.
    """

    # script documentation string
    script_doc = gl_doc
    script_version = __id__

    COORDINATION_SCALE = (math.sqrt(2.0) + 1.0)/2.0

    def __init__(self, argv):
        """Create the script instance and prepare it for running
        
        argv    -- command line arguments

        No return value.
        """
        # define arguments that are not in the ColorFromOverlap
        # input parameters
        self.verbose = False
        # calculated parameters
        self.coordination_radia = {}
        self._filename_width = None
        # initialize from the base class
        self.super = super(CrystalCoordinationScript, self)
        ColorFromOverlap.__init__(self, argv)
        return


    def run(self):
        """Execute atom overlap optimization for all input structures.

        No return value.
        """
        self._updateExpandedFormula()
        indices = range(len(self.structures))
        for idx in indices:
            filename = self.structfiles[idx]
            stru = self.structures[idx]
            ac = self.initialAtomConflicts(stru)
            coorddata = self.evaluateCoordination(ac)
            self.printCoordination(filename, coorddata)
        return


    def initialAtomConflicts(self, stru):
        """Create AtomConflicts instance with randomly shuffled atoms.

        stru    -- initial structure, species in AtomConflicts are set
                   from expanded_formula.

        Return instance of AtomConflicts.
        """
        from diffpy.srreal.atomconflicts import AtomConflicts
        ac0 = AtomConflicts(stru)
        if self.expanded_formula:
            ac0.setSiteColoring(self.expanded_formula)
        unique_smbls = set(ac0.getSiteColoring())
        for elsmbl in unique_smbls:
            if elsmbl in self.radia:
                rsmbl = self.radia[elsmbl]
            else:
                rsmbl = ac0.atomRadius(elsmbl)
            rcoord = self.COORDINATION_SCALE * rsmbl
            self.coordination_radia[elsmbl] = rcoord
        ac0.setAtomRadiaGetter(self.coordination_radia.get)
        return ac0


    def printCoordination(self, filename, coorddata):
        """Format and output filename and associated coordination data.
        The level of detail is controlled by self.verbose.

        filename    -- structure file
        coorddata   -- coordination data returned by evaluateCoordination

        No return value.
        """
        if self._filename_width is None:
            fw0 = max([len(f) for f in self.structfiles])
            fw0 += 1        # colon symbol
            self._filename_width = ((fw0 + 4)/4)*4
        # standard brief output
        if self.verbose is False:
            ffmt = "%%-%is" % self._filename_width
            nwords = []
            neighhist = coorddata["neighhist"]
            for tpl in neighhist:
                s0 = "%ix(%s)-(" % (tpl[1], tpl[0])
                s1 = ["%ix%s" % tpl[i+1:i-1:-1] for i in range(2, len(tpl), 2)]
                s2 = s0 + ",".join(s1) + ")"
                nwords.append(s2)
            line = ffmt % (filename + ":") + "  ".join(nwords)
            print line
        # verbose output
        else:
            coloring = coorddata["coloring"]
            neighlist = coorddata["neighlist"]
            indices = range(len(coloring))
            print filename
            afmt = "%i(%s)"
            centerwords = [afmt % (idx, coloring[idx])
                    for idx in indices]
            cw0 = max([len(w) for w in centerwords])
            cw1 = ((cw0 + 4)/4) * 4
            cwfmt = "%%-%is" % cw1
            for idx in indices:
                w0 = cwfmt % (centerwords[idx] + ":")
                nwords = [afmt % (nidx, coloring[nidx])
                        for nidx in neighlist[idx]]
                line = w0 + " ".join(nwords)
                print line.rstrip()
        return


    def evaluateCoordination(self, ac):
        """Return coordination data for given AtomConflicts.

        ac  -- instance of AtomConflicts with adjusted radia.

        Return dictionary where item
        "coloring"  -- list of atom symbols on each site
        "neighlist" -- nested list of neighbor indices
        "neighhist" -- histogram of element neighborhood environments,
                       arranged as (Center, cnt, Neighbour1, cnt1, ...), e.g.,
                       [ ("Sr", 1, "O", 12),
                         ("Ti", 1, "O", 6),
                         ("O", 3, "Ti", 2, "Sr", 4) ].
        """
        cnt = ac.countAtoms()
        indices = range(cnt)
        neighlist = [[] for idx in indices]
        for ijdd in ac.getConflicts():
            i, j = ijdd[0:2]
            neighlist[i].append(j)
        for nl in neighlist:
            nl.sort()
        coloring = ac.getSiteColoring()
        order_map = dict([(el, coloring.index(el)) for el in set(coloring)])
        order_key = lambda tpl: order_map[tpl[0]]
        neighborhoods = {}
        for idx in indices:
            center = coloring[idx]
            centerneighs = [coloring[j] for j in neighlist[idx]]
            centerneighscounts = dict.fromkeys(centerneighs, 0)
            for cnsmbl in centerneighs:
                centerneighscounts[cnsmbl] += 1
            neighcnt = centerneighscounts.items()
            neighcnt.sort(key=order_key)
            keyl = [center, None] + sum([list(tpl) for tpl in neighcnt], [])
            key = tuple(keyl)
            neighborhoods[key] = neighborhoods.get(key, 0) + 1
        neighhist = [(key[0], value) + key[2:]
                for key, value in neighborhoods.iteritems()]
        neighhist.sort(key=order_key)
        rv = {  "coloring" : coloring,
                "neighlist" : neighlist,
                "neighhist" : neighhist, }
        return rv


    def usage(self, brief=False):
        """Print usage information.

        brief   -- flag for short message.

        No return value.
        """
        import os
        myname = os.path.basename(self.mypath)
        fullmsg = self.script_doc.replace('SCRIPTNAME', myname)
        briefmsg = "\n".join([
            fullmsg.split('\n')[1],
            "Try '%s --help' for more information." % myname,
            ])
        if brief:
            print briefmsg
        else:
            print fullmsg
        return


    def _processCommandLineArgs(self, argv):
        """Process command line arguments and assign input attributes.

        argv    -- command line arguments including the running program

        No return value.
        """
        import getopt
        self.mypath = argv[0]
        opts, args = getopt.gnu_getopt(argv[1:], gl_shortopts, gl_longopts)
        self.structfiles = args
        for o, a in opts:
            if o in ("-f", "--formula"):
                self.formula = a
            elif o in ("-l", "--latpar"):
                latparstrings = a.strip().split(",")
                assert len(latparstrings) == 6
                self.latpar = [float(w) for w in latparstrings]
            elif o in ("-r", "--radia"):
                words = a.strip().split(",")
                for w in words:
                    elsmbl, value = w.split(":", 1)
                    self.radia[elsmbl.strip()] = float(value)
            elif o in ("-v", "--verbose"):
                self.verbose = True
            elif o == "--debug":
                # debug is handled in main()
                pass
            elif o in ("-h", "--help"):
                self.usage()
                sys.exit()
            elif o in ("-V", "--version"):
                print self.script_version
                sys.exit()
        if not self.structfiles:
            self.usage(brief=True)
            sys.exit()
        return


    def _updateExpandedFormula(self):
        """Set expanded_formula either from the formula argument
        or from the first structure.

        No return value.
        Raise ValueError when formula expands to incorrect length.
        """
        if self.formula is not None:
            from colorFromOverlap import parseChemicalFormula
            fm = parseChemicalFormula(self.formula)
            mismatched_structures = [stru for stru in self.structures
                    if len(fm) != len(stru)]
            if mismatched_structures:
                emsg = "Incompatible length of chemical formula."
                raise ValueError, emsg
            self.expanded_formula = fm
        return


# End of class CrystalCoordinationScript

def main():
    try:
        ccs = CrystalCoordinationScript(sys.argv)
        ccs.run()
    except Exception, err:
        if "--debug" in sys.argv:
            raise
        print >> sys.stderr, err
        sys.exit(2)

if __name__ == "__main__":
    main()
