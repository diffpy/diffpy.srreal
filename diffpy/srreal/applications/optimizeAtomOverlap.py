#!/usr/bin/env python

"""SCRIPTNAME   assign atoms over sites to minimize their overlap
Usage: SCRIPTNAME [options] structfile1 [structfile2]...

When several structure files are specified return the one with
minimum overlap.

Options:

  -f, --formula=FORMULA     chemical formula of the unit cell, must be
                            commensurate with number of atom sites.  When
                            not specified, use composition from structfile1.
  -l, --latpar=a,b,c,A,B,G  override lattice parameters in structfiles.
  -o, --outstru=FILE        Filename for saving the best fitting structure.
      --outfmt=FORMAT       Output format for outstru, by default "discus"
                            Can be any format supported by diffpy.Structure.
  -r, --radia=A1:r1,...     Redefine element radia.  By default use covalent
                            radia from the elements package.
      --repeats=N           Number of downhill optimizations from initial
                            random configurations.  By default 5.
      --rotate              Scan all unique rotations for each of the repeats 
                            initial configurations.  Might be slow.
      --rngseed=N           Specifie integer seed for random number generator.
      --debug               Enter python debugger after catching an exception.
  -h, --help                display this message
  -V, --version             show script version
"""

__id__ = "$Id$"

import sys

# global variables
gl_doc = __doc__
gl_opts = [
        # short long
        "f:",   "formula=",
        "l:",   "latpar=",
        "o:",   "outstru=",
        "",     "outfmt=",
        "r:",   "radia=",
        "",     "repeats=",
        "",     "rotate",
        "",     "rngseed=",
        "",     "debug",
        "h",    "help",
        "V",    "version",
]
gl_shortopts = "".join(gl_opts[0::2])
gl_longopts = gl_opts[1::2]


class OptimizeAtomOverlapScript:
    """Class for running overlap optimization script.
    """

    # script documentation string
    script_doc = gl_doc
    script_version = __id__

    def __init__(self, argv):
        """Create the script instance and prepare it for running
        
        argv    -- command line arguments

        No return value.
        """
        # input parameters
        self.mypath = None
        self.structfiles = None
        self.latpar = None
        self.formula = None
        self.outstru = None
        self.outfmt = 'discus'
        self.radia = {}
        self.repeats = 5
        self.rotate = False
        self.rngseed = None
        self.debug = False
        # calculated data
        self.expanded_formula = []
        self.structures = []
        self.optimized_structures = []
        self.best_index = None
        # process command line arguments
        self._processCommandLineArgs(argv)
        self._loadStructFiles()
        self._applyLatticeParameters()
        return


    def run(self):
        """Execute atom overlap optimization for all input structures.

        No return value.
        """
        self._updateExpandedFormula()
        self._checkStructures()
        self._applyRandomSeed()
        cst_idx_ac = []
        for idx in range(len(self.structures)):
            f = self.structfiles[idx]
            stru = self.structures[idx]
            ac = self.optimizeOverlap(stru)
            optstru = ac.getStructure()
            self.optimized_structures.append(optstru)
            c = self.cost(ac)
            print f, c, ac.getSiteColoring()
            cia = c, idx, ac
            cst_idx_ac.append(cia)
        cst_idx_ac.sort()
        cbest = cst_idx_ac[0][0]
        self.best_index = cst_idx_ac[0][1]
        acbest = cst_idx_ac[0][2]
        fbest = self.structfiles[self.best_index]
        print "# BEST " + (72 - 7) * '-'
        print fbest, cbest, acbest.getSiteColoring()
        self.saveOutstru()
        return


    def cost(self, ac):
        """Calculate due to overlapping atoms.

        ac  -- instance of AtomConflicts.

        Return float.
        """
        from math import sqrt
        ddij2 = [ijdd[3]**2 for ijdd in ac.getConflicts()]
        n = ac.countAtoms() or 1
        cst = 0.5 * sum(ddij2) / n
        return cst


    def initialAtomConflicts(self, stru):
        """Create AtomConflicts instance with randomly shuffled atoms.

        stru    -- initial structure, species in AtomConflicts are set
                   from expanded_formula.

        Return instance of AtomConflicts.
        """
        import numpy
        from diffpy.srreal.atomconflicts import AtomConflicts
        ac0 = AtomConflicts(stru)
        shuffle = numpy.random.permutation(self.expanded_formula)
        ac0.setSiteColoring(shuffle)
        # setup custom radia getter if necessary
        if self.radia:
            unique_smbls = set(shuffle)
            for elsmbl in unique_smbls.difference(self.radia):
                self.radia[elsmbl] = ac0.atomRadius(elsmbl)
            ac0.setAtomRadiaGetter(self.radia.get)
        return ac0


    def downhillOverlapMinimization(self, ac0):
        """Downhill minimization of AtomConflicts instance by site flipping.

        ac0 -- initial coloring, an instance of AtomConflicts

        Return a new instance of AtomConflicts when cost could be improved.
        Return ac0 if the original configuration could not be optimized.
        """
        import copy
        col0 = ac0.getSiteColoring()
        ac1 = copy.copy(ac0)
        indices = range(ac0.countAtoms())
        if self.rotate:
            index_offsets = indices
        else:
            index_offsets = [0]
        while True:
            c1 = self.cost(ac1)
            col1 = ac1.getSiteColoring()
            cstcol = []
            did_rotation = set()
            # loop over all unique rotations of the initial coloring col0
            ac2 = copy.copy(ac1)
            for offset in index_offsets:
                col2 = tuple(col1[offset:] + col1[:offset])
                if col2 in did_rotation:
                    continue
                did_rotation.add(col2)
                ac2.setSiteColoring(col2)
                cc2 = self.cost(ac2), col2
                cstcol.append(cc2)
                for i in indices:
                    for j in indices[i+1:]:
                        if col2[i] == col2[j]: continue
                        ac3 = copy.copy(ac2)
                        ac3.flipSiteColoring(i, j)
                        cc3 = self.cost(ac3), ac3.getSiteColoring()
                        cstcol.append(cc3)
            # also try to rotate the sequence
            best_cost, best_coloring = min(cstcol)
            if not best_cost < c1:
                break
            ac1.setSiteColoring(best_coloring)
        # figure out return value:
        if self.cost(ac1) < self.cost(ac0):
            rv = ac1
        else:
            rv = ac0
        return rv


    def optimizeOverlap(self, stru):
        """Perform repeated downhill minimizations of atom overlap in stru.

        stru    -- instance of initial Structure

        Return the best of AtomConflicts instance after repeats runs.
        """
        all_acs = []
        for i in range(self.repeats):
            ac0 = self.initialAtomConflicts(stru)
            ac1 = self.downhillOverlapMinimization(ac0)
            all_acs.append(ac1)
        best_ac = min(all_acs, key=self.cost)
        return best_ac


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


    def saveOutstru(self):
        """Save the best structure from optimized_structures to outstru.

        No return value.
        """
        if self.outstru is None:
            return
        stru = self.optimized_structures[self.best_index]
        stru.write(self.outstru, format=self.outfmt)
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
            elif o in ("-o", "--outstru"):
                self.outstru = a
            elif o == "--outfmt":
                self.outfmt = a
            elif o in ("-r", "--radia"):
                words = a.strip().split(",")
                for w in words:
                    elsmbl, value = w.split(":", 1)
                    self.radia[elsmbl.strip()] = float(value)
            elif o == "--repeats":
                self.repeats = int(a)
            elif o == "--rotate":
                self.rotate = True
            elif o == "--rngseed":
                self.rngseed = int(a)
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


    def _loadStructFiles(self):
        """Load all input structfiles and assign the structures attribute.

        No return value.
        """
        from diffpy.Structure import Structure
        composition = []
        for f in self.structfiles:
            stru = Structure(filename=f)
            self.structures.append(stru)
        return


    def _checkStructures(self):
        """Verify that all structfiles have the same composition.

        No return value.
        Raise RuntimeError for structures with diffent composition.
        """
        composition = None
        for idx in range(len(self.structures)):
            f = self.structfiles[idx]
            stru = self.structures[idx]
            if not composition:
                composition = sorted([a.element for a in stru])
                continue
            strucomp = sorted([a.element for a in stru])
            if composition != strucomp:
                emsg = "Unit cell compositions differ in %s and %s." % \
                        (f, self.structfiles[0])
                raise RuntimeError, emsg
        return


    def _applyLatticeParameters(self):
        """Calculate fractional coordinates with respect to parameter latpar.
        Updates all Structure instance in the structures attribute. 
        No operation when parameter latpar was not provided.

        No return value.
        """
        if self.latpar is None:
            return
        from diffpy.Structure import Lattice
        lattice = Lattice(*self.latpar)
        for stru in self.structures:
            stru.placeInLattice(lattice)
        return


    def _updateExpandedFormula(self):
        """Set expanded_formula either from the formula argument
        or from the first structure.

        No return value.
        Raise RuntimeError when formula is incommensurate with structure.
        """
        composition0 = [a.element for a in self.structures[0]]
        if self.formula is not None:
            fm = parseChemicalFormula(self.formula)
        else:
            fm = composition0
        fmunits = len(composition0) / len(fm)
        if fmunits * len(fm) != len(composition0):
            emsg = "Formula is not commensurate with %i sites in %s." % \
                    (len(composition0), self.structfiles[0])
            raise ValueError, emsg
        self.expanded_formula = fm * fmunits
        return


    def _applyRandomSeed(self):
        """Seed the random number generator if requested.

        No return value.
        """
        if self.rngseed is None:    return
        import numpy
        numpy.random.seed(self.rngseed)
        return


# End of class OptimizeAtomOverlapScript

def parseChemicalFormula(formula):
    """Parse chemical formula and return a list of elements"""
    import re
    # remove all blanks
    fmbare = re.sub('\s', '', formula)
    if not re.match('^[A-Z]', fmbare):
        raise RuntimeError, "InvalidFormula '%s'" % fmbare
    elcnt = re.split('([A-Z][a-z]?)', fmbare)[1:]
    els = []
    for el, scnt in zip(elcnt[::2], elcnt[1::2]):
        cnt = (scnt == "") and 1 or int(scnt)
        els.extend(cnt * [el])
    return els


def main():
    try:
        oaos = OptimizeAtomOverlapScript(sys.argv)
        oaos.run()
    except Exception, err:
        if "--debug" in sys.argv:
            raise
        print >> sys.stderr, err
        sys.exit(2)

if __name__ == "__main__":
    main()
