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
                            Write to the standard output when "-" and suppress
                            any other output.
      --outfmt=FORMAT       Output format for outstru, by default "discus"
                            Can be any format supported by diffpy.Structure.
  -r, --radia=A1:r1,...     Redefine element radia.  By default use covalent
                            radia from the elements package.
      --repeats=N           Number of downhill optimizations from initial
                            random configurations.  By default 5.
      --rotate              Scan all unique rotations for each of the repeats 
                            initial configurations.  Might be slow.
      --rngseed=N           Specify integer seed for random number generator.
      --debug               Enter python debugger after catching an exception.
  -h, --help                display this message
  -V, --version             show script version
"""

__id__ = "$Id$"

import sys
import logging

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

# output logger
outlog = logging.getLogger("colorFromOverlap")
outlog.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(logging.Formatter("%(message)s"))
outlog.addHandler(ch)
del ch


class ColorFromOverlap(object):
    """Class for running overlap optimization script.
    """

    # script documentation string
    script_doc = gl_doc
    script_version = __id__
    script_options = gl_opts

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
        self.optimized_costs = []
        self.optimized_structures = []
        # process command line arguments
        self._processCommandLineArgs(argv)
        self._loadDataFiles()
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
        getcoloring = lambda stru: [a.element for a in stru]
        for idx in range(len(self.structures)):
            f = self.structfiles[idx]
            stru = self.structures[idx]
            optcost, optstru = self.optimizeColoring(stru)
            self.optimized_costs.append(optcost)
            self.optimized_structures.append(optstru)
            outlog.info("%s %g %r", f, optcost, getcoloring(optstru))
        cbest, fbest, strubest = min(zip(
            self.optimized_costs, self.structfiles, self.optimized_structures))
        outlog.info("# BEST " + (78 - 7) * '-')
        outlog.info("%s %g %r", fbest, cbest, getcoloring(strubest))
        self.saveOutstru()
        return


    def cost(self, ac):
        """Calculate cost due to overlapping atoms.

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
        from diffpy.srreal01.atomconflicts import AtomConflicts
        ac0 = AtomConflicts(stru)
        shuffle = numpy.random.permutation(self.expanded_formula)
        ac0.setSiteColoring(shuffle)
        # setup custom radia getter if necessary
        if self.radia:
            unique_smbls = set(shuffle)
            for elsmbl in unique_smbls.difference(self.radia):
                bare_elsmbl = elsmbl.rstrip('12345678-+')
                if bare_elsmbl in self.radia:
                    self.radia[elsmbl] = self.radia[bare_elsmbl]
                else:
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


    def optimizeColoring(self, stru):
        """Perform repeated downhill minimizations of atom overlap in stru.

        stru    -- instance of initial Structure

        Return a tuple of (cost, colored_structure).
        """
        # use initialAtomConflicts to setup proper atom radii
        ac = self.initialAtomConflicts(stru)
        # undo site shuffling
        ac.setStructure(stru)
        coststru = [(self.cost(ac), ac.getStructure())]
        for i in range(self.repeats):
            ac0 = self.initialAtomConflicts(stru)
            ac1 = self.downhillOverlapMinimization(ac0)
            cs = (self.cost(ac1), ac1.getStructure())
            coststru.append(cs)
        rv = min(coststru)
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


    def saveOutstru(self):
        """Save the best structure from optimized_structures to outstru.

        No return value.
        """
        if self.outstru is None:
            return
        from numpy import argmin
        idx = argmin(self.optimized_costs)
        stru = self.optimized_structures[idx]
        if self.outstru == '-':
            s = stru.writeStr(format=self.outfmt)
            sys.stdout.write(s)
        else:
            stru.write(self.outstru, format=self.outfmt)
        return


    def _processCommandLineArgs(self, argv):
        """Process command line arguments and assign input attributes.

        argv    -- command line arguments including the running program

        No return value.
        """
        import getopt
        self.mypath = argv[0]
        shortopts = self.script_options[0::2]
        longopts = self.script_options[1::2]
        optmap = dict([(s.rstrip(':'), l.rstrip('='))
            for s, l in zip(shortopts, longopts) if s])
        opts, args = getopt.gnu_getopt(argv[1:], "".join(shortopts), longopts)
        for o, a in opts:
            obare = o.lstrip('-')
            oname = optmap.get(obare, obare)
            pname = "_parseOption_" + oname
            option_parser = getattr(self, pname)
            option_parser(a)
        self._parseArguments(args)
        return


    def _loadDataFiles(self):
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

    # parsers for command line options and arguments

    def _parseOption_formula(self, a):
        self.formula = a
        return

    def _parseOption_latpar(self, a):
        latparstrings = a.strip().split(",")
        assert len(latparstrings) == 6
        self.latpar = [float(w) for w in latparstrings]
        return

    def _parseOption_outstru(self, a):
        self.outstru = a
        if self.outstru == "-":
            outlog.setLevel(logging.WARNING)
        return

    def _parseOption_outfmt(self, a):
        self.outfmt = a
        return

    def _parseOption_radia(self, a):
        words = a.strip().split(",")
        for w in words:
            elsmbl, value = w.split(":", 1)
            self.radia[elsmbl.strip()] = float(value)
        return

    def _parseOption_repeats(self, a):
        self.repeats = int(a)
        return

    def _parseOption_rotate(self, a):
        self.rotate = True
        return

    def _parseOption_rngseed(self, a):
        self.rngseed = int(a)
        return

    def _parseOption_debug(self, a):
        """No operation, debug is handled in main().
        """
        return

    def _parseOption_help(self, a):
        self.usage()
        sys.exit()

    def _parseOption_version(self, a):
        print self.script_version
        sys.exit()

    def _parseArguments(self, args):
        """Process any non-option command line arguments.

        args   -- list of command line arguments that are not options,
                  excluding the script name.

        No return value.
        """
        if not args:
            self.usage(brief=True)
            sys.exit()
        self.structfiles = args
        return

    # end of command line parsers

# End of class ColorFromOverlap

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
        cfaos = ColorFromOverlap(sys.argv)
        cfaos.run()
    except Exception, err:
        if "--debug" in sys.argv:
            raise
        print >> sys.stderr, err
        sys.exit(2)
    return


if __name__ == "__main__":
    main()

# End of file
