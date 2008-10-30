#!/usr/bin/env python

"""SCRIPTNAME   arrange atoms for minimum overlap, compare PDF Rw.
Usage: SCRIPTNAME [options] grfile structfile1 [structfile2]...

Assign atom species over sites to minimize the overlap of their radii.
The final cost of such colored structure is obtained from PDF fitting
residuum.  This allows to compare several structures that can be colored
to zero overlap.

Options:

  -f, --formula=FORMULA     chemical formula of the unit cell, must be
                            commensurate with number of atom sites.  When
                            not specified, use composition from structfile1.
  -l, --latpar=a,b,c,A,B,G  override lattice parameters in structfiles.
  -o, --outstru=FILE        Filename for saving the best fitting structure.
      --outfmt=FORMAT       Output format for outstru, by default "discus"
                            Can be any format supported by diffpy.Structure.
      --rmin=FLOAT          Lower boundary for PDF fitting.
      --rmax=FLOAT          Upper boundary for PDF fitting.
      --repeats=N           Number of downhill optimizations from initial
                            random configurations.  By default 5.
      --rotate              Scan all unique rotations for each of the repeats 
                            initial configurations.  Might be slow.
      --rngseed=N           Specify integer seed for random number generator.
  -v, --verbose             output coloring progress
      --debug               Enter python debugger after catching an exception.
  -h, --help                display this message
  -V, --version             show script version
"""

__id__ = "$Id$"

import sys

# global variables
gl_doc = __doc__


from colorFromAtomOverlap import ColorFromAtomOverlapScript

class ColorFromOverlapCmpPDF(ColorFromAtomOverlapScript):
    """Class for running overlap optimization script.
    """

    # script documentation string
    script_doc = gl_doc
    script_version = __id__
    script_options = ColorFromAtomOverlapScript.script_options + [
                            "", "rmin=",
                            "", "rmax=",
                            "v", "verbose",
                        ]
    Uiso_default = 0.0001

    def __init__(self, argv):
        """Create the script instance and prepare it for running
        
        argv    -- command line arguments

        No return value.
        """
        # input parameters in addition to those in super class
        self.grfile = None
        self.rmin = None
        self.rmax = None
        self.stype = "X"
        self.qmax = None
        self.verbose = False
        self.__pdffit = None
        self.__cost_cache = {}
        # initialize from the base class
        self.super = super(ColorFromOverlapCmpPDF, self)
        self.super.__init__(argv)
        return


    # overloaded methods:


    def optimizeColoring(self, stru):
        """Perform repeated downhill minimizations of atom overlap in stru.

        stru    -- instance of initial Structure

        Return a tuple of (cost, colored_structure).
        """
        # perform atom overlap optimization
        cost, stru = self.super.optimizeColoring(stru)
        fittedcost, fittedstru = self._doPdfFit(stru)
        return (fittedcost, fittedstru)


    def _loadDataFiles(self):
        """Load all input structfiles and assign the structures attribute.
        Also load the grfile and assign qmax.

        No return value.
        Raises ValueError when qmax cannot be extracted from self.grfile
        """
        self.super._loadDataFiles()
        # extract stype and qmax
        import re
        grdata = open(self.grfile).read()
        # stype
        if re.search('(x-?ray|PDFgetX)', grdata, re.I):
            self.stype = 'X'
        elif re.search('(neutron|PDFgetN)', grdata, re.I):
            self.stype = 'N'
        # qmax
        rxpat = r'\bqmax *= *([-+]?(\d+(\.\d*)?|\.\d+)([eE][-+]?\d+)?)'
        mx = re.search(rxpat, grdata, re.I)
        if mx:
            self.qmax = float(mx.group(1))
        else:
            emsg = "Cannot find qmax in %s" % self.grfile
            raise ValueError, emsg
        # resolve rmin, rmax
        pf = self._getPdfFit()
        pf.read_data(self.grfile, self.stype, self.qmax, 0.0)
        r = pf.getR()
        if self.rmin is None:
            self.rmin = r[0]
        if self.rmax is None:
            self.rmax = r[-1]
        return


    # parsers for command line options and arguments

    def _parseOption_rmin(self, a):
        self.rmin = float(a)
        return

    def _parseOption_rmax(self, a):
        self.rmax = float(a)
        return

    def _parseOption_verbose(self, a):
        self.verbose = True
        return

    def _parseArguments(self, args):
        """Process any non-option command line arguments.

        args   -- list of command line arguments that are not options,
                  excluding the script name.

        No return value.
        """
        if len(args) < 2:
            self.usage(brief=True)
            sys.exit()
        self.grfile = args[0]
        self.structfiles = args[1:]
        return

    # end of command line parsers

    # additional methods

    def _doPdfFit(self, stru):
        """Perform PDF fitting and return Rw and the fitted structure.

        stru -- instance of diffpy.Structure

        Return a tuple of (rw, fitted_structure).
        """
        pf = self._getPdfFit()
        pf.reset()
        pf.add_structure(stru)
        pf.read_data(self.grfile, self.stype, self.qmax, 0.0)
        # set active fitting range
        pf.pdfrange(1, self.rmin, self.rmax)
        # allpars is a map of variable names to parameter index
        allpars = {}
        def quick_constrain(varname, pidx=None, pvalue=None):
            if pidx is None:
                pidx = allpars.setdefault(varname, len(allpars) + 1)
            allpars[varname] = pidx
            pf.constrain(varname, pidx)
            if pvalue is None:
                pvalue = pf.getvar(varname)
            pf.setpar(pidx, pvalue)
            return pidx
        # data scale
        quick_constrain("dscale")
        # qdamp
        quick_constrain("qdamp", pvalue=1.0/self.rmax)
        # structure constraints:
        unique_cell_length = {}
        for vn in ('lat(1)', 'lat(2)', 'lat(3)'):
            abc = pf.getvar(vn)
            uvn = unique_cell_length.setdefault(abc, vn)
            pidx = allpars.get(uvn, None)
            quick_constrain(vn, pidx)
        # isotropic constraints for thermal parameters
        el2par = {}
        # pdffit uses indices starting at 1
        idx1 = 0
        coloring = [a.element for a in stru]
        for elsmbl in coloring:
            idx1 += 1
            varname1 = "u11(%i)" % idx1
            varname2 = "u22(%i)" % idx1
            varname3 = "u33(%i)" % idx1
            pidx = el2par.get(elsmbl, None)
            uiso_initial = pf.getvar(varname1) or self.Uiso_default
            # when pidx is None, assign new unique index
            pidx = quick_constrain(varname1, pidx, uiso_initial)
            quick_constrain(varname2, pidx, uiso_initial)
            quick_constrain(varname3, pidx, uiso_initial)
            el2par[elsmbl] = pidx
        # finally constrain delta2
        quick_constrain('delta2')
        # perform 2 stage refinement to improve stability of qdamp
        from diffpy.pdffit2.pdffit2 import calculationError
        try:
            pf.fixpar('ALL')
            pf.freepar(allpars['dscale'])
            pf.freepar(allpars['qdamp'])
            pf.freepar(allpars['lat(1)'])
            pf.freepar(allpars['lat(2)'])
            pf.freepar(allpars['lat(3)'])
            pf.refine()
            pf.freepar('ALL')
            pf.fixpar(allpars['qdamp'])
            pf.refine()
            pf.freepar('ALL')
            pf.refine()
            rw = pf.getrw()
        except calculationError:
            rw = float('Inf')
        if self.debug:
            print sorted(allpars.items(), key=lambda x: x[::-1])
            print pf.save_res_string()
        fitted_structure = pf.get_structure(1)
        return rw, fitted_structure


    def _getPdfFit(self):
        """Instance of PdfFit owned by this object.

        Return reference to PdfFit.
        """
        if self.__pdffit is None:
            import diffpy.pdffit2
            sink = open('/dev/null', 'w')
            diffpy.pdffit2.redirect_stdout(sink)
            self.__pdffit = diffpy.pdffit2.PdfFit()
        return self.__pdffit


# End of class ColorFromPDF


def main():
    try:
        script = ColorFromOverlapCmpPDF(sys.argv)
        script.run()
    except Exception, err:
        if "--debug" in sys.argv:
            raise
        print >> sys.stderr, err
        sys.exit(2)

if __name__ == "__main__":
    main()
