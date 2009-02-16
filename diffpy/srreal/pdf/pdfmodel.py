#!/usr/bin/env python
"""The PDFModel class."""

__id__ = "$Id$"

import numpy
import park
from diffpy.srreal.pdf.dynamicmodel import DynamicModel
from diffpy.srreal.pdf.parameters import PDFTopLevelParameter

class PDFModel(DynamicModel):
    """park.Model for PDF refinement"""

    def __init__(self, name="", **kw):
        """initialize"""
        DynamicModel.__init__(self, name, **kw)
        self.stype = "X"
        self.qmax = 0
        self.parameterset = park.ParameterSet(name=name)

        # The engine
        self._engine = None

        # phases
        self._phases = []

        # Flag indicating whether we need to interpolate
        self._interp = False

        # The x-values from the engine needed for interpolation
        self._ex = []

        # Add the top-level parameters
        self.parameterset.append(PDFTopLevelParameter("dscale", 1.0))
        self.parameterset.append(PDFTopLevelParameter("qdamp", 0.0))
        self.parameterset.append(PDFTopLevelParameter("qbroad", 0.0))

        return

    def eval(self, x):
        """Evaluate the PDF function."""
        x = numpy.asarray(x)
        p = self._getEngine(x)
        #print "starting calculation"
        p.calc()
        y = numpy.asarray(p.getpdf_fit())
        #print "ending calculation"
        # The engine works on a fixed-space grid. We must see if this is the
        # same as the requested grid. If not, then we interpolate on the
        # requesteed grid.
        if self._interp:
            # FIXME - the engine can't handle this right now. There seems to be
            # a bug in getR.
            raise NotImplementedError("Can't interpolate")
        #print self._engine
        return y

    def addPhase(self, phase):
        """Add a phase to the model.

        Do not add atoms to a phase after it has been given to the model, as
        there is no way to add more atoms to the engine once a structure has
        been loaded.
        
        Arguments
        phase   --  phase to add
        """
        self._phases.append(phase)
        return

    def getNumPhases(self):
        """Get a phase count."""
        return len(self._phases)

    def getPhases(self):
        """Get the phases."""
        return self._phases[:]

    ## Internals. Users ignore this stuff. ##

    def _getEngine(self, x):
        """Get a stored engine by key.

        If the requested engine does not exist, this will create a new engine
        and inform all parameters tied to this model about that engine.
        """
        if not numpy.array_equal(x, self._ex):
            self._setupEngine(x)
        return self._engine

    def _setupEngine(self, x):
        """Set up the engine for the calculation."""

        ## Create the engine
        self._engine = _createEngine()
        self._ex = numpy.array(x)

        for i, phase in enumerate(self._phases):
            # If the phase was loaded from a structure, this will save some
            # time when the phase is added to the engine.
            if phase._stru is None:
                from diffpy.srreal.pdf import converters
                phase._stru = converters.structureFromPhase(phase)
            self._engine.add_structure(phase._stru)
            phase._addEngine(self._engine, i+1)

        # allocate room in the engine for the calculation
        stype = self.stype
        qmax = self.qmax
        rmin = x[0]
        rmax = x[-1]
        bin = len(x)
        self._engine.alloc(stype, qmax, 0, rmin, rmax, bin)

        # We have to interpolate if the requested calculation points are not
        # equidistant.
        dx = x[1] - x[0]
        toler = 1e-6
        for i in range(2, len(x)):
            if abs(dx - (x[i]-x[i-1])) > toler:
                self._interp = True
                break;

        ## Link the other parameters to this engine.
        for pname in ["dscale", "qdamp", "qbroad"]:
            self.parameterset[pname]._addEngine(self._engine, pname)

        return

# End class PDFModel

def _createEngine():
    """Create the pdffit2 engine instance."""

    from diffpy.pdffit2 import PdfFit
    from diffpy.pdffit2 import redirect_stdout

    class CustomPdfFit(PdfFit):
        """A PdfFit class that works well as a function calculator.
        
        This class suppresses output.
        """

        def __init__(self):
            import os
            redirect_stdout(os.tmpfile())
            PdfFit.__init__(self)

    p = CustomPdfFit()
    return p
