#!/usr/bin/env python

import park
import numpy

class BaseParameter(park.Parameter):
    """Parameter class with some utility functions."""

    def __str__(self):
        return "%s (%8.6f)"% (self.name, self.value)

class PDFTopLevelParameter(BaseParameter):
    """Custom class that communicates between top-level PDFModel parameters and
    the underlying pdffit2 engine.

    Parameters of this type include:

    "dscale"
    "qdamp"
    "qbroad"
    """

    def __init__(self, name, value):
        BaseParameter.__init__(self, name)
        self.status = "fixed"

        # The engines.
        self._engine = None

        # Name used by the engine for this parameter
        self._ename = name

        # We store the value in a temporary container until we have an engine
        self._localval = value
        return

    def _addEngine(self, engine, ename):
        """Set the engine and load the structure into it."""
        self._ename = ename
        self._engine = engine
        # Assign the variable in the engine
        engine.setvar(ename, self._localval)
        return

    def _getvalue(self):
        """Overrides _getvalue of park parameter."""
        return self._localval

    def _setvalue(self, val):
        """Overrides _setvalue of park parameter."""
        self._localval = val
        if self._engine:
            self._engine.setvar(self._ename, val)
        return

    value = property(_getvalue,_setvalue)

# End class PDFTopLevelParameter

class PDFPhaseParameter(BaseParameter):
    """Custom class that communicates between phase parameters and the
    underlying pdffit2 engine.

    pdffit2 Parameters of this type include:

    "pscale", etc.
    "lat1", "lat2", etc.
    "x(1)", "u11(1)", etc.
    """

    def __init__(self, name, value):
        BaseParameter.__init__(self, name)
        self.status = "fixed"
        
        # A dictionary of engines. The keys are PdfFit instances, and the values
        # are (ename, phasenum) tuples.
        self._edict = {}

        # We store the value in a temporary container until we have an engine
        self._localval = value

        return

    def _addEngine(self, engine, ename, phasenum):
        """Set the engine and load the structure into it."""
        if engine not in self._edict:
            #print ename, phasenum, engine, self._localval
            self._edict[engine] = (ename, phasenum)
        # Assign the variable in the engine
        engine.setphase(phasenum)
        engine.setvar(ename, self._localval)
        return

    def _getvalue(self):
        """Overrides _getvalue of park parameter."""
        return self._localval

    def _setvalue(self, val):
        """Overrides _setvalue of park parameter."""
        self._localval = val
        for engine, id in self._edict.items():
            #print engine, id, val
            engine.setphase(id[1])
            engine.setvar(id[0], val)
        return

    value = property(_getvalue,_setvalue)

# End class PDFPhaseParameter
