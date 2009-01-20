#!/usr/bin/env python
"""This module contains the Lattice class for holding lattice information.

See the class documentation for more information.
"""

__id__ = "$Id$"

import park
from parameters import PDFPhaseParameter 

class Lattice(park.ParameterSet):
    """Class for lattice information.

    Attributes
    spcgrp    --  The space group.

    Refinable parameters
    a, b, c     --  Lattice parameters (default 1.0)
    alpha, beta, gamma  --  Lattice angles (default 90.0)
    """

    def __init__(self):
        """Initialize the attributes.

        The name of a lattice is always "lattice".
        """
        park.ParameterSet.__init__(self, "lattice")
        self.append(PDFPhaseParameter("a",1.0))
        self.append(PDFPhaseParameter("b",1.0))
        self.append(PDFPhaseParameter("c",1.0))
        self.append(PDFPhaseParameter("alpha",90.0))
        self.append(PDFPhaseParameter("beta",90.0))
        self.append(PDFPhaseParameter("gamma",90.0))

        return

    def _addEngine(self, engine, phasenum):
        """Add an engine for these parameters."""
        from diffpy.pdffit2 import PdfFit
        # Set the engine name for each parameter
        for par in self:
            pname = PdfFit.lat(par.name)
            par._addEngine(engine, pname, phasenum)
        return

    def __str__(self):
        """String representation."""
        s = "lattice (%6.4f %6.4f %6.4f %6.4f %6.4f %6.4f)" % \
            (self["a"].get(), self["b"].get(), self["c"].get(),
             self["alpha"].get(), self["beta"].get(), self["gamma"].get())
        return s

# End of Lattice

if __name__ == "__main__":
    # Check to see if everything imports correctly
    pass
