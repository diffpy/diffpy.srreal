#!/usr/bin/env python
"""This module contains classes for containing atomic information.

See the class documentation for more information.
"""

__id__ = "$Id$"

import park
from parameters import PDFPhaseParameter

class Atom(park.ParameterSet):
    """Class for atomic information.

    Attributes
    element    --  An element symbol (string, default "C").


    Refinable parameters
    x, y, z     --  Fractional coordinates (float, unitless, default 0.0)
    occ         --  Occupation of the lattice site (float, unitless, default
                    1.0)
    U11, U22, U33   --  Diagonal displacment factors (default 0.01).
    U12, U23, U31   --  Off-diagonal displacment factors (default 0.0).
    """

    def __init__(self, name):
        """Initialize the attributes.

        Arguments:
        name    --  The name of this phase (string, default "").
        """
        park.ParameterSet.__init__(self, name)
        self.element = "C"
        self.append(PDFPhaseParameter("x",0.0))
        self.append(PDFPhaseParameter("y",0.0))
        self.append(PDFPhaseParameter("z",0.0))
        self.append(PDFPhaseParameter("occ",1.0))
        self.append(PDFPhaseParameter("u11",0.003))
        self.append(PDFPhaseParameter("u22",0.003))
        self.append(PDFPhaseParameter("u33",0.003))
        self.append(PDFPhaseParameter("u12",0.0))
        self.append(PDFPhaseParameter("u23",0.0))
        self.append(PDFPhaseParameter("u13",0.0))
        return

    def __str__(self):
        """String representation."""
        s = "%-2s (%8.6f %8.6f %8.6f %6.4f)" % \
            (self.element, self["x"].get(), 
             self["y"].get(), self["z"].get(), self["occ"].get())
        return s

    def _addEngine(self, engine, phasenum, atomnum):
        """Add an engine for these parameters."""
        from diffpy.pdffit2 import PdfFit
        for par in self:
            f = getattr(PdfFit, par.name)
            par._addEngine(engine, f(atomnum), phasenum)
        return

if __name__ == "__main__":
    # Check to see if everything imports correctly
    pass
