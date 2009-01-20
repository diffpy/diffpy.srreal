#!/usr/bin/env python
"""This module contains classes for holding structural phase information for
calculating a diffraction pattern.

See the class documentation for more information.
"""
__id__ = "$Id$"

import park
from parameters import PDFPhaseParameter
from dynamicmodel import DynamicModel

class CrystalPhase(DynamicModel):
    """Class for holding crystal structure information.

    Attributes

    Refinable parameters
    pscale     --  The phase weight (default 1)
    delta1      --  The linear atomic correlation factor (default 0)
    delta2      --  The quadratic atomic correlation factor (default 0)
    sratio      --  Sigma ratio (default 1)
    rcut        --  Cutoff for sigma ratio (default 0)
    
    ParameterSets
    addLattice and addAtom add managed parameter sets to this phase.
    """

    def __init__(self, name = "", **kw):
        """Initialize the attributes.
        
        Arguments:
        name    --  The name of this phase (string, default "").
        """
        DynamicModel.__init__(self, name, **kw)

        self.parameterset = park.ParameterSet(name=name)
        self.parameterset.append(PDFPhaseParameter("pscale", 1.0))
        self.parameterset.append(PDFPhaseParameter("delta1", 0.0))
        self.parameterset.append(PDFPhaseParameter("delta2", 0.0))
        self.parameterset.append(PDFPhaseParameter("sratio", 1.0))
        self.parameterset.append(PDFPhaseParameter("rcut", 0.0))

        self._atoms = []

        # Handle to an equivalent diffpy.Structure object
        self._stru = None
        return

    def eval(self, x):
        return 0.0

    def addLattice(self, _l):
        """Add a configured lattice to the phase."""
        self.parameterset.append(_l)
        return

    def getLattice(self):
        """Get the lattice."""
        return self.parameterset["lattice"]

    def addAtom(self, _a):
        """Add a configured atom to the phase."""
        self.parameterset.append(_a)
        self._atoms.append(_a)
        return

    def getAtoms(self):
        """Get all the atoms."""
        return self._atoms[:]

    def getAtom(self, name):
        """Get an atom by name."""
        return self.parameterset[name]

    def _addEngine(self, engine, phasenum):
        """Set the engine and load the structure into it."""

        # Set up the atoms
        for atomnum, par in enumerate(self._atoms):
            par._addEngine(engine, phasenum, 1+atomnum)

        # Set up the lattice
        self.parameterset["lattice"]._addEngine(engine, phasenum)

        # Set up the others
        for pname in ["pscale", "delta1", "delta2", "sratio", "rcut"]:
            self.parameterset[pname]._addEngine(engine, pname, phasenum)

        return

if __name__ == "__main__":
    # Check to see if everything imports correctly
    pass

