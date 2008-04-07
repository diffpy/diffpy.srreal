#!/usr/bin/env python
"""This module contains classes for holding structural phase information for
calculating a diffraction pattern.

See the class documentation for more information.
"""
__id__ = "$Id$"

from RefinementAPI.Model import Model
from RefinementAPI.Refinable import Refinable
from RefinementAPI.exceptiontools import verifyType, isFloatable

class Phase(Model, Refinable):
    """Abstract class for holding structure information.

    Attributes
    _comp       --  The parent RietveldComponent instance.

    Common Exceptions
    TypeError is thrown when method arguments are of inappropriate type.
    ValueError when method argument does not fall within specified bounds.
    """

    def __init__(self, name = ""):
        """Initialize the attributes.
        
        Arguments:
        name    --  The name of this phase (string, default "").
        """
        Model.__init__(self)
        Refinable.__init__(self)
        self.setName(name)
        self._comp = None
        return

# End of Phase

class CrystalPhase(Phase):
    """Class for holding crystal structure information.

    This class has refinable variables. Its subclasses must also inherit from a
    derivative of the Refinable class.

    Attributes
    _atoms      --  List of Atom instances (default []).
    _lattice    --  Lattice instance (default Lattice(1,1,1,90,90,90)).
    _shift      --  Shift in the origin (3-list, default [0.0, 0.0, 0.0])
    _spcgrp     --  The Hermann-Mauguin space group symbol of the structure
                    (string, default "P1").

    Refinable Parameters
    weight      --  The weight of the phase in a multi-phase structure (float in
                    [0,1], default 1). This is automaically set to 1 if the
                    phase is not part of the multi-phase structure.
    delta1      --  Linear correlation term in DW factor (low temperature)
                    (default 0).
    delta2      --  Quadratic correlation term in DW factor (high temperature)
                    (default 0).
    sratio      --  Low-r peak sharpening (default 0).
    rcut        --  Peak sharpening cutoff (default 0).

    Common Exceptions
    TypeError is thrown when method arguments are of inappropriate type.
    ValueError when method argument does not fall within specified bounds.
    """

    def __init__(self, name = ""):
        """Initialize the attributes.
        
        Arguments:
        name    --  The name of this phase (string, default "").
        """
        Phase.__init__(self)
        self._atoms = []
        self._lattice = None
        self._shift = [0.0, 0.0, 0.0]
        self._spcgrp = "P1"
        self.addParameter("weight", 1.0)
        self.addParameter("delta1", 1.0)
        self.addParameter("delta2", 1.0)
        self.addParameter("sratio", 1.0)
        self.addParameter("rcut", 1.0)

    def setSpaceGroup(self, spcgrp):
        """Set the space group.

        This forces the lattice parameters to conform to the space group
        symmetry. Equality between lattice parameters is enforced with hard
        constraints. The protocol this follows is listed under Crystal Systems
        below.

        Note that calling this method destroys any existing constraints on the
        lattice, including those made to the Fit using mapVP.

        Arguments
        spcgrp  --  The Hermann-Mauguin space group symbol

        Crystal Systems:
        Triclinic       --  No constraints.
        Monoclinic      --  alpha and beta are constrained to 90 unless alpha !=
                            beta and alpha == gamma, in which case alpha and
                            gamma are constrained to 90.
        Orthorhombic    --  alpha, beta and gamma are constrained to 90
        Tetragonal      --  b is constrained to a and alpha, beta and gamma are
                            constrained to 90.
        Trigonal        --  If gamma == 120, then b is constrained to a, alpha
                            and beta are constrained to 90 and gamma is
                            constrained to 120.  Otherwise, b and c are
                            constrained to a, beta and gamma are constrained to
                            alpha.
        Hexagonal       --  b is constrained to a, alpha and beta are
                            constrained to 90 and gamma is constrained to 120.
        Cubic           --  b and c are constrained to a, and alpha, beta and
                            gamma are constrained to 90.


        Raises 
        AttributeError if the lattice has not been set.
        ValueError if the space group symbol is unrecognized.
        """
        verifyType(spcgrp, str)

        # Verify that the space group is known
        import diffpy.Structure.SpaceGroups as SpaceGroups
        sg = SpaceGroups.GetSpaceGroup(spcgrp)
        if spcgrp != sg.short_name:
            raise ValueError("Space group '%s' is not recognized"%spcgrp)
        self._spcgrp = spcgrp

        if self._lattice is None: return

        lat = self._lattice
        # Freeze everything. Note that this kills any existing constraints to
        # other VariableOrganizers, including that in a Fit.
        lat._V.freeze("a", lat.a)
        lat._V.freeze("b", lat.b)
        lat._V.freeze("c", lat.c)
        lat._V.freeze("alpha", lat.alpha)
        lat._V.freeze("beta", lat.beta)
        lat._V.freeze("gamma", lat.gamma)

        # Create constraints that are consistent with symmetry.
        # ref: Benjamin, W. A., Introduction to crystallography, New York
        # (1969), p.60
        constraintMap = {
          "TRICLINIC"  : self._constrainTriclinic,
          "MONOCLINIC" : self._constrainMonoclinic,
          "ORTHORHOMBIC" : self._constrainOrthorhombic, 
          "TETRAGONAL" : self._constrainTetragonal,
          "TRIGONAL"   : self._constrainTrigonal,
          "HEXAGONAL"  : self._constrainHexagonal,
          "CUBIC"      : self._constrainCubic
        }

        constraintMap[sg.crystal_system]()
        return

    def _constrainTriclinic(self):
        """Make constraints for Triclinic systems.

        No constraints are made.
        """
        return

    def _constrainMonoclinic(self):
        """Make constraints for Monoclinic systems.
        
        alpha and beta are constrained to 90 unless alpha != beta and alpha ==
        gamma, in which case alpha and gamma are constrained to 90.
        """
        lat = self._lattice
        lat._V.constrain("alpha", "__90")
        if lat.alpha != lat.beta and lat.alpha == lat.gamma:
            lat._V.constrain("gamma", "__90")
        else:
            lat._V.constrain("beta", "__90")
        return

    def _constrainOrthorhombic(self):
        """Make constraints for Orthorhombic systems.
        
        alpha, beta and gamma are constrained to 90
        """
        lat = self._lattice
        lat._V.constrain("alpha", "__90")
        lat._V.constrain("beta", "__90")
        lat._V.constrain("gamma", "__90")
        return

    def _constrainTetragonal(self):
        """Make constraints for Tetragonal systems.

        b is constrained to a and alpha, beta and gamma are constrained to 90.
        """
        lat = self._lattice
        lat._V.constrain("b", "a")
        lat._V.constrain("alpha", "__90")
        lat._V.constrain("beta", "__90")
        lat._V.constrain("gamma", "__90")
        return

    def _constrainTrigonal(self):
        """Make constraints for Trigonal systems.

        If gamma == 120, then b is constrained to a, alpha and beta are
        constrained to 90 and gamma is constrained to 120. Otherwise, b and c
        are constrained to a, beta and gamma are constrained to alpha.
        """
        lat = self._lattice
        if lat.gamma == 120:
            lat._V.constrain("b", "a")
            lat._V.constrain("alpha", "__90")
            lat._V.constrain("beta", "__90")
            lat._V.constrain("gamma", "__120")
        else:
            lat._V.constrain("b", "a")
            lat._V.constrain("c", "a")
            lat._V.constrain("beta", "alpha")
            lat._V.constrain("gamma", "alpha")
        return

    def _constrainHexagonal(self):
        """Make constraints for Hexagonal systems.

        b is constrained to a, alpha and beta are constrained to 90 and gamma is
        constrained to 120.
        """
        lat = self._lattice
        lat._V.constrain("b", "a")
        lat._V.constrain("alpha", "__90")
        lat._V.constrain("beta", "__90")
        lat._V.constrain("gamma", "__120")
        return

    def _constrainCubic(self):
        """Make constraints for Cubic systems.

        b and c are constrained to a, alpha, beta and gamma are constrained to
        90.
        """
        lat = self._lattice
        lat._V.constrain("b", "a")
        lat._V.constrain("c", "a")
        lat._V.constrain("alpha", "__90")
        lat._V.constrain("beta", "__90")
        lat._V.constrain("gamma", "__90")
        return

    def getSpaceGroup(self):
        """Get the space group symbol."""
        return self._spcgrp

    def setShift(self, shift):
        """Set the origin shift.

        Arguments
        shift   --  The (x,y,z) shift in the origin (3-tuple of floats)
        """
        msg = "Shift must be a 3-tuple"
        if not isIterable(shift):
            raise TypeError(msg)
        if len(shift) != 3:
            raise ValueError(msg)
        for x in shift:
            if not isFloatable(x):
                raise TypeError("Shift values must be floats")

        self._shift = map(float, shift)
        return
    
    def getShift(self):
        """Get the origin shift."""
        return self._shift

    def setLattice(self, lat):
        """Set the lattice for the phase.

        This automatically constrains the lattice parameters according to the
        space group. See setSpaceGroup.

        Arguments
        lat     --  Lattice instance.
        """
        from Lattice import Lattice
        verifyType(lat, Lattice)
        self._lattice = lat
        # Do the space-group constraints
        self.setSpaceGroup(self._spcgrp)
        return

    def getLattice(self):
        """Get the lattice for the phase. """
        return self._lattice

    def getNumAtoms(self):
        """Get the number of atoms.

        Returns the number of atoms in the phase (non-negative integer).
        """
        return len(self._atoms)

    def addAtom(self, a):
        """Add an atom to the phase.

        Arguments
        a       --  Atom instance.
        """
        from Atom import Atom
        verifyType(a, Atom)
        self._atoms.append(a)
        return

    def getAtom(self, index):
        """Get an atom from the phase.
        
        This uses python list notation, so index -n returns the nth atom from
        the end.

        Arguments:
        index  --  index of atom (integer, starting at 0).

        Raises 
        IndexError if requesting an atom that does not exist
        """
        if index > len(self._atoms):
            raise IndexError("Atom index out of range")
        if -index > len(self._atoms):
            raise IndexError("Atom index out of range")
        return self._atoms[index]

    def getSubRefinables(self):
        """Get a list of subordinate refinables."""
        refs = [self._lattice]
        refs.extend(self._atoms)
        return refs
       
# End of CrystalPhase

class _MultiPhase(Model):
    """Class for holding multiple phases.

    Attributes
    _phases --  List of stored Phase instances (default []).

    Common Exceptions
    TypeError is thrown when method arguments are of inappropriate type.
    ValueError when method argument does not fall within specified bounds.
    """

    def __init__(self):
        """Initialize the attributes.
        
        Arguments:
        name    --  The name of the multi-phase (string, default "").
        """
        Model.__init__(self)
        self._phases = []
        self._comp = None

    def addPhase(self, phase):
        """Add a phase.

        Arguments
        phase   --  A Phase instance.

        Raises
        ValueError when attempting to add a phase that has already been added.
        """
        verifyType(phase, Phase)
        if phase in self._phases:
            raise ValueError("Phase has already been added")
        self._phases.append(phase)
        phase._V.freeze("weight", 1.0)
        phase._comp = self._comp
        return

    def getPhase(self, index):
        """Get a phase.
        
        This uses python list notation, so index -n returns the nth phase from
        the end.

        Arguments:
        index  --  index of phase (integer, starting at 0).

        Returns Phase instance

        Raises 
        IndexError if requesting a phase that does not exist
        """
        if index > len(self._phases):
            raise IndexError("Phase index out of range")
        if -index > len(self._phases):
            raise IndexError("Phase index out of range")
        return self._phases[index]

    def getNumPhases(self):
        """Get the number of phases."""
        return len(self._phases)

    def getSubRefinables(self):
        """Get a list of subordinate refinables."""
        return self._phases

# End of _MultiPhase

if __name__ == "__main__":
    # Check to see if everything imports correctly
    pass

