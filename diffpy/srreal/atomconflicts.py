"""class PairHistogram -- weighed pair distance counts from a structure model.
"""

# module version
__id__ = "$Id$"


##############################################################################
class AtomConflicts(object):
    """Evaluate number of spatial atom conflicts in a crystal structure.
    """


    def __init__(self, stru, atomradiagetter=None):
        """Evaluate atom conflicts for a given crystal structure.

        stru    -- structure model, instance of class Structure
                   from diffpy.Structure.
        atomradiagetter -- optional reference to a function which
                   returns radius of specified element.  By default
                   this recovers covalent radia from Paul Kienzle
                   elements library.  See setAtomRadiaGetter for
                   further details.
        """
        # create data items
        # inputs
        self._structure = None
        # reference to a function that returns atom radia
        self._atomradiagetter = None
        # outputs
        self._conflicts = []
        self._conflicts_cached = False
        # generated data
        # store coloring in separate list to allow shallow copy of structure
        self._site_coloring = []
        self._pair_lengths = []
        self._pair_lengths_cached = False
        # assign argumets
        self.setStructure(stru)
        # use default routine when argument was not specified
        if atomradiagetter is None:
            self.setAtomRadiaGetter(getCovalentRadius)
        else:
            self.setAtomRadiaGetter(atomradiagetter)
        return


    # public methods


    def getConflicts(self):
        """List of all conflicting atom pairs in a crystal structure.

        Return a list of tuples (i, j, dij, ddij), where i, j are atom
        indices, dij their distance and ddij=(dmin - dij).  Distances
        dij are calculated with respect to periodic boundary conditions.
        """
        self._update_conflicts()
        return self._conflicts[:]


    def countConflicts(self):
        """Number of conflicting pairs in a unit cell.

        Return int.
        """
        self._update_conflicts()
        cnt = len(self._conflicts)
        return cnt


    def countAtoms(self):
        """Number of atoms in the internal structure model.

        Return int.
        """
        return len(self._structure)


    def setStructure(self, stru):
        """Specify crystal structure for evaluating atom conflicts.

        stru -- structure model, instance of Structure from diffpy.Structure.
                Makes a copy of stru for internal storage.

        No return value.
        """
        from diffpy.Structure import Structure
        self._uncache('conflicts', 'pair_lengths')
        self._structure = Structure(stru)
        self._site_coloring = [a.element for a in self._structure]
        return


    def getStructure(self):
        """Structure model for which atom conflicts are calculated.

        Return new instance of Structure, a copy of internal model.
        """
        from diffpy.Structure import Structure
        stru = Structure(self._structure)
        for a, smbl in zip(stru, self._site_coloring):
            a.element = smbl
        return stru


    def setSiteColoring(self, coloring):
        """Assign new atom types per each site in structure model.

        coloring -- list of string symbols for elements or isotopes.
                    The length of coloring must be equal to countAtoms().

        No return value.
        Raise ValueError for invalid argument.
        """
        if len(coloring) != self.countAtoms():
            emsg = "Invalid length of element list."
            raise ValueError, emsg
        # convert argument to a list
        newcoloring = list(coloring)
        # short circuit
        if self._site_coloring == newcoloring:
            return
        self._uncache('conflicts', 'pair_lengths')
        # assign new coloring:
        self._site_coloring = newcoloring
        return


    def getSiteColoring(self):
        """List of atom types per each site in structure model.

        Return list of string symbols for elements or isotopes.
        """
        rv = self._site_coloring[:]
        return rv


    def flipSiteColoring(self, i, j):
        """Exchange atom types at sites i and j.

        i   -- zero based index of the first site
        j   -- zero based index of the second site

        No return value.
        Raise IndexError for invalid arguments.
        """
        # negative indices are valid in python
        if i < 0:   i += self.countAtoms()
        if j < 0:   j += self.countAtoms()
        smbi = self._site_coloring[i]
        smbj = self._site_coloring[j]
        if smbi == smbj: return
        # _site_coloring changes so we need to create a new copy
        self._site_coloring = self._site_coloring[:]
        self._site_coloring[i] = smbj
        self._site_coloring[j] = smbi
        self._uncache('conflicts')
        return


    def setAtomRadiaGetter(self, atomradiagetter):
        """Specify funtion for looking up atom radia.

        atomradiagetter -- reference to a function of one argument returning
                           atom radius.  A dictionary can be also used by
                           passing its bound get method, for example
                           atomradiagetter={Na" : 1.85}.get.

        No return value.
        """
        self._uncache('conflicts', 'pair_lengths')
        self._atomradiagetter = atomradiagetter
        return


    def getRmax(self):
        """Radius of the largest atom in the structure or zero
        for empty structure.

        Return float.
        """
        allradia = [self.atomRadius(smbl) for smbl in self._site_coloring]
        rmax = max([0.0] + allradia)
        return rmax


    def atomRadius(self, elsmbl):
        """Radius of a specified atom type.

        elsmbl  -- string symbol for element, isotope or ion.
                   Must be a valid argument to atom radia getter.

        See also setAtomRadiaGetter.

        Return float.
        """
        rv = float(self._atomradiagetter(elsmbl))
        return rv


    # protected methods


    def _update_conflicts(self):
        """Recalculate atom conflicts.  Update data in _conflicts.
        Do nothing when _conflicts_cached flag is True.

        No return value.
        """
        if self._conflicts_cached:
            return
        self._update_pair_lengths()
        cnfls = []
        for i, j, dij in self._pair_lengths:
            smbi = self._site_coloring[i]
            smbj = self._site_coloring[j]
            dminij = self.atomRadius(smbi) + self.atomRadius(smbj)
            ddij = dminij - dij
            if ddij > 0:
                cnfls.append((i, j, dij, ddij))
        self._conflicts = cnfls
        self._conflicts_cached = True
        return


    def _update_pair_lengths(self):
        """Recalculate pair lengths in the crystal structure.  Update data
        in _pair_lengths.  Do nothing when _pair_lengths_cached flag is True.

        No return value.
        """
        if self._pair_lengths_cached:
            return
        from diffpy.srreal.pairhistogram import PairHistogram
        pf = PairHistogram._getPdfFit()
        # Because, this method only evaluates geometry, let us use
        # all carbon atoms so there are no problems with unknown types.
        carbonstru = self.getStructure()
        for a in carbonstru:
            a.element = "C"
        pf.add_structure(carbonstru)
        dmin = 1.0e-8
        dmax = 2 * self.getRmax()
        bldict = pf.bond_length_types('ALL', 'ALL', dmin, dmax)
        self._pair_lengths = [(i, j, dij)
                for (i, j), dij in zip(bldict['ij0'], bldict['dij'])]
        self._pair_lengths_cached = True
        return


    def _uncache(self, *args):
        """Reset cached flag for a list of internal attributes.

        *args -- list of strings, currently supported are "conflicts",
                 "pair_lengths".

        No return value.
        Raise AttributeError for any invalid args.
        """
        for a in args:
            attrname = "_" + a + "_cached"
            setattr(self, attrname, False)
        return


# End of class AtomConflicts


########################################################################
# Routines
########################################################################


__elements_periodic_table = None

def getCovalentRadius(elsmbl):
    """Return covalent radius.  Adaptor to Paul Kienzle elements package.
    
    elsmbl  -- string symbol for element or isotope.

    Return covalent radius in Angstroms.
    """
    # this may be called many times, therefore we avoid repeated imports
    # and cache a reference to the periodic table
    global __elements_periodic_table
    if __elements_periodic_table is None:
        import elements
        __elements_periodic_table = elements.periodic_table
    e = getattr(__elements_periodic_table, elsmbl)
    return e.covalent_radius


# End of routines
