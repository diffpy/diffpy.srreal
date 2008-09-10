"""class PairHistogram -- weighed pair distance counts from a structure model.
"""

# module version
__id__ = "$Id$"


import bisect


class PairHistogram(object):
    """Pair distance counts weighed by scattering factors.

    Class data:

    resolution -- cutoff for distinguishing 2 close pair distances.
                  Default value 0.01.
    __pdffit   -- shared instance of PdfFit from diffpy.pdffit2
    """

    # class data
    resolution = 0.01
    __pdffit = None


    def __init__(self, stru, rmax, pbc=True, resolution=None):
        """Create pair histogram for a given structure:

        stru  -- structure model, instance of class Structure
                 from diffpy.Structure.
        rmax  -- upper boundary for calculated pair distances.
        pbc   -- boolean flag for periodic boundary conditions.
                 May be removed later, when pbc would be identified
                 by the stru argument.
        resolution -- cutoff for distinguishing 2 nearby distances.
                      Instance override of default class value.
        """
        # create data items
        # inputs
        self._structure = None
        self._rmax = None
        self._pbc = None
        self._resolution = PairHistogram.resolution
        self._radiation_type = "X"
        # generated data
        self._x = []
        self._y = []
        self._x_cached = False
        self._y_cached = False
        # counts of pair types contributing to each x
        # list of dictionaries of atom pair indices (i <= j) and their counts
        self._ij_count = []
        # store coloring in separate list to allow shallow copy of structure
        self._site_coloring = []
        # normalized scattering factor
        self._nmsf = {}
        self._meansf = None
        self._customsf = {}
        self._sf_cached = False
        # finally assign arguments:
        self.setStructure(stru)
        self.setRmax(rmax)
        self.setPBC(pbc)
        if resolution is not None:
            self.setResolution(resolution)
        return


    # public methods


    def x(self):
        """List of unique pair distances.

        Return a copy of internal list.
        """
        if not self._x_cached:
            self._update_x()
        return self._x[:]


    def y(self):
        """List of pair distance weights.

        Return a copy of internal list.
        """
        if not self._y_cached:
            self._update_y()
        return self._y[:]


    def countBars(self):
        """Number of unique pair distances, same as len(self.x()).

        Return int.
        """
        if not self._x_cached:
            self._update_x()
        return len(self._x)


    def countAtoms(self):
        """Number of atoms in the internal structure model.

        Return int.
        """
        return len(self._structure)


    def nmsf(self, element):
        """Normalized scattering factor of specified element in structure.
        Returns scattering factor at the active radiation type normalized
        by the average factor of all atoms in the structure.

        element  -- case-insensitive symbol for element or isotope.
                    Must be present in internal structure.

        See also setRadiationType.

        Return float.
        Raise ValueError if element is not present in the structure.
        """
        if not self._sf_cached:
            self._update_sf()
        if element not in self._nmsf:
            emsg = 'Structure does not contain any "%s"' % element
            raise ValueError, emsg
        return self._nmsf[element]


    def meansf(self):
        """Mean value of scattering factor for the active radiation type.

        Return float.
        """
        if not self._sf_cached:
            self._update_sf()
        return self._meansf


    def setScatteringFactors(self, sfs):
        """Set custom scattering factor values for atom species in the model.

        sfs  -- dictionary of custom (element : scattering_factor) values.
                Any elements contained in the sfs use associated value
                as their scattering factor.  For all other atoms, the
                scattering factor is looked up for active radiation type.
                The element symbol in sfs may be arbitrary, for example
                sfs = {"A" : 0.1, "B" : 0.9}.  Use sfs={} to remove custom
                values.

        No return value.
        """
        self._customsf = dict(sfs)
        self._uncache('y', 'sf')
        return


    def setStructure(self, stru):
        """Specify structure model for which to calculate pair histogram.

        stru -- structure model, instance of Structure from diffpy.Structure.
                Makes a copy of stru for internal storage.

        No return value.
        """
        # Note: So far no handling of occupancies.
        for a in stru:
            assert a.occupancy == 1
        from diffpy.Structure import Structure
        self._uncache('x', 'y', 'sf')
        self._structure = Structure(stru)
        self._site_coloring = [a.element for a in self._structure]
        return


    def getStructure(self):
        """Structure model for which the pair histogram is calculated.

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
        # is is probably more work to check for sf cache than to update it
        self._uncache('y', 'sf')
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
        """Exchange atom types at sites i and j.  This performs smart
        update of internal pair distance weights.

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
        # uncomment to disable quick update:
        # self._uncache('y')
        if not self._y_cached:
            return
        # Here y is up to date.  We can do quick update of weights.
        # This makes about 40% gain in speed, but is more tricky
        # to maintain, because _y gets updated at two places.
        sf1 = [self.nmsf(smbl) for smbl in self._site_coloring]
        # scattering factors before flip
        sf0 = sf1[:]
        sf0[j], sf0[i] = sf1[i], sf1[j]
        # build a set of all pairs containing i or j
        ithenj = [i] * self.countAtoms() + [j] * self.countAtoms()
        kalltwice = range(self.countAtoms()) * 2
        ijset = set(zip(ithenj, kalltwice) + zip(kalltwice, ithenj))
        ijset.remove((i, j))
        ijset.remove((j, i))
        # calculate change of y due to i, j swap
        ychange = self.countBars() * [0.0]
        for idx in range(self.countBars()):
            ijpairs = ijset.intersection(self._ij_count[idx])
            for ij1 in ijpairs:
                i1, j1 = ij1
                cnt = self._ij_count[idx][ij1]
                ychange[idx] -= sf0[i1] * sf0[j1] * cnt
                ychange[idx] += sf1[i1] * sf1[j1] * cnt
            # normalize
            ychange[idx] /= self.countAtoms()
        # _y must be assigned new list to allow cheap copy
        self._y = [self._y[idx] + ychange[idx]
                    for idx in range(self.countBars())]
        return


    def setRmax(self, rmax):
        """Change upper limit up to which the pair distances are calculated.

        rmax -- new upper boundary for pair distances.

        No return value.
        """
        # quick reduction of histogram if it is up to date
        if self._rmax and rmax < self._rmax and self._x_cached:
            idx = bisect.bisect_right(self._x, rmax)
            del self._x[idx:]
            del self._y[idx:]
        elif rmax > self._rmax:
            self._uncache('x', 'y')
        self._rmax = float(rmax)
        return


    def getRmax(self):
        """Upper radius limit for calculating pair distances.

        Return float.
        """
        return self._rmax


    def setPBC(self, pbc):
        """Specify if periodic boundary conditions apply for structure model.

        pbc  -- boolean flag for periodic boundary conditions

        No return value.
        """
        pbcflag = bool(pbc)
        if pbcflag is not self._pbc:
            self._uncache('x', 'y')
        self._pbc = pbcflag
        return


    def getPBC(self):
        """True when periodic boundary conditions are in effect.

        Return bool.
        """
        return self._pbc


    def setResolution(self, resolution):
        """Specify radius resolution for distinguishing two close distances.
        If a difference of two distances is less or equal resolution, they
        are merged together in the histogram.

        resolution -- new distance resolution, must be non-negative.

        No return value.
        Raise ValueError for invalid resolution.
        """
        res = float(resolution)
        if res < 0:
            emsg = "resolution must be non-negative."
            raise ValueError, emsg
        if res != self._resolution:
            self._uncache('x', 'y')
        self._resolution = res
        return


    def getResolution(self):
        """Radius resolution for separating two close distances.

        Return float.
        """
        return self._resolution


    def setRadiationType(self, tp):
        """Specify radiation type for obtaining scattering factors.
        Default radiation type is "X".

        tp   -- radiation type, "X" for x-rays or "N" for neutrons.

        No return value.
        Raise ValueError for invalid argument.
        """
        if not tp in ("X", "N"):
            emsg = 'Radiation type must be "X" or "N".'
            raise ValueError, emsg
        if tp != self._radiation_type:
            self._uncache("y", "sf")
        self._radiation_type = tp
        return


    def getRadiationType(self):
        """Identify radiation type for obtaining scattering factors.

        Return "X" for x-rays or "N" for neutrons.
        """
        return self._radiation_type


    # protected methods


    def _update_x(self):
        """Recalculate unique pair distances.
        Updates data in _x and _ij_count.

        No return value.
        """
        self._x = []
        self._ij_count = []
        if self.getPBC():
            dstij = self._allPairsCrystal()
        else:
            dstij = self._allPairsCluster()
        # loop over all merged intervals
        dlast = -1e32
        dsums = []
        dcnts = []
        res = self.getResolution()
        for dij, i, j in dstij:
            if dij - dlast > res:
                dsums.append(0.0)
                dcnts.append(0)
                ijcnt = {}
                self._ij_count.append(ijcnt)
            dsums[-1] += dij
            dcnts[-1] += 1
            # ij is sorted tuple of pair indices
            ij = i <= j and (i, j) or (j, i)
            ijcnt[ij] = ijcnt.get(ij, 0) + 1
            dlast = dij
        # set x to mean values of merged distances
        self._x = [s/c for s, c in zip(dsums, dcnts)]
        assert len(self._x) == len(self._ij_count)
        self._x_cached = True
        return


    def _update_y(self):
        """Recalculate weighed counts of unique pair lengths.
        Updates data in _y.

        No return value.
        """
        # array of weighed scattering factors of site i
        # call to nmsf makes sure scattering factors are up to date
        sf = [self.nmsf(smbl) for smbl in self._site_coloring]
        # call to countBars updates _x
        self._y = self.countBars() * [0.0]
        for idx in range(self.countBars()):
            for ij, cnt in self._ij_count[idx].iteritems():
                i, j = ij
                self._y[idx] += sf[i] * sf[j] * cnt
            # normalize by number of atoms in the structure
            self._y[idx] /= self.countAtoms()
        self._y_cached = True
        return


    def _update_sf(self):
        """Recalculate normalized scattering factor for elements in the
        internal structure.  Updates data in _nmsf, _meansf.

        No return value.
        """
        pf = self._getPdfFit()
        radtp = self.getRadiationType()
        self._nmsf = {}
        sfsum = 0.0
        for smbl in self._site_coloring:
            if smbl in self._customsf:
                sfa = self._customsf[smbl]
            else:
                sfa = pf.get_scat(radtp, smbl)
            self._nmsf[smbl] = sfa
            sfsum += sfa
        # calculate mean scattering factor
        self._meansf = self.countAtoms() and sfsum/self.countAtoms() or 0.0
        # normalize scattering factors by their mean value
        for smbl in self._nmsf:
            self._nmsf[smbl] /= self._meansf
        self._sf_cached = True
        return


    def _uncache(self, *args):
        """Reset cached flag for a list of internal attributes.

        *args -- list of strings, currently supported are "x", "y", "sf"

        No return value.
        Raise AttributeError for any invalid args.
        """
        for a in args:
            attrname = "_" + a + "_cached"
            setattr(self, attrname, False)
        return


    def _allPairsCluster(self):
        """Pair lengths up to rmax without periodic boundary conditions.

        Return sorted list of tuples (d, i, j), where i and j are indices
        of contributing atoms.
        """
        dstij = []
        for i in range(0, self.countAtoms()):
            for j in range(i + 1, self.countAtoms()):
                dst = self._structure.distance(i, j)
                dstij.append((dst, i, j))
                dstij.append((dst, j, i))
        dstij.sort()
        idx = bisect.bisect_right(dstij, (self.getRmax(), 0, 0))
        del dstij[idx:]
        return dstij


    def _allPairsCrystal(self):
        """Pair lengths up to rmax assuming periodic boundary conditions.

        Return sorted list of tuples (d, i, j), where i and j are indices
        of contributing atoms.
        """
        pf = self._getPdfFit()
        # Because, this method only evaluates geometry, let us use
        # all carbon atoms so there are no problems with unknown types.
        carbonstru = self.getStructure()
        for a in carbonstru:
            a.element = "C"
        pf.add_structure(carbonstru)
        rmin = 1.0e-8
        rmax = self.getRmax()
        bldict = pf.bond_length_types('ALL', 'ALL', rmin, rmax)
        dstij = [ (d, i, j)
                for d, (i, j) in zip(bldict['dij'], bldict['ij0']) ]
        return dstij


    @staticmethod
    def _getPdfFit():
        """Common shared instance of PdfFit from diffpy.pdffit2.
        This methods shuts off PdfFit output and resets instance.

        Return reference to PdfFit.
        """
        if PairHistogram.__pdffit is None:
            import diffpy.pdffit2
            sink = open('/dev/null', 'w')
            diffpy.pdffit2.redirect_stdout(sink)
            pf2 = diffpy.pdffit2.PdfFit()
            PairHistogram.__pdffit = pf2
        PairHistogram.__pdffit.reset()
        return PairHistogram.__pdffit


# End of class PairHistogram
