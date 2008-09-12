########################################################################
#
# diffpy.srreal     by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2008 Trustees of the Columbia University
#                   in the city of New York.  All rights reserved.
#
# File coded by:    Pavol Juhas
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
########################################################################


"""class CompareHistograms -- compare reference and model histograms.
"""

# module version
__id__ = "$Id$"


import bisect
import math


class CompareHistograms(object):
    """Compare reference and model histograms.  This object can
    evaluate match in bin positions and match in associated amplitudes.

    Class data:

    binsize -- threshold for inserting zero-amplitude bins, when
               x_model is far away from the nearest x_reference.
               Default value Inf, which compares to reference bins.
    """

    # class data
    _binsize = float('Inf')


    def __init__(self, refhist, modelhist=None, binsize=None):
        """Create CompareHistograms object for given reference histogram.

        refhist   -- reference histogram
        modelhist -- model histogram that is compared with the reference.
                     Both refhist and modelhist can be specified either as
                     a tuple of (xlist, ylist) or as a class instance, which
                     has x() and y() methods that return bin centers and
                     associated amplitudes.
        binsize   -- cutoff for inserting zero-amplitude bins, when model
                     x value is far away from the nearest reference.
                     Instance override of the default class value.
        """
        # create data items
        # inputs
        self._xref = []
        self._yref = []
        self._xmod = []
        self._ymod = []
        self._binsize = CompareHistograms._binsize
        # generated data
        self._xbins = []
        self._xbinszero = set()
        self._xbinsedges = []
        self._xbins_cached = False
        self._yrefbinned = []
        self._yrefbinned_cached = False
        self._ymodbinned = []
        self._ymodbinned_cached = False
        self._tsdx = None
        self._tsdx_cached = False
        self._tsdy = None
        self._tsdy_cached = False
        # finally assign arguments:
        self.setReference(refhist)
        if modelhist is not None:
            self.setModel(modelhist)
        if binsize is not None:
            self.setBinSize(binsize)
        return


    # public methods


    def xref(self):
        """List of bin centers of the reference histogram.

        See also setReference.

        Return a copy of the internal list.
        """
        return self._xref[:]


    def yref(self):
        """List of bin amplitudes of the reference histogram.

        See also setReference.

        Return a copy of the internal list.
        """
        return self._yref[:]
    

    def xmod(self):
        """List of bin centers of the model histogram that is compared
        to the reference.

        See also setModel.

        Return a copy of the internal list.
        """
        return self._xmod[:]


    def ymod(self):
        """List of bin amplitudes of the model histogram that is compared
        to the reference.

        See also setModel.

        Return a copy of the internal list.
        """
        return self._ymod[:]
    

    def setReference(self, refhist):
        """Specify the reference histogram.

        refhist -- reference histogram.  It can be a tuple of (xlist, ylist)
                   or a class instance with x() and y() methods.

        See also setModel, xref, yref.

        No return value.
        Raise TypeError for invalid argument type.
        """
        tpl = CompareHistograms._histargToTuple(refhist)
        self._xref, self._yref = tpl[0][:], tpl[1][:]
        self._uncache('xbins', 'yrefbinned', 'ymodbinned', 'tsdx', 'tsdy')
        return


    def setModel(self, modhist):
        """Specify the model histogram to be compared with the reference.

        modhist -- model histogram.  It can be a tuple of (xlist, ylist)
                   or a class instance with x() and y() methods.

        See also setReference, xmod, ymod.

        No return value.
        Raise TypeError for invalid argument type.
        """
        tpl = CompareHistograms._histargToTuple(modhist)
        self._xmod, self._ymod = tpl[0][:], tpl[1][:]
        self._uncache('ymodbinned', 'tsdx', 'tsdy')
        return


    def getBinSize(self):
        """Active cutoff for adding zero-amplitude bins when
        neighboring x values in the reference are far apart.

        See also setBinSize.

        Return float.
        """
        return self._binsize


    def setBinSize(self, binsize):
        """Specify new cutoff for adding zero-amplitude bins when
        neighboring x values in the reference are far apart.

        See also getBinSize.

        No return value.
        """
        self._binsize = float(binsize)
        self._uncache('xbins', 'yrefbinned', 'ymodbinned', 'tsdx', 'tsdy')
        return


    def xbins(self):
        """Reference to the internal list of x-bin centers.  Intended as
        read-only, do not change the returned value.  When binsize is smaller
        than the largest separation of neighboring xref, xbins contain
        additional fake values with zero amplitudes.  The fake zero values
        are stored in the xbinszero set.

        See also setBinSize, xbinszero.

        Return reference to the internal list.
        """
        if not self._xbins_cached:
            self._update_xbins()
        return self._xbins


    def xbinszero(self):
        """Set of extra zero elements in xbins that were inserted in
        addition to values from xref.  Intended as read-only return
        value, do not modify.

        See also setBinSize, xbins.

        Return reference to the internal set.
        """
        if not self._xbins_cached:
            self._update_xbins()
        return self._xbinszero


    def xbinsedges(self):
        """Reference to the internal list of x-bin boundaries.  Return value
        is intended as read-only, do not modify.  The xbinsedges have one
        element less than the xbins list.

        See also xbins.

        Return reference to the internal list.
        """
        if not self._xbins_cached:
            self._update_xbins()
        return self._xbinsedges


    def countBins(self):
        """Total number of x-bins used for comparison, i.e., all reference and
        faked zero bins.  Same as the length of xbins.

        See also xbins.

        Return int.
        """
        return len(self.xbins())


    def yrefbinned(self):
        """Y-values from the reference histogram rebinned to xbins.  They may
        include zero bins in addition to yref items.  Returned list is
        read-only, do not modify.

        See also binsize, xbins, ymodbinned.

        Return reference to the internal list.
        """
        if not self._yrefbinned_cached:
            self._update_yrefbinned()
        return self._yrefbinned


    def ymodbinned(self):
        """Y-values from the model histogram rebineed to xbins.
        Returned list is read-only, do not modify.

        See also binsize, xbins, yrefbinned.

        Return reference to the internal list.
        """
        if not self._ymodbinned_cached:
            self._update_ymodbinned()
        return self._ymodbinned


    def tsdx(self):
        """Total SquaredDifference in X.  Sum of squares of the
        differences of the model X-values from the nearest
        xbin center.

        Return float.
        """
        if not self._tsdx_cached:
            self._update_tsdx()
        return self._tsdx


    def tsdy(self):
        """Total Squared Difference in Y.  Sum of squares of the
        differences of the rebinned Y-values from the nearest
        xbin center.

        Return float.
        """
        if not self._tsdy_cached:
            self._update_tsdy()
        return self._tsdy


    def msdx(self):
        """Mean Squared Difference in X.  The msdx is calculated from the
        differences between model and nearest reference X-values.

        Return float.
        """
        if self.countBins() == 0:
            rv = 0.0
        else:
            rv = self.tsdx() / self.countBins()
        return rv


    def msdy(self):
        """Mean Squared Difference in Y.  The msdy is calculated from the
        differences between model and reference histogram amplitudes.

        Return float.
        """
        if self.countBins() == 0:
            rv = 0.0
        else:
            rv = self.tsdy() / self.countBins()
        return rv


    # protected methods

    def _update_xbins(self):
        """Recalculate common x-bins from xref and active binsize.
        Update data in _xbins, _xbinszero, _xbinsedges.

        No return value.
        Raise ValueError if _xref is not sorted.
        """
        self._xbins = []
        self._xbinszero = set()
        # build the _xbins list and _xbinszero set:
        self._xbins += self._xref[:1]
        for xhi in self._xref[1:]:
            xlo = self._xbins[-1]
            xstep = xhi - xlo
            if xstep <= 0:
                emsg = "Reference histogram must have sorted unique x-values."
                raise ValueError, emsg
            nzero = int(math.floor(xstep / self.getBinSize()))
            xsepzero = xstep / (nzero + 1)
            for i in range(nzero):
                xzero = xlo + (i + 1)*xsepzero
                self._xbins.append(xzero)
                self._xbinszero.add(xzero)
            self._xbins.append(xhi)
        # calculate the edges:
        self._xbinsedges = [ (self._xbins[i] + self._xbins[i + 1]) / 2.0
                for i in range(0, len(self._xbins) - 1) ]
        assert len(self._xbinszero) + len(self._xref) == len(self._xbins)
        assert len(self._xbinsedges) + 1 == len(self._xbins)
        self._xbins_cached = True
        return


    def _update_yrefbinned(self):
        """Recalculate the list of rebinned Y-values from the reference
        histogram.  The rebinned list may contain extra zero values. 
        Update data in _yrefbinned.

        No return value.
        """
        self._yrefbinned = []
        xbz = self.xbinszero()
        yri = iter(self.yref())
        for x in self.xbins():
            if x in xbz:
                y = 0.0
            else:
                y = yri.next()
            self._yrefbinned.append(y)
        self._yrefbinned_cached = True
        return


    def _update_ymodbinned(self):
        """Recalculate the list of rebinned Y-values from the model
        histogram.  Update data in _ymodbinned.

        No return value.
        """
        edges = self.xbinsedges()
        self._ymodbinned = [0.0] * self.countBins()
        for xm, ym in zip(self._xmod, self._ymod):
            idx = bisect.bisect(edges, xm)
            self._ymodbinned[idx] += ym
        self._ymodbinned_cached = True
        return


    def _update_tsdx(self):
        """Recalculate the total squared difference in X.
        Update data in _tsdx.

        No return value.
        """
        edges = self.xbinsedges()
        xb = self.xbins()
        self._tsdx = 0.0
        for xm in self._xmod:
            idx = bisect.bisect(edges, xm)
            dx = xb[idx] - xm
            self._tsdx += dx*dx
        self._tsdx_cached = True
        return


    def _update_tsdy(self):
        """Recalculate the total squared difference in Y.
        Update data in _tsdy.

        No return value.
        """
        self._tsdy = 0.0
        for yrb, ymb in zip(self.yrefbinned(), self.ymodbinned()):
            dy = yrb - ymb
            self._tsdy += dy*dy
        self._tsdy_cached = True
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


    # static class methods


    @staticmethod
    def _histargToTuple(histarg):
        """Convert histogram argument to a tuple of (xlist, ylist).
        This is a utility function is used by setReference and setModel.

        histarg -- tuple of (xlist, ylist) or class instance that has
                   x() and y() methods.

        Return a tuple of x and y lists.
        Raise TypeError for invalid argument type.
        """
        import types
        rv = None
        if type(histarg) is types.TupleType and len(histarg) == 2:
            rv = histarg
        else:
            try:
                rv = (histarg.x(), histarg.y())
            except AttributeError:
                pass
        if rv is None:
            emsg = "Argument must be a tuple or instance with x(), y() methods."
            raise TypeError, emsg
        return rv


# End of class CompareHistograms
