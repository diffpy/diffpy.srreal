##############################################################################
#
# diffpy.srreal     by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2010 Trustees of the Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Pavol Juhas
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################


"""class PairQuantity    -- base class for Python defined calculators.
"""

# module version
__id__ = "$Id$"

# exported items
__all__ = ['PairQuantity']

from diffpy.srreal.srreal_ext import PairQuantity_ext

# ----------------------------------------------------------------------------

class PairQuantity(PairQuantity_ext):

    '''Base class for Python defined pair quantity calculators.
    No action by default.  Concrete calculators must overload the
    _addPairContribution method to get some results.
    '''

    def _resizeValue(self, sz):
        '''Resize the internal contributions array to the specified size.

        sz   -- new size of the internal array.

        No return value.
        '''
        PairQuantity_ext._resizeValue(self, sz)
        return


    def _resetValue(self):
        '''Reset all contributions in the internal array to zero.
        May be overloaded to add other calculator initializations.

        No return value.
        '''
        PairQuantity_ext._resetValue(self)
        return


    def _configureBondGenerator(self, bnds):
        '''Configure bond generator just before start the of summation.
        The default method sets the upper and lower limits for the pair
        distances.  An overloaded method should call the base one to
        preserve the distance limits setup.

        bnds -- instance of BaseBondGenerator to be configured.

        No return value.
        '''
        PairQuantity_ext._configureBondGenerator(self, bnds)
        return


    def _addPairContribution(self, bnds, sumscale):
        '''Process pair contribution at the bond generator state.
        No action by default, needs to be overloaded to do something.

        bnds     -- instance of BaseBondGenerator holding data for
                    a particular pair of atoms during summation.

        sumscale -- integer scaling for this contribution passed from
                    PQEvaluator.  Equals 1 if bnds.site0() == bnds.site1(),
                    2 otherwise.  Can be negative when contribution is
                    removed for fast quantity updates.

        No return value.
        '''
        PairQuantity_ext._addPairContribution(self, bnds)
        return

# class PairQuantity

# BasePairQuantity_ext pickling support --------------------------------------

from diffpy.srreal.srreal_ext import BasePairQuantity_ext

def _bpqe_getstate(self):
    maskdata = self._getMaskData()
    defaultmask = self.getPairMask(0, 0) ^ ((0, 0) in maskdata)
    state = (defaultmask, maskdata)
    return state

def _bpqe_setstate(self, state):
    if len(state) != 2:
        emsg = ("expected 2-item tuple in call to __setstate__, got %r" +
                repr(state))
        raise ValueError(emsg)
    st = iter(state)
    defaultmask = st.next()
    self.maskAllPairs(defaultmask)
    invdefaultmask = not defaultmask
    for i, j in st.next():
        self.maskSitePair(i, j, invdefaultmask)
    return

# inject pickle methods to PDFCalculator_ext

BasePairQuantity_ext.__getstate__ = _bpqe_getstate
BasePairQuantity_ext.__setstate__ = _bpqe_setstate

# End of file
