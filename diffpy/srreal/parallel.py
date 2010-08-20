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


"""ParallelPairQuantity  -- proxy class for converting PairQuantity types
into parallel calculators.
"""

# module version
__id__ = "$Id$"

# exported items
__all__ = ['ParallelPairQuantity']

# ----------------------------------------------------------------------------

class ParallelPairQuantity(object):

    '''Class for running parallel calculations.  This is a proxy class
    with the same interface as the wrapped PairQuantity type.

    Instance data:

    pqclass  -- some concrete PairQuantity class.
    ncpu     -- number of parallel jobs
    pmap     -- a parallel map function used to submit job to workers
    '''

    def __init__(self, pqtype, ncpu, pmap, **kwargs):
        '''Initialize a parallel proxy to the PairQuantity pqtype

        pqclass  -- some concrete PairQuantity class.
        ncpu     -- number of parallel jobs
        pmap     -- a parallel map function used to submit job to workers
        kwargs   -- optional keyword arguments that are passed to the
                    pqclass constructor

        No return value.
        '''
        self.pqtype = pqtype
        self.ncpu = ncpu
        self.pmap = pmap
        self._pqobj = self.pqtype(**kwargs)
        return


    def eval(self, structure):
        '''Perform parallel calculation and return internal value array.

        structure    -- structure to be evaluated, an instance of
                        diffpy Structure or pyobjcryst Crystal
        kwargs       -- optional parameter settings for this calculator

        Return numpy array.
        '''
        self._pqobj.setStructure(structure)
        kwd = { 'ncpu' : self.ncpu,
                'pqobj' : self._pqobj,
                'structure' : structure }
        arglist = [kwd.copy() for kwd['cpuindex'] in range(self.ncpu)]
        for y in self.pmap(_partialValue, arglist):
            self._pqobj._mergeParallelValue(y)
        return self._pqobj.value()


    def __call__(self, *args, **kwargs):
        '''Call the wrapped instance as a function, but with parallel evaluation.
        Uses the same arguments and return values as wrapped pqtype.
        '''
        savedeval = self._pqobj.__dict__.get('eval')
        def restore_eval():
            if savedeval:
                self._pqobj.eval = savedeval
            else:
                self._pqobj.__dict__.pop('eval', None)
        def parallel_eval(structure):
            assert self._pqobj.eval is parallel_eval
            restore_eval()
            return self.eval(structure)
        self._pqobj.eval = parallel_eval
        try:
            rv = self._pqobj(*args, **kwargs)
        finally:
            restore_eval()
        return rv

# class ParallelPairQuantity


def _partialValue(kwd):
    '''Helper function for calculating partial value on a worker node.
    '''
    pqobj = kwd['pqobj']
    pqobj._setupParallelRun(kwd['cpuindex'], kwd['ncpu'])
    return pqobj.eval(kwd['structure'])


# End of file
