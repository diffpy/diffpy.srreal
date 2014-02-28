#!/usr/bin/env python
##############################################################################
#
# diffpy.srreal     by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2010 The Trustees of Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Pavol Juhas
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE_DANSE.txt for license information.
#
##############################################################################


"""ParallelPairQuantity  -- proxy class for converting PairQuantity types
into parallel calculators.
"""


# exported items
__all__ = ['createParallelCalculator']

import copy
import inspect
from diffpy.srreal.attributes import Attributes

# ----------------------------------------------------------------------------

def createParallelCalculator(pqobj, ncpu, pmap):
    '''Create a proxy parallel calculator to a PairQuantity instance.

    pqobj    -- instance of PairQuantity calculator to be run in parallel
    ncpu     -- number of parallel jobs
    pmap     -- a parallel map function used to submit job to workers

    Return a proxy calculator instance that has the same interface,
    but executes the calculation in parallel split among ncpu jobs.
    '''

    class ParallelPairQuantity(Attributes):

        '''Class for running parallel calculations.  This is a proxy class
        to the wrapper PairQuantity type with the same interface.

        Instance data:

        pqobj    -- the master PairQuantity object to be evaluated in parallel
        ncpu     -- number of parallel jobs
        pmap     -- a parallel map function used to submit job to workers
        '''

        def __init__(self, pqobj, ncpu, pmap):
            '''Initialize a parallel proxy to the PairQuantity instance.

            pqobj    -- instance of PairQuantity calculator to be run
                        in parallel
            ncpu     -- number of parallel jobs
            pmap     -- a parallel map function used to submit job to workers
            '''
            # use explicit assignment to avoid setattr forwarding to the pqobj
            object.__setattr__(self, 'pqobj', pqobj)
            object.__setattr__(self, 'ncpu', ncpu)
            object.__setattr__(self, 'pmap', pmap)
            return


        def eval(self, stru=None):
            '''Perform parallel calculation and return internal value array.

            stru -- object that can be converted to StructureAdapter,
                    e.g., example diffpy Structure or pyobjcryst Crystal.
                    Use the last structure when None.

            Return numpy array.
            '''
            # use StructureAdapter for faster pickles
            from diffpy.srreal.structureadapter import createStructureAdapter
            if stru is None:
                struadpt = self.pqobj.getStructure()
            else:
                struadpt = createStructureAdapter(stru)
            self.pqobj.setStructure(struadpt)
            kwd = { 'cpuindex' : None,
                    'ncpu' : self.ncpu,
                    'pqobj' : copy.copy(self.pqobj),
                    }
            # shallow copies of kwd dictionary each with a unique cpuindex
            arglist = [kwd.copy() for kwd['cpuindex'] in range(self.ncpu)]
            for pdata in self.pmap(_parallelData, arglist):
                self.pqobj._mergeParallelData(pdata, self.ncpu)
            return self.pqobj.value


        def __call__(self, *args, **kwargs):
            '''Call the wrapped calculator using parallel evaluation.

            The arguments and return value are the same as for the wrapped
            PairQuantity calculator.
            '''
            savedeval = self.pqobj.__dict__.get('eval')
            def restore_eval():
                if savedeval:
                    self.pqobj.eval = savedeval
                else:
                    self.pqobj.__dict__.pop('eval', None)
            def parallel_eval(stru):
                assert self.pqobj.eval is parallel_eval
                restore_eval()
                return self.eval(stru)
            self.pqobj.eval = parallel_eval
            try:
                rv = self.pqobj(*args, **kwargs)
            finally:
                restore_eval()
            return rv

    # class ParallelPairQuantity

    # Create proxy method and properties to the wrapped PairQuantity

    pqtype = type(pqobj)

    # create proxy methods to all public methods and some protected methods

    proxy_forced = set('''_getDoubleAttr _setDoubleAttr _hasDoubleAttr
        _namesOfDoubleAttributes _namesOfWritableDoubleAttributes
        '''.split())

    def _make_proxymethod(name, f):
        def proxymethod(self, *args, **kwargs):
            pqobj = object.__getattribute__(self, 'pqobj')
            return f(pqobj, *args, **kwargs)
        proxymethod.__name__ = name
        proxymethod.__doc__ = f.__doc__
        return proxymethod

    for n, f in inspect.getmembers(pqtype, inspect.ismethod):
        ignore = n not in proxy_forced and (n.startswith('_') or
                hasattr(ParallelPairQuantity, n))
        if ignore:  continue
        setattr(ParallelPairQuantity, n, _make_proxymethod(n, f))

    # create proxy properties to all properties that do not conflict with
    # existing class items

    def _make_proxyproperty(prop):
        fget = fset = fdel = None
        if prop.fget:
            def fget(self):  return prop.fget(self.pqobj)
        if prop.fset:
            def fset(self, value):  return prop.fset(self.pqobj, value)
        if prop.fdel:
            def fdel(self):  return prop.fdel(self.pqobj)
        return property(fget, fset, fdel, prop.__doc__)

    for n, p in inspect.getmembers(pqtype, lambda x: type(x) is property):
        if hasattr(ParallelPairQuantity, n):  continue
        setattr(ParallelPairQuantity, n, _make_proxyproperty(p))

    # finally create an instance of this very custom class
    return ParallelPairQuantity(pqobj, ncpu, pmap)


def _parallelData(kwd):
    '''Helper for calculating and fetching raw results from a worker node.
    '''
    pqobj = kwd['pqobj']
    pqobj._setupParallelRun(kwd['cpuindex'], kwd['ncpu'])
    pqobj.eval()
    return pqobj._getParallelData()

# End of file
