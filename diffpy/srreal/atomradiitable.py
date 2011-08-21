#!/usr/bin/env python
##############################################################################
#
# diffpy.srreal     by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2011 Trustees of the Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Pavol Juhas
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################


"""class AtomRadiiTable -- storage of empirical atom radii.
"""

# module version
__id__ = "$Id$"

# exported items, these also makes them show in pydoc.
__all__ = ['AtomRadiiTable', 'ZeroRadiiTable', 'CovalentRadiiTable']

import copy
from diffpy.srreal.srreal_ext import AtomRadiiTable, ZeroRadiiTable

# class CovalentRadiiTable ---------------------------------------------------

class CovalentRadiiTable(AtomRadiiTable):
    '''Covalent radii from Cordero et al., 2008, doi:10.1039/b801115j.
    Instantiation of this class requires the periodictable module.
    '''

    # class variable that holds the periodictable.elements object
    _elements = None
    # flag that checks if this class has been already registered
    __registered = False

    def __init__(self):
        '''Initialize the CovalentRadiiTable class.
        This makes sure that the periodictable.elements can be imported.
        '''
        super(CovalentRadiiTable, self).__init__()
        if self._elements is None:
            from periodictable import elements
            CovalentRadiiTable._elements = elements
        assert self._elements is not None
        if not CovalentRadiiTable.__registered:
            CovalentRadiiTable.__registered = True
            self._registerThisType()
            assert self.type() in AtomRadiiTable.getRegisteredTypes()
        return


    def _tableLookup(self, smbl):
        '''Return covalent atom radius in Angstroms.

        smbl -- string symbol of an element

        Return float.  Raise ValueError for unknown element symbol.
        '''
        e = self._elements.isotope(smbl)
        rv = e.covalent_radius
        if rv is None:
            emsg = "Undefined covalent radius for %r." % smbl
            raise ValueError(emsg)
        return rv

    # HasClassRegistry overloads:

    def create(self):
        '''Create new instance of the CovalentRadiiTable.
        '''
        return CovalentRadiiTable()


    def clone(self):
        '''Return a new duplicate instance of self.
        '''
        return copy.copy(self)


    def type(self):
        '''Unique string identifier of the CovalentRadiiTable type.
        This is used for class registration and as an argument for the
        createByType function.

        Return string.
        '''
        return "covalentradii"

# End of class CovalentRadiiTable

# class AtomRadiiTable ----------------------------------------------------------

# pickling support

def _atomradiitable_getstate(self):
    state = (self.__dict__, )
    return state

def _atomradiitable_setstate(self, state):
    if len(state) != 1:
        emsg = ("expected 1-item tuple in call to __setstate__, got " +
                repr(state))
        raise ValueError(emsg)
    self.__dict__.update(state[0])
    return

def _atomradiitable_reduce(self):
    from diffpy.srreal.srreal_ext import _AtomRadiiTable_tostring
    args = (_AtomRadiiTable_tostring(self),)
    rv = (_atomradiitable_create, args, self.__getstate__())
    return rv

def _atomradiitable_create(s):
    from diffpy.srreal.srreal_ext import _AtomRadiiTable_fromstring
    return _AtomRadiiTable_fromstring(s)

# inject pickle methods

AtomRadiiTable.__getstate__ = _atomradiitable_getstate
AtomRadiiTable.__setstate__ = _atomradiitable_setstate
AtomRadiiTable.__reduce__ = _atomradiitable_reduce

ZeroRadiiTable.__getstate__ = _atomradiitable_getstate
ZeroRadiiTable.__setstate__ = _atomradiitable_setstate
ZeroRadiiTable.__reduce__ = _atomradiitable_reduce

# End of file
