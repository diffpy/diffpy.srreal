#!/usr/bin/env python
##############################################################################
#
# diffpy.srreal     by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2011 The Trustees of Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Pavol Juhas
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE_DANSE.txt for license information.
#
##############################################################################


"""class AtomRadiiTable -- storage of empirical atom radii.
"""


# exported items, these also makes them show in pydoc.
__all__ = ['AtomRadiiTable', 'ConstantRadiiTable', 'CovalentRadiiTable']

from diffpy.srreal.srreal_ext import AtomRadiiTable, ConstantRadiiTable

# class CovalentRadiiTable ---------------------------------------------------

class CovalentRadiiTable(AtomRadiiTable):
    '''Covalent radii from Cordero et al., 2008, doi:10.1039/b801115j.
    Instantiation of this class requires the periodictable module.
    '''

    # class variable that holds the periodictable.elements object
    _elements = None

    def _standardLookup(self, smbl):
        '''Return covalent atom radius in Angstroms.

        smbl -- string symbol of an element

        Return float.  Raise ValueError for unknown element symbol.
        '''
        if CovalentRadiiTable._elements is None:
            from periodictable import elements
            CovalentRadiiTable._elements = elements
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
        import copy
        return copy.copy(self)


    def type(self):
        '''Unique string identifier of the CovalentRadiiTable type.
        This is used for class registration and as an argument for the
        createByType function.

        Return string.
        '''
        return "covalent"

# End of class CovalentRadiiTable
CovalentRadiiTable()._registerThisType()

# End of file
