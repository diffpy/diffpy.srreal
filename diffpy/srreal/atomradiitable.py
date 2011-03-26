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
__all__ = ['AtomRadiiTable', 'CovalentRadiiTable']

from diffpy.srreal.srreal_ext import AtomRadiiTable


class CovalentRadiiTable(AtomRadiiTable):
    '''Covalent radii from Cordero et al., 2008, doi:10.1039/b801115j.
    Instantiation of this class requires the periodictable module.
    '''

    # class variable that holds the periodictable.elements object
    _elements = None


    def __init__(self):
        '''Initialize the CovalentRadiiTable class.
        This makes sure that the periodictable.elements can be imported.
        '''
        super(CovalentRadiiTable, self).__init__()
        if self._elements is None:
            from periodictable import elements
            CovalentRadiiTable._elements = elements
        assert self._elements is not None
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

# End of class CovalentRadiiTable

# End of file
