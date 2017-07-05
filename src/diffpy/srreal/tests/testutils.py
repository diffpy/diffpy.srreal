#!/usr/bin/env python

"""Helper routines for running other unit tests.
"""


import sys
import copy
import numpy

from diffpy.srreal.structureconverters import convertObjCrystCrystal
from diffpy.srreal.tests import logger

# Handle renamed diffpy.structure module by importing it to a common name.
# TODO simplify when Python 2 support is dropped.

if sys.version_info[0] >= 3:
    import diffpy.structure as mod_structure
else:
    import diffpy.Structure as mod_structure

# Resolve availability of optional packages.

# pyobjcryst

_msg_nopyobjcryst = "No module named 'pyobjcryst'"
try:
    import pyobjcryst.crystal
    convertObjCrystCrystal(pyobjcryst.crystal.Crystal())
    has_pyobjcryst = True
except ImportError:
    has_pyobjcryst = False
    logger.warning('Cannot import pyobjcryst, pyobjcryst tests skipped.')
except TypeError:
    has_pyobjcryst = False
    logger.warning('Compiled without ObjCryst, pyobjcryst tests skipped.')

# periodictable

_msg_noperiodictable = "No module named 'periodictable'"
try:
    import periodictable
    has_periodictable = True
    # silence the pyflakes syntax checker
    del periodictable
except ImportError:
    has_periodictable = False
    logger.warning('Cannot import periodictable, periodictable tests skipped.')

# helper functions

def datafile(filename):
    from pkg_resources import resource_filename
    rv = resource_filename(__name__, "testdata/" + filename)
    return rv


def loadObjCrystCrystal(filename):
    from pyobjcryst import loadCrystal
    fullpath = datafile(filename)
    crst = loadCrystal(fullpath)
    return crst


def loadDiffPyStructure(filename):
    fullpath = datafile(filename)
    stru = mod_structure.loadStructure(fullpath)
    return stru

# helper class for testing overloading of StructureAdapter

from diffpy.srreal.structureadapter import StructureAdapter
from diffpy.srreal.structureadapter import AtomicStructureAdapter
from diffpy.srreal.structureadapter import PeriodicStructureAdapter
from diffpy.srreal.structureadapter import CrystalStructureAdapter

class HasCustomPQConfig(object):

    cpqcount = 0

    def _customPQConfig(self, pqobj):
        self.cpqcount += 1
        return


class DerivedStructureAdapter(HasCustomPQConfig, StructureAdapter):

    def __init__(self):
        StructureAdapter.__init__(self)
        self.positions = []


    def clone(self):
        rv = DerivedStructureAdapter()
        rv.positions[:] = copy.deepcopy(self.positions)
        return rv


    def createBondGenerator(self):
        from diffpy.srreal.structureadapter import BaseBondGenerator
        return BaseBondGenerator(self)


    def countSites(self):
        return len(self.positions)

    # reuse base totalOccupancy
    # reuse base numberDensity

    def siteAtomType(self, idx):
        self._checkindex(idx)
        return "Cu"


    def siteCartesianPosition(self, idx):
        return self.positions[idx]


    def siteMultiplicity(self, idx):
        self._checkindex(idx)
        return 2 * StructureAdapter.siteMultiplicity(self, idx)


    def siteOccupancy(self, idx):
        self._checkindex(idx)
        return 0.5 * StructureAdapter.siteOccupancy(self, idx)

    def siteAnisotropy(self, idx):
        self._checkindex(idx)
        return False


    def siteCartesianUij(self, idx):
        self._checkindex(idx)
        return numpy.identity(3, dtype=float) * 0.005


    def _checkindex(self, idx):
        self.positions[idx]
        return

# End of class DerivedStructureAdapter

class DerivedAtomicStructureAdapter(
        HasCustomPQConfig, AtomicStructureAdapter):
    pass

class DerivedPeriodicStructureAdapter(
        HasCustomPQConfig, PeriodicStructureAdapter):
    pass

class DerivedCrystalStructureAdapter(
        HasCustomPQConfig, CrystalStructureAdapter):
    pass

# End of file
