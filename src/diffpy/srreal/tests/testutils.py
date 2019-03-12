#!/usr/bin/env python

"""Helper routines for running other unit tests.
"""


import copy
import numpy
import pickle

from diffpy.srreal.structureconverters import convertObjCrystCrystal
from diffpy.srreal.tests import logger

# Deprecated in 1.3 - import of old camel-case diffpy.Structure names.
# TODO drop this in version 1.4.

try:
    import diffpy.structure as mod_structure
    from diffpy.structure.parsers import getParser
except ImportError as e:
    try:
        import diffpy.Structure as mod_structure
        from diffpy.Structure.Parsers import getParser
    except ImportError:
        raise e
    del e

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


def loadCrystalStructureAdapter(filename):
    from diffpy.srreal.structureconverters import _fetchDiffPyStructureData
    fullpath = datafile(filename)
    pcif = getParser('cif')
    stru = pcif.parseFile(fullpath)
    asu = stru[:0] + pcif.asymmetric_unit
    adpt = CrystalStructureAdapter()
    _fetchDiffPyStructureData(adpt, asu)
    for op in pcif.spacegroup.iter_symops():
        adpt.addSymOp(op.R, op.t)
    return adpt


def pickle_with_attr(obj, **attr):
    "Return pickle dump after setting one or more attributes."
    assert attr, "keyword argument must be set"
    for k, v in attr.items():
        setattr(obj, k, v)
    rv = pickle.dumps(obj)
    return rv

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
