#!/usr/bin/env python

"""Helper routines for running other unit tests.

TestCaseObjCrystOptional -- use this as a TestCase base class that
    disables unit tests when pyobjcryst is not installed.

TestCasePeriodictableOptional -- use this as a TestCase base class to
    skip unit tests when periodictable is not installed.
"""


from diffpy.srreal.structureconverters import convertObjCrystCrystal
from diffpy.srreal.tests import logger

# class TestCaseObjCrystOptional

try:
    import pyobjcryst.crystal
    from unittest import TestCase as TestCaseObjCrystOptional
    convertObjCrystCrystal(pyobjcryst.crystal.Crystal())
except ImportError:
    TestCaseObjCrystOptional = object
    logger.warning('Cannot import pyobjcryst, pyobjcryst tests skipped.')
except TypeError:
    TestCaseObjCrystOptional = object
    logger.warning('Compiled without ObjCryst, pyobjcryst tests skipped.')

# class TestCasePeriodictableOptional

try:
    import periodictable
    from unittest import TestCase as TestCasePeriodictableOptional
except ImportError:
    TestCasePeriodictableOptional = object
    logger.warning('Cannot import periodictable, periodictable tests skipped.')

# helper functions

def datafile(filename):
    from pkg_resources import resource_filename
    rv = resource_filename(__name__, "testdata/" + filename)
    return rv


def loadObjCrystCrystal(filename):
    from pyobjcryst.crystal import CreateCrystalFromCIF
    fullpath = datafile(filename)
    crst = CreateCrystalFromCIF(open(fullpath))
    return crst


def loadDiffPyStructure(filename):
    from diffpy.Structure import Structure
    fullpath = datafile(filename)
    stru = Structure(filename=fullpath)
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

    def countSites(self):
        return 0

    def createBondGenerator(self):
        from diffpy.srreal.structureadapter import BaseBondGenerator
        return BaseBondGenerator(self)

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
