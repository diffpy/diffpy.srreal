#!/usr/bin/env python

"""Helper routines for running other unit tests."""


import copy
import pickle

import numpy

import diffpy.structure as mod_structure
from diffpy.srreal.structureadapter import (
    AtomicStructureAdapter,
    CrystalStructureAdapter,
    PeriodicStructureAdapter,
    StructureAdapter,
)
from diffpy.structure.parsers import getParser

# helper functions


def datafile(filename):
    from pathlib import Path

    rv = Path(__file__).parent / "testdata" / filename
    return str(rv)


def loadObjCrystCrystal(filename):
    from pyobjcryst.crystal import create_crystal_from_cif

    fullpath = datafile(filename)
    crst = create_crystal_from_cif(fullpath)
    return crst


def loadDiffPyStructure(filename):
    fullpath = datafile(filename)
    stru = mod_structure.loadStructure(fullpath)
    return stru


def loadCrystalStructureAdapter(filename):
    from diffpy.srreal.structureconverters import _fetchDiffPyStructureData

    fullpath = datafile(filename)
    pcif = getParser("cif")
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


def _maxNormDiff(yobs, ycalc):
    """Returned maximum difference normalized by RMS of the yobs."""
    yobsa = numpy.array(yobs)
    obsmax = numpy.max(numpy.fabs(yobsa)) or 1
    ynmdiff = (yobsa - ycalc) / obsmax
    rv = max(numpy.fabs(ynmdiff))
    return rv


# helper class for testing overloading of StructureAdapter


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


class DerivedAtomicStructureAdapter(HasCustomPQConfig, AtomicStructureAdapter):
    pass


class DerivedPeriodicStructureAdapter(
    HasCustomPQConfig, PeriodicStructureAdapter
):
    pass


class DerivedCrystalStructureAdapter(
    HasCustomPQConfig, CrystalStructureAdapter
):
    pass


# End of file
