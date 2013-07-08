#!/usr/bin/env python

"""Helper routines for running other unit tests.

TestCaseObjCrystOptional -- use this as a TestCase base class that
    disables unit tests when pyobjcryst is not installed.
"""


import logging
import os.path

# class TestCaseObjCrystOptional

try:
    import pyobjcryst
    from unittest import TestCase as TestCaseObjCrystOptional
except ImportError:
    TestCaseObjCrystOptional = object
    logging.warning('Cannot import pyobjcryst, pyobjcryst tests skipped.')

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

# End of file
