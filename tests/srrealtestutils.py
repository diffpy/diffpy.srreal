#!/usr/bin/env python

"""Helper routines for running other unit tests.

TestCaseObjCrystOptional -- use this as a TestCase base class that
    disables unit tests when pyobjcryst is not installed.
"""

# version
__id__ = '$Id$'

import logging
import os.path

try:
    import pyobjcryst
    from unittest import TestCase as TestCaseObjCrystOptional
except ImportError:
    TestCaseObjCrystOptional = object
    logging.warning('Cannot import pyobjcryst, pyobjcryst tests skipped.')


# useful variables
thisfile = locals().get('__file__', 'file.py')
tests_dir = os.path.dirname(os.path.abspath(thisfile))
testdata_dir = os.path.join(tests_dir, 'testdata')

# helper functions

def datafile(filename):
    rv = os.path.join(testdata_dir, filename)
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
