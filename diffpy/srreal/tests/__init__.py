#!/usr/bin/env python
##############################################################################
#
# diffpy.srreal     by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2012 The Trustees of Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Pavol Juhas
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE_DANSE.txt for license information.
#
##############################################################################

"""Unit tests for diffpy.srreal.
"""


# create logger instance for the tests subpackage
import logging

logging.basicConfig()
logger = logging.getLogger(__name__)
del logging


def testsuite():
    '''Build a unit tests suite for the diffpy.srreal package.

    Return a unittest.TestSuite object.
    '''
    import unittest
    modulenames = '''
        diffpy.srreal.tests.testatomradiitable
        diffpy.srreal.tests.testattributes
        diffpy.srreal.tests.testbondcalculator
        diffpy.srreal.tests.testbvscalculator
        diffpy.srreal.tests.testdebyepdfcalculator
        diffpy.srreal.tests.testoverlapcalculator
        diffpy.srreal.tests.testpairquantity
        diffpy.srreal.tests.testparallel
        diffpy.srreal.tests.testpdfbaseline
        diffpy.srreal.tests.testpdfcalcobjcryst
        diffpy.srreal.tests.testpdfcalculator
        diffpy.srreal.tests.testpdfenvelope
        diffpy.srreal.tests.testpeakprofile
        diffpy.srreal.tests.testpeakwidthmodel
        diffpy.srreal.tests.testscatteringfactortable
        diffpy.srreal.tests.testsfaverage
        diffpy.srreal.tests.teststructureadapter
    '''.split()
    suite = unittest.TestSuite()
    loader = unittest.defaultTestLoader
    mobj = None
    for mname in modulenames:
        exec ('import %s as mobj' % mname)
        suite.addTests(loader.loadTestsFromModule(mobj))
    return suite


def test():
    '''Execute all unit tests for the diffpy.srreal package.
    Return a unittest TestResult object.
    '''
    import unittest
    suite = testsuite()
    runner = unittest.TextTestRunner()
    result = runner.run(suite)
    return result


# End of file
