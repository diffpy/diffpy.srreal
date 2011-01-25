#!/usr/bin/env python

"""Unit tests for diffpy.srreal.bonddistancecalculator
"""

# version
__id__ = '$Id$'

import os
import unittest
import cPickle

# useful variables
thisfile = locals().get('__file__', 'file.py')
tests_dir = os.path.dirname(os.path.abspath(thisfile))
testdata_dir = os.path.join(tests_dir, 'testdata')

from diffpy.srreal.bonddistancecalculator import BondDistanceCalculator
from diffpy.Structure import Structure

##############################################################################
class TestBondCalculator(unittest.TestCase):

    def setUp(self):
        self.bdc = BondDistanceCalculator()
        if not hasattr(self, 'rutile'):
            rutile_cif = os.path.join(testdata_dir, 'rutile.cif')
            TestBondCalculator.rutile = Structure(filename=rutile_cif)
        if not hasattr(self, 'nickel'):
            nickel_stru = os.path.join(testdata_dir, 'Ni.stru')
            TestBondCalculator.rutile = Structure(filename=nickel_stru)
        return


    def tearDown(self):
        return


    def test___init__(self):
        """check BondDistanceCalculator.__init__()
        """
        self.assertEqual(0, self.bdc.rmin)
        self.assertEqual(5, self.bdc.rmax)
        return


    def test___call__(self):
        """check BondDistanceCalculator.__call__()
        """
        bdc = self.bdc
        print bdc(self.rutile)
        bdc.rmax = 0
        self.assertEqual([], bdc(self.rutile))
        bdc.rmax = 2
        print bdc(self.rutile)
        self.assertEqual(12, len(bdc(self.rutile)))
        return


    def xtest_distances(self):
        """check BondDistanceCalculator.distances()
        """
        return


    def xtest_directions(self):
        """check BondDistanceCalculator.directions()
        """
        return


    def xtest_sites(self):
        """check BondDistanceCalculator.sites()
        """
        return


    def xtest_filterCone(self):
        """check BondDistanceCalculator.filterCone()
        """
        return


    def xtest_filterOff(self):
        """check BondDistanceCalculator.filterOff()
        """
        return


# End of class TestBondCalculator

if __name__ == '__main__':
    unittest.main()

# End of file
