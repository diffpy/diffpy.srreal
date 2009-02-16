#!/usr/bin/env python

"""Unit tests for atomconflicts.py
"""

# version
__id__ = '$Id$'

import os
import unittest

from diffpy.srreal.pdf import converters

thisfile = locals().get('__file__', 'file.py')
tests_dir = os.path.dirname(os.path.abspath(thisfile))
testdata_dir = os.path.join(tests_dir, 'testdata')

class TestConverters(unittest.TestCase):

    def test_Converters(self):
        """Assert that stru == structureFromPhase(phaseFromStructure(stru))"""
        from diffpy.Structure import Structure
        S1 = Structure()
        S1.read(os.path.join(testdata_dir, "ni.stru"), "pdffit")

        p = converters.phaseFromStructure(S1)
        S2 = converters.structureFromPhase(p)

        self.assertEqual(S1.lattice.abcABG(), S2.lattice.abcABG())
        self.assertEqual(len(S1), len(S2))
        for i in range(len(S1)):
            self.assertEqual(S1[i].element, S2[i].element)
            self.assertEqual(S1[i].occupancy, S2[i].occupancy)
            self.assertTrue((S1[i].xyz == S2[i].xyz).all())
            self.assertTrue((S1[i].U == S2[i].U).all())

        return

if __name__ == "__main__":
    unittest.main()
