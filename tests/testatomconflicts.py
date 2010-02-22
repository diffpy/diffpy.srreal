#!/usr/bin/env python

"""Unit tests for atomconflicts.py
"""

# version
__id__ = '$Id$'

import os
import math
import unittest

# useful variables
thisfile = locals().get('__file__', 'file.py')
tests_dir = os.path.dirname(os.path.abspath(thisfile))
testdata_dir = os.path.join(tests_dir, 'testdata')

from diffpy.srreal.atomconflicts import getCovalentRadius
from diffpy.srreal.atomconflicts import AtomConflicts


##############################################################################
class TestRoutines(unittest.TestCase):

    def setUp(self):
        return

    def tearDown(self):
        return

    def test_getCovalentRadius(self):
        """check getCovalentRadius()
        """
        self.assertAlmostEqual(1.41, getCovalentRadius('Mg'), 3)
        self.assertAlmostEqual(1.19, getCovalentRadius('As'), 3)
        return

# End of class TestRoutines


##############################################################################
class TestAtomConflicts(unittest.TestCase):

    _structures_loaded = False
    rmax = 10
    rutile = None
    silicon = None
    sphalerite = None

    def setUp(self):
        """load test structures.
        """
        datanames = ("rutile", "silicon", "sphalerite")
        # load tested structures just one time
        if not TestAtomConflicts._structures_loaded:
            from diffpy.Structure import Structure
            for name in datanames:
                basename = name + ".cif"
                stru = Structure(filename=testdata(basename))
                cnfl = AtomConflicts(stru, self.rmax)
                setattr(TestAtomConflicts, name, stru)
            TestAtomConflicts._structures_loaded = True
        # prepare fresh atom conflict instances
        self.cnfl_rutile = AtomConflicts(self.rutile)
        self.cnfl_silicon = AtomConflicts(self.silicon)
        self.cnfl_sphalerite = AtomConflicts(self.sphalerite)
        return

    def tearDown(self):
        return

    def test___init__(self):
        """check AtomConflicts.__init__()
        """
        cnfl0 = self.cnfl_silicon
        self.assertEqual(0, cnfl0.countConflicts())
        return

    def test_getConflicts(self):
        """check AtomConflicts.getConflicts()
        """
        bigradia = {'Zn' : 1.0, 'S' : 1.4, 'Si' : 1.2}
        # sphalerite
        self.assertEqual([], self.cnfl_sphalerite.getConflicts())
        self.cnfl_sphalerite.setAtomRadiaGetter(bigradia.get)
        # there should be 4 conflicts per atom
        cnt0 = 4 * len(self.sphalerite)
        cnfls0 = self.cnfl_sphalerite.getConflicts()
        self.assertEqual(cnt0, len(cnfls0))
        # atom 0 should have 4 conflicts:
        i0neighbors = [ijdd[0] for ijdd in cnfls0 if ijdd[0] == 0]
        self.assertEqual(4, len(i0neighbors))
        # similar check for silicon
        self.assertEqual([], self.cnfl_silicon.getConflicts())
        self.cnfl_silicon.setAtomRadiaGetter(bigradia.get)
        cnt1 = 4 * len(self.silicon)
        cnfls1 = self.cnfl_silicon.getConflicts()
        self.assertEqual(cnt1, len(cnfls1))
        # atom 0 has 4 neighbors
        # atom 0 should have 4 conflicts:
        i0neighbors = [ijdd[1] for ijdd in cnfls1 if ijdd[0] == 0]
        self.assertEqual(4, len(i0neighbors))
        # check the nearest conflict distance
        d0 = cnfls1[0][2]
        dcheck = self.silicon.lattice.a * math.sqrt(3) / 4
        self.assertAlmostEqual(dcheck, d0, 8)
        # check conflict magnitude
        dd0 = cnfls1[0][3]
        ddcheck = 2*bigradia['Si'] - dcheck
        self.assertAlmostEqual(ddcheck, dd0, 8)
        return

    def test_countConflicts(self):
        """check AtomConflicts.countConflicts()
        """
        cnfl0 = self.cnfl_silicon
        self.assertEqual(0, cnfl0.countConflicts())
        bigradia = {'Si' : 1.2}
        cnfl0.setAtomRadiaGetter(bigradia.get)
        self.assertEqual(4*8, cnfl0.countConflicts())
        return

    def test_countAtoms(self):
        """check AtomConflicts.countAtoms()
        """
        self.assertEqual(6, self.cnfl_rutile.countAtoms())
        return

    def test_setStructure(self):
        """check AtomConflicts.setStructure()
        """
        bigradia = {'Si' : 1.2, 'Ti' : 0, 'O' : 0}
        self.cnfl_rutile.setAtomRadiaGetter(bigradia.get)
        self.assertEqual(0, self.cnfl_rutile.countConflicts())
        self.cnfl_rutile.setStructure(self.silicon)
        self.failUnless(0 < self.cnfl_rutile.countConflicts())
        self.cnfl_rutile.setStructure(self.rutile)
        self.assertEqual(0, self.cnfl_rutile.countConflicts())
        return

    def test_getStructure(self):
        """check AtomConflicts.getStructure()
        """
        stru = self.cnfl_rutile.getStructure()
        self.assertEqual(6, len(stru))
        return

    def test_setSiteColoring(self):
        """check AtomConflicts.setSiteColoring()
        """
        cnflsi = self.cnfl_silicon
        self.assertEqual(0, cnflsi.countConflicts())
        # Zn has larger covalent radius than Si
        cnflsi.setSiteColoring(['Zn'] * cnflsi.countAtoms())
        self.failUnless(0 < cnflsi.countConflicts())
        # put it back
        cnflsi.setSiteColoring(['Si'] * cnflsi.countAtoms())
        cnflsi = self.cnfl_silicon
        return

    def test_getSiteColoring(self):
        """check AtomConflicts.getSiteColoring()
        """
        colors = self.cnfl_rutile.getSiteColoring()
        self.assertEqual(2, colors.count('Ti'))
        self.assertEqual(4, colors.count('O'))
        return

    def test_flipSiteColoring(self):
        """check AtomConflicts.flipSiteColoring()
        """
        czns = self.cnfl_sphalerite
        iZn = czns.getSiteColoring().index('Zn')
        iS = czns.getSiteColoring().index('S')
        self.assertEqual(0, czns.countConflicts())
        czns.flipSiteColoring(iZn, iS)
        self.failUnless(0 < czns.countConflicts())
        czns.flipSiteColoring(iZn, iS)
        self.assertEqual(0, czns.countConflicts())
        return

    def test_setAtomRadiaGetter(self):
        """check AtomConflicts.setAtomRadiaGetter()
        """
        csi = self.cnfl_silicon
        self.assertEqual(0, csi.countConflicts())
        csi.setAtomRadiaGetter({'Si' : 1.2}.get)
        self.failUnless(0 < csi.countConflicts())
        csi.setAtomRadiaGetter(getCovalentRadius)
        self.assertEqual(0, csi.countConflicts())
        return

    def test_getRmax(self):
        """check AtomConflicts.getRmax()
        """
        csi = self.cnfl_silicon
        self.assertAlmostEqual(getCovalentRadius('Si'), csi.getRmax(), 4)
        csi.setSiteColoring(['Na']*csi.countAtoms())
        self.assertAlmostEqual(getCovalentRadius('Na'), csi.getRmax(), 4)
        csi.setAtomRadiaGetter({'Na' : 7}.get)
        self.assertEqual(7, csi.getRmax())
        return

#   def test__update_conflicts(self):
#       """check AtomConflicts._update_conflicts()
#       """
#       return
#
#   def test__update_pair_lengths(self):
#       """check AtomConflicts._update_pair_lengths()
#       """
#       return
#
#   def test__uncache(self):
#       """check AtomConflicts._uncache()
#       """
#       return

# End of class TestAtomConflicts


##############################################################################
# helper routines


def testdata(basename):
    """Prepend testdata_dir to the basename.
    """
    filename = os.path.join(testdata_dir, basename)
    return filename


if __name__ == '__main__':
    unittest.main()

# End of file
