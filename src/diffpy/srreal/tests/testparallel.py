#!/usr/bin/env python

"""Unit tests for diffpy.srreal.parallel
"""


import unittest
import multiprocessing
import numpy
from diffpy.srreal.tests.testutils import loadDiffPyStructure
from diffpy.srreal.parallel import createParallelCalculator

##############################################################################
class TestRoutines(unittest.TestCase):

    _pool = None
    ncpu = 4
    cdse = None
    nickel = None

    def setUp(self):
        if self.cdse is None:
            type(self).cdse = loadDiffPyStructure('CdSe_cadmoselite.cif')
            for a in self.cdse:  a.Uisoequiv = 0.003
        if self.nickel is None:
            type(self).nickel = loadDiffPyStructure('Ni.cif')
            for a in self.nickel:  a.Uisoequiv = 0.003
        return

    def tearDown(self):
        if self._pool:
            self._pool.terminate()
            self._pool.join()
        self._pool = None
        return

    @property
    def pool(self):
        if not self._pool:
            self._pool = multiprocessing.Pool(processes=self.ncpu)
        return self._pool

    def test_parallel_pdf(self):
        """check parallel PDFCalculator
        """
        from diffpy.srreal.pdfcalculator import PDFCalculator
        pdfc = PDFCalculator()
        r0, g0 = pdfc(self.cdse)
        ppdfc1 = createParallelCalculator(PDFCalculator(), 3, map)
        r1, g1 = ppdfc1(self.cdse)
        self.assertTrue(numpy.array_equal(r0, r1))
        self.assertTrue(numpy.allclose(g0, g1))
        ppdfc2 = createParallelCalculator(PDFCalculator(),
                self.ncpu, self.pool.imap_unordered)
        r2, g2 = ppdfc2(self.cdse)
        self.assertTrue(numpy.array_equal(r0, r2))
        self.assertTrue(numpy.allclose(g0, g2))
        pdfc.rmax = ppdfc1.rmax = ppdfc2.rmax = 5
        pdfc.qmax = ppdfc1.qmax = ppdfc2.qmax = 15
        r0a, g0a = pdfc()
        self.assertTrue(numpy.all(r0a <= 5))
        self.assertFalse(numpy.allclose(g0a, numpy.interp(r0a, r0, g0)))
        r1a, g1a = ppdfc1()
        self.assertTrue(numpy.array_equal(r0a, r1a))
        self.assertTrue(numpy.allclose(g0a, g1a))
        r2a, g2a = ppdfc2()
        self.assertTrue(numpy.array_equal(r0a, r2a))
        self.assertTrue(numpy.allclose(g0a, g2a))
        return

    def test_parallel_bonds(self):
        """check parallel BondCalculator
        """
        from diffpy.srreal.bondcalculator import BondCalculator
        nickel = self.nickel
        bc = BondCalculator()
        d0 = bc(nickel)
        pbc1 = createParallelCalculator(BondCalculator(), 3, map)
        d1 = pbc1(nickel)
        self.assertTrue(numpy.array_equal(d0, d1))
        pbc2 = createParallelCalculator(BondCalculator(),
                self.ncpu, self.pool.imap_unordered)
        d2 = pbc2(nickel)
        self.assertTrue(numpy.array_equal(d0, d2))
        bc.rmax = pbc1.rmax = pbc2.rmax = 2.5
        for bci in (bc, pbc1, pbc2):
            bci.maskAllPairs(False)
            bci.setPairMask(0, 'all', True)
            bci.filterCone([1, 0, 0], 48)
        d0a = bc(nickel)
        self.assertEqual(8, len(d0a))
        d1a = pbc1(nickel)
        self.assertTrue(numpy.array_equal(d0a, d1a))
        d2a = pbc2(nickel)
        self.assertTrue(numpy.array_equal(d0a, d2a))
        return

# End of class TestRoutines

if __name__ == '__main__':
    unittest.main()

# End of file
