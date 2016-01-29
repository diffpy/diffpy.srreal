#!/usr/bin/env python

"""Unit tests for diffpy.srreal.sfaverage
"""


import unittest
import numpy

from diffpy.srreal.sfaverage import SFAverage
from diffpy.srreal.scatteringfactortable import ScatteringFactorTable
from diffpy.srreal.tests.testutils import loadDiffPyStructure
from diffpy.srreal.tests.testutils import TestCaseObjCrystOptional
from diffpy.srreal.tests.testutils import loadObjCrystCrystal


##############################################################################
class TestSFAverage(unittest.TestCase):

    def setUp(self):
        self.sftx = ScatteringFactorTable.createByType('X')
        self.sftn = ScatteringFactorTable.createByType('N')
        return

    def tearDown(self):
        return


    def test_fromStructure_CdSe(self):
        """check SFAverage.fromStructure() for CdSe
        """
        cdse = loadDiffPyStructure('CdSe_cadmoselite.cif')
        sfavg = SFAverage.fromStructure(self.sftx, cdse)
        fcd = self.sftx.lookup('Cd')
        fse = self.sftx.lookup('Se')
        self.assertTrue(isinstance(sfavg.f1sum, float))
        self.assertAlmostEqual(0.5 * (fcd + fse), sfavg.f1avg)
        self.assertAlmostEqual(0.5 * (fcd**2 + fse**2), sfavg.f2avg)
        self.assertEqual(4, sfavg.count)
        self.assertEqual(cdse.composition, sfavg.composition)
        qa = numpy.arange(0, 25, 0.1)
        sfavg2 = SFAverage.fromStructure(self.sftx, cdse, qa)
        self.assertTrue(isinstance(sfavg2.f1sum, numpy.ndarray))
        self.assertNotEqual(sfavg2.f1sum[0], sfavg2.f1sum[-1])
        self.assertEqual(sfavg.f1sum, sfavg2.f1sum[0])
        self.assertEqual(sfavg.f2sum, sfavg2.f2sum[0])
        sfavg3 = SFAverage.fromStructure(self.sftn, cdse, qa)
        self.assertEqual(sfavg3.f1sum[0], sfavg3.f1sum[-1])
        self.assertRaises(TypeError, SFAverage.fromStructure,
                self.sftx, 'notastructure')
        return


    def test_fromComposition(self):
        """check SFAverage.fromComposition()
        """
        sfavg1 = SFAverage.fromComposition(self.sftx, {'Na' : 1, 'Cl' : 1})
        fm = ['Na', 0.25, 'Cl', 0.75, 'Cl', 0.25, 'Na', 0.5, 'Na', 0.25]
        smblcnts = zip(fm[0::2], fm[1::2])
        sfavg2 = SFAverage.fromComposition(self.sftx, smblcnts)
        self.assertEqual(sfavg1.f1sum, sfavg2.f1sum)
        self.assertEqual(sfavg1.f2sum, sfavg2.f2sum)
        self.assertEqual(sfavg1.count, sfavg2.count)
        sfempty = SFAverage.fromComposition(self.sftx, [])
        self.assertEqual(0, sfempty.f1avg)
        self.assertEqual(0, sfempty.f2avg)
        self.assertEqual(0, sfempty.count)
        return

# End of class TestSFAverage

##############################################################################
class TestSFAverageObjCryst(TestCaseObjCrystOptional):

    def setUp(self):
        self.sftx = ScatteringFactorTable.createByType('X')
        return

    def tearDown(self):
        return


    def test_from_rutile(self):
        """check SFAverage.fromStructure for pyobjcryst Crystal of rutile.
        """
        rutile = loadObjCrystCrystal('rutile.cif')
        qa = numpy.arange(0, 25, 0.1)
        sfavg = SFAverage.fromStructure(self.sftx, rutile, qa)
        fti = self.sftx.lookup('Ti', qa)
        fo = self.sftx.lookup('O', qa)
        self.assertTrue(numpy.allclose((fti + 2 * fo) / 3.0, sfavg.f1avg))
        fti2, fo2 = fti**2, fo**2
        self.assertTrue(numpy.allclose((fti2 + 2 * fo2) / 3.0, sfavg.f2avg))
        self.assertEqual(6, sfavg.count)

# End of class TestSFAverageObjCryst

if __name__ == '__main__':
    unittest.main()

# End of file
