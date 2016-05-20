#!/usr/bin/env python

"""Unit tests for the PeakWidthModel classes from diffpy.srreal.peakwidthmodel
"""


import unittest
import cPickle

from diffpy.srreal.peakwidthmodel import PeakWidthModel
from diffpy.srreal.peakwidthmodel import DebyeWallerPeakWidth, JeongPeakWidth
from diffpy.srreal.pdfcalculator import PDFCalculator
from diffpy.srreal.structureadapter import createStructureAdapter
from diffpy.srreal.tests.testutils import loadDiffPyStructure

##############################################################################
class TestPeakWidthModel(unittest.TestCase):

    tio2stru = None
    tio2adpt = None

    def setUp(self):
        self.pwconst = PeakWidthModel.createByType('constant')
        self.pwconst.width = 2
        if self.tio2stru is None:
            self.tio2stru = loadDiffPyStructure('rutile.cif')
            self.tio2adpt = createStructureAdapter(self.tio2stru)
        return


    def tearDown(self):
        return


    def test___init__(self):
        """check PeakWidthModel.__init__()
        """
        self.assertEqual(2.0, self.pwconst.width)
        self.pwconst._setDoubleAttr('width', 3.0)
        self.assertEqual(3.0, self.pwconst.width)
        return


    def test_create(self):
        """check PeakWidthModel.create
        """
        # this is a virtual method in the base class
        self.assertRaises(RuntimeError, PeakWidthModel().create)
        self.assertEqual('constant', self.pwconst.create().type())
        self.pwconst.width = 17
        self.assertEqual(0.0, self.pwconst.create().width)
        return


    def test_clone(self):
        """check PeakWidthModel.clone
        """
        # this is a virtual method in the base class
        self.assertRaises(RuntimeError, PeakWidthModel().clone)
        self.pwconst.width = 17
        pwc2 = self.pwconst.clone()
        self.assertEqual('constant', pwc2.type())
        self.assertEqual(17.0, pwc2.width)
        self.assertEqual(17.0, pwc2._getDoubleAttr('width'))
        return


    def test_type(self):
        """check PeakWidthModel.type
        """
        # this is a virtual method in the base class
        self.assertRaises(RuntimeError, PeakWidthModel().type)
        self.assertEqual('constant', self.pwconst.type())
        return


    def test_calculate(self):
        """check PeakWidthModel.calculate()
        """
        pwm = PeakWidthModel()
        bnds = self.tio2adpt.createBondGenerator()
        self.assertRaises(RuntimeError, pwm.calculate, bnds)
        self.assertEqual(2.0, self.pwconst.calculate(bnds))
        return


    def test_maxWidth(self):
        """check PeakWidthModel.maxWidth()
        """
        self.assertRaises(RuntimeError, PeakWidthModel().maxWidth,
                self.tio2adpt, 0, 10)
        self.assertEqual(2.0, self.pwconst.maxWidth(self.tio2adpt, 0, 10))
        self.assertEqual(2.0, self.pwconst.maxWidth(self.tio2stru, 0, 10))
        return


    def test_ticker(self):
        """check PeakWidthModel.ticker()
        """
        from diffpy.srreal.eventticker import EventTicker
        et0 = EventTicker(self.pwconst.ticker())
        self.pwconst.width = 3
        et1 = self.pwconst.ticker()
        self.assertNotEqual(et0, et1)
        self.assertTrue(et0 < et1)
        return


    def test_ticker_override(self):
        """check PeakWidthModel.ticker override in a Python-derived class.
        """
        pwm = MyPWM()
        self.assertEqual(0, pwm.tcnt)
        et0 = pwm.ticker()
        self.assertEqual(1, pwm.tcnt)
        et1 = PeakWidthModel.ticker(pwm)
        self.assertEqual(1, pwm.tcnt)
        self.assertEqual(et0, et1)
        et0.click()
        self.assertEqual(et0, et1)
        # check that implicit ticker call from PDFCalculator is
        # handled by the Python ticker override.
        pc = PDFCalculator()
        pc.peakwidthmodel = pwm
        pc.ticker()
        self.assertEqual(2, pwm.tcnt)
        return


    def test_getRegisteredTypes(self):
        """check PeakWidthModel.getRegisteredTypes
        """
        regtypes = PeakWidthModel.getRegisteredTypes()
        self.assertTrue(3 <= len(regtypes))
        self.assertTrue(regtypes.issuperset(
            ['constant', 'debye-waller', 'jeong']))
        return


    def test_pickling(self):
        '''check pickling and unpickling of PeakWidthModel.
        '''
        pwc = self.pwconst
        pwc.width = 11
        pwc2 = cPickle.loads(cPickle.dumps(pwc))
        self.assertEqual('constant', pwc2.type())
        self.assertEqual(11, pwc2.width)
        self.assertEqual(11, pwc2._getDoubleAttr('width'))
        return

# ----------------------------------------------------------------------------

class TestDebyeWallerPeakWidth(unittest.TestCase):

    def setUp(self):
        self.pwm = DebyeWallerPeakWidth()
        return


    def test_type(self):
        """check DebyeWallerPeakWidth.type
        """
        self.assertEqual('debye-waller', self.pwm.type())
        self.assertEqual(0, len(self.pwm._namesOfDoubleAttributes()))
        return


    def test_pickling(self):
        """check pickling of DebyeWallerPeakWidth class.
        """
        self.assertEqual('debye-waller', self.pwm.type())
        pwm = self.pwm
        pwm2 = cPickle.loads(cPickle.dumps(pwm))
        self.assertEqual(DebyeWallerPeakWidth, type(pwm2))
        return

# ----------------------------------------------------------------------------

class TestJeongPeakWidth(unittest.TestCase):

    def setUp(self):
        self.pwm = JeongPeakWidth()
        return


    def test_type(self):
        """check JeongPeakWidth.type
        """
        self.assertEqual('jeong', self.pwm.type())
        self.assertTrue(hasattr(self.pwm, 'delta1'))
        self.assertTrue(hasattr(self.pwm, 'delta2'))
        self.assertTrue(hasattr(self.pwm, 'qbroad'))
        return


    def test_pickling(self):
        """check pickling of the DebyeWallerPeakWidth class
        """
        pwm = self.pwm
        pwm.delta1 = 1
        pwm.delta2 = 2
        pwm.qbroad = 3
        pwm2 = cPickle.loads(cPickle.dumps(pwm))
        self.assertEqual(JeongPeakWidth, type(pwm2))
        self.assertEqual(1, pwm2.delta1)
        self.assertEqual(2, pwm2.delta2)
        self.assertEqual(3, pwm2.qbroad)
        return

# ----------------------------------------------------------------------------

class MyPWM(PeakWidthModel):
    "Helper class for testing PeakWidthModelOwner."

    pwmscale = 1.5

    def __init__(self):
        PeakWidthModel.__init__(self)
        self._registerDoubleAttribute('pwmscale')
        return

    def type(self):
        return "mypwm"

    def create(self):
        return MyPWM()

    def clone(self):
        import copy
        return copy.copy(self)

    def calculate(self, bnds):
        return self.pwmscale * bnds.msd()

    tcnt = 0
    def ticker(self):
        self.tcnt += 1
        return PeakWidthModel.ticker(self)

MyPWM()._registerThisType()

# End of class MyPWM

class TestPeakWidthOwner(unittest.TestCase):

    def setUp(self):
        self.pc = PDFCalculator()
        self.pwm = MyPWM()
        self.pc.peakwidthmodel = self.pwm
        return


    def test_pwmtype(self):
        '''Check type of the owned PeakWidthModel instance.
        '''
        self.assertEqual('mypwm', self.pc.peakwidthmodel.type())
        return


    def test_pickling(self):
        '''Check pickling of an owned PeakWidthModel instance.
        '''
        pc1 = cPickle.loads(cPickle.dumps(self.pc))
        self.pwm.pwmscale = 3
        pc2 = cPickle.loads(cPickle.dumps(self.pc))
        self.assertEqual('mypwm', pc1.peakwidthmodel.type())
        self.assertEqual(1.5, pc1.peakwidthmodel.pwmscale)
        self.assertEqual(1.5, pc1.pwmscale)
        self.assertEqual('mypwm', pc2.peakwidthmodel.type())
        self.assertEqual(3, pc2.peakwidthmodel.pwmscale)
        self.assertEqual(3, pc2.pwmscale)
        return

# ----------------------------------------------------------------------------


if __name__ == '__main__':
    unittest.main()

# End of file
