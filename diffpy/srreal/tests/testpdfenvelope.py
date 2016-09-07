#!/usr/bin/env python

"""Unit tests for the PDFEnvelope class from diffpy.srreal.pdfcalculator
"""


import unittest
import cPickle
import numpy

from diffpy.srreal.pdfenvelope import PDFEnvelope, makePDFEnvelope
from diffpy.srreal.pdfenvelope import QResolutionEnvelope, ScaleEnvelope
from diffpy.srreal.pdfenvelope import SphericalShapeEnvelope, StepCutEnvelope
from diffpy.srreal.pdfcalculator import PDFCalculator

##############################################################################
class TestPDFEnvelope(unittest.TestCase):

    def setUp(self):
        self.fstepcut = PDFEnvelope.createByType('stepcut')
        self.fstepcut.stepcut = 5
        self.fscale = PDFEnvelope.createByType('scale')
        return


    def tearDown(self):
        return


    def test___init__(self):
        """check PDFEnvelope.__init__()
        """
        self.assertEqual(1.0, self.fscale.scale)
        self.fscale._setDoubleAttr('scale', 2.0)
        self.assertEqual(2.0, self.fscale.scale)
        return


    def test___call__(self):
        """check PDFEnvelope.__call__()
        """
        x = numpy.arange(0, 9.1, 0.3)
        # this is a virtual method in the base class
        self.assertRaises(RuntimeError, PDFEnvelope().__call__, 37)
        self.assertEqual(0.0, self.fstepcut(10))
        self.assertEqual(1.0, self.fstepcut(3.45))
        ycheck = numpy.array(17 * [1] + 14 * [0])
        self.assertTrue(numpy.array_equal(ycheck, self.fstepcut(x)))
        self.assertEqual(1.0, self.fscale(3.45))
        self.assertEqual(1.0, self.fscale(345))
        self.assertTrue(numpy.array_equal(numpy.ones_like(x), self.fscale(x)))
        self.fscale.scale = -2
        self.assertEqual(-2.0, self.fscale(3.5))
        return


    def test_clone(self):
        """check PDFEnvelope.clone
        """
        # this is a virtual method in the base class
        self.assertRaises(RuntimeError, PDFEnvelope().clone)
        self.fstepcut.stepcut = 17
        e2 = self.fstepcut.clone()
        self.assertEqual('stepcut', e2.type())
        self.assertEqual(17.0, e2.stepcut)
        self.assertEqual(17.0, e2._getDoubleAttr('stepcut'))
        return


    def test_create(self):
        """check PDFEnvelope.create
        """
        # this is a virtual method in the base class
        self.assertRaises(RuntimeError, PDFEnvelope().create)
        self.assertEqual('stepcut', self.fstepcut.create().type())
        self.assertEqual('scale', self.fscale.create().type())
        self.fstepcut.stepcut = 17
        self.assertEqual(0.0, self.fstepcut.create().stepcut)
        return


    def test_type(self):
        """check PDFEnvelope.type
        """
        # this is a virtual method in the base class
        self.assertRaises(RuntimeError, PDFEnvelope().type)
        self.assertEqual('stepcut', self.fstepcut.type())
        self.assertEqual('scale', self.fscale.type())
        return


    def test_createByType(self):
        """check PDFEnvelope.createByType()
        """
        self.assertRaises(ValueError, PDFEnvelope.createByType, 'notregistered')
        return


    def test_getRegisteredTypes(self):
        """check PDFEnvelope.getRegisteredTypes
        """
        regtypes = PDFEnvelope.getRegisteredTypes()
        self.assertTrue(2 <= len(regtypes))
        self.assertTrue('stepcut' in regtypes)
        self.assertTrue('scale' in regtypes)
        return


    def test_pickling(self):
        '''check pickling and unpickling of PDFEnvelope.
        '''
        stp = self.fstepcut
        stp.stepcut = 11
        stp2 = cPickle.loads(cPickle.dumps(stp))
        self.assertEqual('stepcut', stp2.type())
        self.assertEqual(11, stp2.stepcut)
        self.assertEqual(11, stp2._getDoubleAttr('stepcut'))
        return


    def test_makePDFEnvelope(self):
        '''check the makePDFEnvelope wrapper.
        '''
        pbl = makePDFEnvelope('parabolaenvelope',
                parabola_envelope, a=1, b=2, c=3)
        self.assertEqual(3, pbl(0))
        self.assertEqual(6, pbl(1))
        self.assertEqual(11, pbl(2))
        pbl.b = 0
        self.assertEqual([7, 3, 28], map(pbl, [-2, 0, 5]))
        pbl2 = pbl.clone()
        self.assertEqual(1, pbl2.a)
        self.assertEqual(0, pbl2.b)
        self.assertEqual(3, pbl2.c)
        self.assertEqual([7, 3, 28], map(pbl2, [-2, 0, 5]))
        pbl3 = PDFEnvelope.createByType('parabolaenvelope')
        self.assertEqual(1, pbl3.a)
        self.assertEqual(2, pbl3.b)
        self.assertEqual(3, pbl3.c)
        pbl3.a = 0
        pbl3.foo = 'asdf'
        pbl3cp = cPickle.loads(cPickle.dumps(pbl3))
        self.assertEqual(0, pbl3cp.a)
        self.assertEqual('asdf', pbl3cp.foo)
        pc = PDFCalculator()
        pc.envelopes = (pbl2,)
        pc2 = cPickle.loads(cPickle.dumps(pc))
        pbl2cp = pc2.envelopes[0]
        self.assertEqual('parabolaenvelope', pbl2cp.type())
        self.assertEqual(1, pbl2cp.a)
        self.assertEqual(0, pbl2cp.b)
        self.assertEqual(3, pbl2cp.c)
        return

# ----------------------------------------------------------------------------

class TestQResolutionEnvelope(unittest.TestCase):

    def setUp(self):
        self.evlp = QResolutionEnvelope()
        return


    def test_type(self):
        self.assertEqual('qresolution', self.evlp.type())
        self.assertTrue(hasattr(self.evlp, 'qdamp'))
        return


    def test_pickling(self):
        evlp = self.evlp
        evlp.qdamp = 3
        evlp2 = cPickle.loads(cPickle.dumps(evlp))
        self.assertEqual(QResolutionEnvelope, type(evlp2))
        self.assertEqual(3, evlp2.qdamp)
        return

# ----------------------------------------------------------------------------

class TestScaleEnvelope(unittest.TestCase):

    def setUp(self):
        self.evlp = ScaleEnvelope()
        return


    def test_type(self):
        self.assertEqual('scale', self.evlp.type())
        self.assertTrue(hasattr(self.evlp, 'scale'))
        return


    def test_pickling(self):
        evlp = self.evlp
        evlp.scale = 3
        evlp2 = cPickle.loads(cPickle.dumps(evlp))
        self.assertEqual(ScaleEnvelope, type(evlp2))
        self.assertEqual(3, evlp2.scale)
        return

# ----------------------------------------------------------------------------

class TestSphericalShapeEnvelope(unittest.TestCase):

    def setUp(self):
        self.evlp = SphericalShapeEnvelope()
        return


    def test_type(self):
        self.assertEqual('sphericalshape', self.evlp.type())
        self.assertTrue(hasattr(self.evlp, 'spdiameter'))
        return


    def test_pickling(self):
        evlp = self.evlp
        evlp.spdiameter = 3
        evlp2 = cPickle.loads(cPickle.dumps(evlp))
        self.assertEqual(SphericalShapeEnvelope, type(evlp2))
        self.assertEqual(3, evlp2.spdiameter)
        return

# ----------------------------------------------------------------------------

class TestStepCutEnvelope(unittest.TestCase):

    def setUp(self):
        self.evlp = StepCutEnvelope()
        return


    def test_type(self):
        self.assertEqual('stepcut', self.evlp.type())
        self.assertTrue(hasattr(self.evlp, 'stepcut'))
        return


    def test_pickling(self):
        evlp = self.evlp
        evlp.stepcut = 3
        evlp2 = cPickle.loads(cPickle.dumps(evlp))
        self.assertEqual(StepCutEnvelope, type(evlp2))
        self.assertEqual(3, evlp2.stepcut)
        return

# ----------------------------------------------------------------------------

def parabola_envelope(x, a, b, c):
    'parabola function for wrapping by makePDFEnvelope'
    return a * x**2 + b * x + c

if __name__ == '__main__':
    unittest.main()

# End of file
