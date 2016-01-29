#!/usr/bin/env python

"""Unit tests for the PDFBaseline class from diffpy.srreal.pdfcalculator
"""


import unittest
import cPickle

from diffpy.srreal.pdfbaseline import PDFBaseline, makePDFBaseline
from diffpy.srreal.pdfbaseline import ZeroBaseline, LinearBaseline

##############################################################################
class TestPDFBaseline(unittest.TestCase):

    def setUp(self):
        self.linear = PDFBaseline.createByType('linear')
        self.zero = PDFBaseline.createByType('zero')
        return


    def tearDown(self):
        return


    def test___init__(self):
        """check PDFBaseline.__init__()
        """
        self.assertEqual(0.0, self.linear.slope)
        self.linear._setDoubleAttr('slope', 2.0)
        self.assertEqual(2.0, self.linear.slope)
        return


    def test___call__(self):
        """check PDFBaseline.__call__()
        """
        # this is a virtual method in the base class
        self.assertRaises(RuntimeError, PDFBaseline().__call__, 37)
        self.assertEqual(0.0, self.zero(10))
        self.assertEqual(0.0, self.zero(3.45))
        self.assertEqual(0.0, self.linear(3.45))
        self.assertEqual(0.0, self.linear(345))
        self.linear.slope = -2
        self.assertEqual(-7.0, self.linear(3.5))
        self.assertEqual(-2.0, self.linear._getDoubleAttr('slope'))
        return


    def test_clone(self):
        """check PDFBaseline.clone
        """
        # this is a virtual method in the base class
        self.assertRaises(RuntimeError, PDFBaseline().clone)
        self.linear.slope = 17
        bl2 = self.linear.clone()
        self.assertEqual('linear', bl2.type())
        self.assertEqual(17.0, bl2.slope)
        self.assertEqual(17.0, bl2._getDoubleAttr('slope'))
        return


    def test_create(self):
        """check PDFBaseline.create
        """
        # this is a virtual method in the base class
        self.assertRaises(RuntimeError, PDFBaseline().create)
        self.assertEqual('zero', self.zero.create().type())
        self.assertEqual('linear', self.linear.create().type())
        self.linear.slope = 17
        self.assertEqual(0.0, self.linear.create().slope)
        return


    def test_type(self):
        """check PDFBaseline.type
        """
        # this is a virtual method in the base class
        self.assertRaises(RuntimeError, PDFBaseline().type)
        self.assertEqual('linear', self.linear.type())
        self.assertEqual('zero', self.zero.type())
        self.assertTrue(type(self.linear) is LinearBaseline)
        self.assertTrue(type(self.zero) is ZeroBaseline)
        return


    def test_createByType(self):
        """check PDFBaseline.createByType()
        """
        self.assertRaises(ValueError, PDFBaseline.createByType, 'notregistered')
        return


    def test_getRegisteredTypes(self):
        """check PDFBaseline.getRegisteredTypes
        """
        regtypes = PDFBaseline.getRegisteredTypes()
        self.assertTrue(2 <= len(regtypes))
        self.assertTrue('linear' in regtypes)
        self.assertTrue('zero' in regtypes)
        return


    def test_pickling(self):
        '''check pickling and unpickling of PDFBaseline.
        '''
        linear = self.linear
        linear.slope = 11
        linear2 = cPickle.loads(cPickle.dumps(linear))
        self.assertEqual('linear', linear2.type())
        self.assertEqual(11, linear2.slope)
        self.assertEqual(11, linear2._getDoubleAttr('slope'))
        return


    def test_makePDFBaseline(self):
        '''check the makePDFBaseline wrapper.
        '''
        pbl = makePDFBaseline('parabolabaseline',
                parabola_baseline, a=1, b=2, c=3)
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
        pbl3 = PDFBaseline.createByType('parabolabaseline')
        self.assertEqual(1, pbl3.a)
        self.assertEqual(2, pbl3.b)
        self.assertEqual(3, pbl3.c)
        return

# End of class TestPDFBaseline

# function for wrapping by makePDFBaseline

def parabola_baseline(x, a, b, c):
    return a * x**2 + b * x + c

if __name__ == '__main__':
    unittest.main()

# End of file
