#!/usr/bin/env python

"""Unit tests for the PDFBaseline class from diffpy.srreal.pdfcalculator
"""

# version
__id__ = '$Id$'

import os
import unittest
import cPickle

from diffpy.srreal.pdfcalculator import PDFBaseline

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
        linear.foo = "qwer"
        linear.slope = 11
        linear2 = cPickle.loads(cPickle.dumps(linear))
        self.assertEqual('linear', linear2.type())
        self.assertEqual(11, linear2.slope)
        self.assertEqual(11, linear2._getDoubleAttr('slope'))
        self.assertEqual("qwer", linear2.foo)
        return


# End of class TestBVSCalculator

if __name__ == '__main__':
    unittest.main()

# End of file
