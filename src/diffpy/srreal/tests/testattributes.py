#!/usr/bin/env python

"""Unit tests for the wrapped diffpy::Attributes
"""


import unittest
import weakref

from diffpy.srreal.attributes import Attributes
from diffpy.srreal.pairquantity import PairQuantity
from diffpy.srreal.pdfcalculator import PDFCalculator

##############################################################################
class TestAttributes(unittest.TestCase):

    def setUp(self):
        return


    def tearDown(self):
        return


    def test___setattr__(self):
        """check Attributes.__setattr__()
        """
        # normal attribute
        a = Attributes()
        a.x = 45
        self.assertTrue("x" in a.__dict__)
        self.assertFalse("x" in a._namesOfDoubleAttributes())
        self.assertRaises(AttributeError, a._getDoubleAttr, "x")
        self.assertRaises(AttributeError, a._setDoubleAttr, "x", 13)
        del a.x
        a._registerDoubleAttribute("x")
        self.assertTrue("x" in a._namesOfDoubleAttributes())
        a.x = 27
        self.assertEqual(27, a._getDoubleAttr("x"))
        return


    def test___getattr__(self):
        """check Attributes.__getattr__()
        """
        a = Attributes()
        self.assertRaises(AttributeError, getattr, a, 'invalid')
        a.x = 11
        self.assertEqual(11, a.x)
        pdfc = PDFCalculator()
        pdfc._setDoubleAttr('rmax', 12.34)
        self.assertEqual(12.34, pdfc.rmax)
        return


    def test_garbage_collection(self):
        """check garbage collection for Python defined Attributes
        """
        # check if attributes are garbage collected
        pq = PairQuantity()
        wpq = weakref.ref(pq)
        self.assertFalse(wpq() is None)
        pq._registerDoubleAttribute('foo')
        pq.foo = 45
        self.assertEqual(45, pq._getDoubleAttr('foo'))
        del pq
        self.assertTrue(wpq() is None)
        return


    def test__getDoubleAttr(self):
        """check Attributes._getDoubleAttr()
        """
        pdfc = PDFCalculator()
        pdfc.foo = 11
        self.assertRaises(AttributeError, pdfc._getDoubleAttr, 'foo')
        pdfc._registerDoubleAttribute('foo')
        self.assertEqual(11, pdfc._getDoubleAttr('foo'))
        pdfc.rmax = 22
        self.assertEqual(22, pdfc._getDoubleAttr('rmax'))
        setattr(pdfc, 'rmax', 23)
        self.assertEqual(23, pdfc._getDoubleAttr('rmax'))
        self.assertRaises(Exception, setattr, pdfc, 'rmax', 'xxx')
        return


    def test__hasDoubleAttr(self):
        """check Attributes._hasDoubleAttr()
        """
        a = Attributes()
        a.foo = 45
        self.assertFalse(a._hasDoubleAttr('foo'))
        a._registerDoubleAttribute('foo')
        self.assertTrue(a._hasDoubleAttr('foo'))
        return


    def test__namesOfDoubleAttributes(self):
        """check Attributes._namesOfDoubleAttributes()
        """
        a = Attributes()
        self.assertEqual(0, len(a._namesOfDoubleAttributes()))
        pq = PairQuantity()
        self.assertNotEqual(0, len(pq._namesOfDoubleAttributes()))
        self.assertFalse('bar' in pq._namesOfDoubleAttributes())
        pq._registerDoubleAttribute('bar')
        self.assertTrue('bar' in pq._namesOfDoubleAttributes())
        return


    def test__namesOfWritableDoubleAttributes(self):
        """check Attributes._namesOfDoubleAttributes()
        """
        a = Attributes()
        self.assertEqual(0, len(a._namesOfDoubleAttributes()))
        a._registerDoubleAttribute('bar', lambda obj: 13)
        self.assertEqual(13, a._getDoubleAttr('bar'))
        self.assertEqual(13, a.bar)
        self.assertEqual(1, len(a._namesOfDoubleAttributes()))
        self.assertEqual(0, len(a._namesOfWritableDoubleAttributes()))
        pdfc = PDFCalculator()
        self.assertTrue('extendedrmin' in pdfc._namesOfDoubleAttributes())
        self.assertTrue('extendedrmax' in pdfc._namesOfDoubleAttributes())
        nwda = pdfc._namesOfWritableDoubleAttributes()
        self.assertFalse('extendedrmin' in nwda)
        self.assertFalse('extendedrmax' in nwda)
        return


    def test__registerDoubleAttribute(self):
        """check Attributes._registerDoubleAttribute()
        """
        d = {'g_called' : False,  's_called' : False, 'value' : 0}
        def g(obj):
            d['g_called'] = True
            return d['value']
        def s(obj, value):
            d['s_called'] = True
            d['value'] = value
            return
        a = Attributes()
        wa = weakref.ref(a)
        self.assertFalse(wa() is None)
        a._registerDoubleAttribute('a1', g, s)
        self.assertFalse('a1' in a.__dict__)
        self.assertFalse(d['g_called'])
        self.assertFalse(d['s_called'])
        self.assertEqual(0, a.a1)
        self.assertTrue(d['g_called'])
        self.assertFalse(d['s_called'])
        a.a1 = 47
        self.assertTrue(d['s_called'])
        self.assertEqual(47, d['value'])
        self.assertTrue(hasattr(a, 'a1'))
        a._registerDoubleAttribute('a1readonly', g)
        self.assertEqual(47, a.a1readonly)
        self.assertTrue(hasattr(a, 'a1readonly'))
        self.assertRaises(AttributeError, a._setDoubleAttr, 'a1readonly', 7)
        self.assertRaises(AttributeError, setattr, a, 'a1readonly', 5)
        self.assertEqual(47, a.a1readonly)
        a.a1 = 9
        self.assertEqual(9, a.a1readonly)
        del a
        self.assertTrue(wa() is None)
        return


    def test__setDoubleAttr(self):
        """check Attributes._setDoubleAttr()
        """
        pdfc = PDFCalculator()
        pdfc._setDoubleAttr('scale', 1.23)
        self.assertFalse('scale' in pdfc.__dict__)
        self.assertEqual(1.23, pdfc.scale)
        return

# End of class TestAttributes

if __name__ == '__main__':
    unittest.main()

# End of file
