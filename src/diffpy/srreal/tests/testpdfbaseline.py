#!/usr/bin/env python

"""Unit tests for the PDFBaseline class from diffpy.srreal.pdfcalculator."""


import pickle
import unittest

import numpy

from diffpy.srreal.pdfbaseline import (
    LinearBaseline,
    PDFBaseline,
    ZeroBaseline,
    makePDFBaseline,
)
from diffpy.srreal.pdfcalculator import PDFCalculator
from diffpy.srreal.tests.testutils import pickle_with_attr

# ----------------------------------------------------------------------------


class TestPDFBaseline(unittest.TestCase):

    def setUp(self):
        self.linear = PDFBaseline.createByType("linear")
        self.zero = PDFBaseline.createByType("zero")
        return

    def tearDown(self):
        for tp in PDFBaseline.getRegisteredTypes():
            PDFBaseline._deregisterType(tp)
        self.linear._registerThisType()
        self.zero._registerThisType()
        return

    def test___init__(self):
        """Check PDFBaseline.__init__()"""
        self.assertEqual(0.0, self.linear.slope)
        self.linear._setDoubleAttr("slope", 2.0)
        self.assertEqual(2.0, self.linear.slope)
        return

    def test___call__(self):
        """Check PDFBaseline.__call__()"""
        # this is a virtual method in the base class
        self.assertRaises(RuntimeError, PDFBaseline().__call__, 37)
        self.assertEqual(0.0, self.zero(10))
        self.assertEqual(0.0, self.zero(3.45))
        self.assertEqual(0.0, self.linear(3.45))
        self.assertEqual(0.0, self.linear(345))
        self.linear.slope = -2
        self.assertEqual(-7.0, self.linear(3.5))
        self.assertEqual(-2.0, self.linear._getDoubleAttr("slope"))
        x = numpy.arange(0, 10.001, 0.1)
        xb = numpy.array([(0.0, xi) for xi in x])[:, 1]
        self.assertTrue(xb.strides > x.strides)
        self.assertTrue(numpy.array_equal(-2 * x, self.linear(x)))
        self.assertTrue(numpy.array_equal(-2 * x, self.linear(xb)))
        return

    def test_clone(self):
        """Check PDFBaseline.clone."""
        # this is a virtual method in the base class
        self.assertRaises(RuntimeError, PDFBaseline().clone)
        self.linear.slope = 17
        bl2 = self.linear.clone()
        self.assertEqual("linear", bl2.type())
        self.assertEqual(17.0, bl2.slope)
        self.assertEqual(17.0, bl2._getDoubleAttr("slope"))
        return

    def test_create(self):
        """Check PDFBaseline.create."""
        # this is a virtual method in the base class
        self.assertRaises(RuntimeError, PDFBaseline().create)
        self.assertEqual("zero", self.zero.create().type())
        self.assertEqual("linear", self.linear.create().type())
        self.linear.slope = 17
        self.assertEqual(0.0, self.linear.create().slope)
        return

    def test_type(self):
        """Check PDFBaseline.type."""
        # this is a virtual method in the base class
        self.assertRaises(RuntimeError, PDFBaseline().type)
        self.assertEqual("linear", self.linear.type())
        self.assertEqual("zero", self.zero.type())
        self.assertTrue(type(self.linear) is LinearBaseline)
        self.assertTrue(type(self.zero) is ZeroBaseline)
        return

    def test__aliasType(self):
        """Check PDFBaseline._aliasType."""
        self.assertRaises(ValueError, PDFBaseline.createByType, "alias")
        self.assertRaises(RuntimeError, PDFBaseline._aliasType, "invalid", "alias")
        self.assertRaises(RuntimeError, PDFBaseline._aliasType, "linear", "zero")
        PDFBaseline._aliasType("linear", "alias")
        bl = PDFBaseline.createByType("alias")
        self.assertEqual("linear", bl.type())
        self.assertTrue(isinstance(bl, LinearBaseline))
        # second registration is a no-op
        PDFBaseline._aliasType("linear", "alias")
        bl1 = PDFBaseline.createByType("alias")
        self.assertTrue(isinstance(bl1, LinearBaseline))
        # no other type can be aliased to the existing name.
        self.assertRaises(RuntimeError, PDFBaseline._aliasType, "zero", "alias")
        return

    def test__deregisterType(self):
        """Check PDFBaseline._deregisterType."""
        self.assertEqual(0, PDFBaseline._deregisterType("nonexistent"))
        PDFBaseline._aliasType("linear", "alias")
        self.assertEqual(2, PDFBaseline._deregisterType("alias"))
        self.assertFalse("linear" in PDFBaseline.getRegisteredTypes())
        self.assertEqual(0, PDFBaseline._deregisterType("alias"))
        return

    def test_createByType(self):
        """Check PDFBaseline.createByType()"""
        self.assertRaises(ValueError, PDFBaseline.createByType, "notregistered")
        return

    def test_isRegisteredType(self):
        """Check PDFBaseline.isRegisteredType()"""
        self.assertTrue(PDFBaseline.isRegisteredType("linear"))
        self.assertFalse(PDFBaseline.isRegisteredType("nonexistent"))
        PDFBaseline._deregisterType("linear")
        self.assertFalse(PDFBaseline.isRegisteredType("linear"))
        return

    def test_getAliasedTypes(self):
        """Check PDFBaseline.getAliasedTypes()"""
        self.assertEqual({}, PDFBaseline.getAliasedTypes())
        PDFBaseline._aliasType("linear", "foo")
        PDFBaseline._aliasType("linear", "bar")
        PDFBaseline._aliasType("linear", "linear")
        PDFBaseline._aliasType("bar", "foo")
        self.assertEqual(
            {"bar": "linear", "foo": "linear"}, PDFBaseline.getAliasedTypes()
        )
        return

    def test_getRegisteredTypes(self):
        """Check PDFBaseline.getRegisteredTypes."""
        regtypes = PDFBaseline.getRegisteredTypes()
        self.assertTrue(2 <= len(regtypes))
        self.assertTrue("linear" in regtypes)
        self.assertTrue("zero" in regtypes)
        return

    def test_pickling(self):
        """Check pickling and unpickling of PDFBaseline."""
        linear = self.linear
        linear.slope = 11
        linear2 = pickle.loads(pickle.dumps(linear))
        self.assertEqual("linear", linear2.type())
        self.assertEqual(11, linear2.slope)
        self.assertEqual(11, linear2._getDoubleAttr("slope"))
        self.assertRaises(RuntimeError, pickle_with_attr, linear, foo="bar")
        self.assertRaises(RuntimeError, pickle_with_attr, self.zero, foo="bar")
        return

    def test_makePDFBaseline(self):
        """Check the makePDFBaseline wrapper."""
        pbl = makePDFBaseline("parabolabaseline", parabola_baseline, a=1, b=2, c=3)
        self.assertEqual(3, pbl(0))
        self.assertEqual(6, pbl(1))
        self.assertEqual(11, pbl(2))
        pbl.b = 0
        self.assertEqual([7, 3, 28], [pbl(x) for x in [-2, 0, 5]])
        pbl2 = pbl.clone()
        self.assertEqual(1, pbl2.a)
        self.assertEqual(0, pbl2.b)
        self.assertEqual(3, pbl2.c)
        self.assertEqual([7, 3, 28], [pbl2(x) for x in [-2, 0, 5]])
        pbl3 = PDFBaseline.createByType("parabolabaseline")
        self.assertEqual(1, pbl3.a)
        self.assertEqual(2, pbl3.b)
        self.assertEqual(3, pbl3.c)
        pbl.foo = "bar"
        pbl4 = pickle.loads(pickle.dumps(pbl))
        self.assertEqual([7, 3, 28], [pbl4(x) for x in [-2, 0, 5]])
        self.assertEqual("bar", pbl4.foo)
        # fail if this baseline type already exists.
        self.assertRaises(
            RuntimeError, makePDFBaseline, "linear", parabola_baseline, a=1, b=2, c=3
        )
        self.assertRaises(
            RuntimeError,
            makePDFBaseline,
            "parabolabaseline",
            parabola_baseline,
            a=1,
            b=2,
            c=3,
        )
        # check replacement of an existing type.
        makePDFBaseline("linear", parabola_baseline, replace=True, a=1, b=2, c=4)
        pbl4 = PDFBaseline.createByType("linear")
        self.assertEqual(set(("a", "b", "c")), pbl4._namesOfDoubleAttributes())
        self.assertEqual(4, pbl4.c)
        # check baseline with no attributes
        pbl5 = makePDFBaseline("myzero", lambda x: 0.0)
        self.assertEqual(0, pbl5(33))
        self.assertEqual(set(), pbl5._namesOfDoubleAttributes())
        return

    def test_picking_owned(self):
        """Verify pickling of PDFBaseline owned by PDF calculators."""
        pbl = makePDFBaseline("parabolabaseline", parabola_baseline, a=1, b=2, c=3)
        pbl.a = 7
        pbl.foobar = "asdf"
        pc = PDFCalculator()
        pc.baseline = pbl
        self.assertIs(pbl, pc.baseline)
        pc2 = pickle.loads(pickle.dumps(pc))
        pbl2 = pc2.baseline
        self.assertEqual(7, pbl2.a)
        self.assertEqual("asdf", pbl2.foobar)
        self.assertEqual("parabolabaseline", pbl2.type())
        return


# End of class TestPDFBaseline

# ----------------------------------------------------------------------------

# function for wrapping by makePDFBaseline


def parabola_baseline(x, a, b, c):
    return a * x**2 + b * x + c


if __name__ == "__main__":
    unittest.main()

# End of file
