#!/usr/bin/env python

"""Unit tests for the AtomRadiiTable class.
"""

# version
__id__ = '$Id$'

import os
import unittest
import cPickle
from diffpy.srreal.atomradiitable import AtomRadiiTable, CovalentRadiiTable

##############################################################################
class TestAtomRadiiTable(unittest.TestCase):

    def setUp(self):
        self.rtb = AtomRadiiTable()
        return

    def tearDown(self):
        return

    def test_pickling(self):
        '''check pickling and unpickling of AtomRadiiTable.
        '''
        rtb1 = cPickle.loads(cPickle.dumps(self.rtb))
        self.assertTrue(type(rtb1) is AtomRadiiTable)
        self.assertEqual({}, rtb1.getAllCustom())
        self.rtb.setCustom('Na', 1.3)
        self.rtb.foobar = 'foo'
        rtb2 = cPickle.loads(cPickle.dumps(self.rtb))
        self.assertEqual({'Na' : 1.3}, rtb2.getAllCustom())
        self.assertEqual('foo', rtb2.foobar)
        return

    def test__tableLookup(self):
        """check AtomRadiiTable._tableLookup()
        """
        self.assertEqual(0.0, self.rtb._tableLookup('anything'))
        return

    def test_fromString(self):
        """check AtomRadiiTable.fromString()
        """
        self.rtb.fromString('H:0.33, B:0.42')
        self.assertEqual({'H' : 0.33, 'B' : 0.42}, self.rtb.getAllCustom())
        self.assertRaises(ValueError, self.rtb.fromString,
                'C:2.3, U:asdf')
        self.assertEqual({'H' : 0.33, 'B' : 0.42}, self.rtb.getAllCustom())
        self.rtb.fromString('C:2.3,,,')
        self.assertEqual(3, len(self.rtb.getAllCustom()))
        self.assertEqual(2.3, self.rtb.lookup('C'))
        return

    def test_getAllCustom(self):
        """check AtomRadiiTable.getAllCustom()
        """
        self.assertEqual({}, self.rtb.getAllCustom())
        return

    def test_lookup(self):
        """check AtomRadiiTable.lookup()
        """
        self.assertEqual(0.0, self.rtb.lookup('C'))
        self.rtb.setCustom('C', 1.23)
        self.assertEqual(1.23, self.rtb.lookup('C'))
        return

    def test_resetCustom(self):
        """check AtomRadiiTable.resetCustom()
        """
        self.rtb.setCustom('C', 1.23)
        self.assertTrue(self.rtb.getAllCustom())
        self.rtb.resetAll()
        self.assertFalse(self.rtb.getAllCustom())
        return

    def test_setCustom(self):
        """check AtomRadiiTable.setCustom()
        """
        self.rtb.setCustom('C', 1.23)
        self.assertEqual(1.23, self.rtb.lookup('C'))
        self.rtb.setCustom('C', 3.3)
        self.assertEqual(3.3, self.rtb.lookup('C'))
        return

    def test_toString(self):
        """check AtomRadiiTable.toString()
        """
        rtb = self.rtb
        self.assertEqual('', rtb.toString())
        self.assertEqual('', rtb.toString('; '))
        rtb.fromString('C :  1.5,  B:2.0')
        self.assertEqual('B:2,C:1.5', rtb.toString())
        self.assertEqual('B:2; C:1.5', rtb.toString('; '))
        return

# End of class TestAtomRadiiTable

##############################################################################
class TestCovalentRadiiTable(unittest.TestCase):

    def setUp(self):
        self.rtb = CovalentRadiiTable()
        return

    def tearDown(self):
        return

    def test_pickling(self):
        '''check pickling and unpickling of CovalentRadiiTable.
        '''
        rtb1 = cPickle.loads(cPickle.dumps(self.rtb))
        self.assertTrue(type(rtb1) is CovalentRadiiTable)
        self.assertEqual({}, rtb1.getAllCustom())
        self.rtb.setCustom('Na', 1.3)
        self.rtb.foobar = 'foo'
        rtb2 = cPickle.loads(cPickle.dumps(self.rtb))
        self.assertEqual({'Na' : 1.3}, rtb2.getAllCustom())
        self.assertEqual('foo', rtb2.foobar)
        return

    def test__tableLookup(self):
        """check CovalentRadiiTable._tableLookup()
        """
        self.assertEqual(1.22, self.rtb._tableLookup('Ga'))
        return

    def test_fromString(self):
        """check CovalentRadiiTable.fromString()
        """
        self.rtb.fromString('Ga:2.22')
        self.assertEqual(2.22, self.rtb.lookup('Ga'))
        return

    def test_getAllCustom(self):
        """check CovalentRadiiTable.getAllCustom()
        """
        self.assertEqual({}, self.rtb.getAllCustom())
        return

    def test_lookup(self):
        """check CovalentRadiiTable.lookup()
        """
        self.assertEqual(1.22, self.rtb.lookup('Ga'))
        self.rtb.fromString('Ga:2.22')
        self.assertEqual(2.22, self.rtb.lookup('Ga'))
        return

    def test_resetCustom(self):
        """check CovalentRadiiTable.resetCustom()
        """
        self.rtb.fromString('B:2.33, Ga:2.22')
        self.rtb.resetCustom('B')
        self.rtb.resetCustom('nada')
        self.assertEqual(1, len(self.rtb.getAllCustom()))
        self.assertEqual(0.84, self.rtb.lookup('B'))
        self.assertEqual(2.22, self.rtb.lookup('Ga'))
        return

    def test_setCustom(self):
        """check CovalentRadiiTable.setCustom()
        """
        self.assertEqual(0.84, self.rtb.lookup('B'))
        self.rtb.setCustom('B', 0.9)
        self.assertEqual(0.9, self.rtb.lookup('B'))
        return

    def test_toString(self):
        """check CovalentRadiiTable.toString()
        """
        self.assertEqual('', self.rtb.toString(';---'))
        return

# End of class TestCovalentRadiiTable

if __name__ == '__main__':
    unittest.main()

# End of file
