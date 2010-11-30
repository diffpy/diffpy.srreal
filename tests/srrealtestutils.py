#!/usr/bin/env python

"""Helper routines for running other unit tests.

TestCaseObjCrystOptional -- use this as a TestCase base class that
    disables unit tests when pyobjcryst is not installed.
"""

# version
__id__ = '$Id$'

import logging

try:
    import pyobjcryst
    from unittest import TestCase as TestCaseObjCrystOptional
except ImportError:
    TestCaseObjCrystOptional = object
    logging.warning('Cannot import pyobjcryst, pyobjcryst tests skipped.')


# End of file
