#!/usr/bin/env python
##############################################################################
#
# diffpy.srreal     by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2010 The Trustees of Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Pavol Juhas
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE_DANSE.txt for license information.
#
##############################################################################


"""\
Classes for configuring PDF baseline:
    PDFBaseline, ZeroBaseline, LinearBaseline
"""


# exported items
__all__ = '''
    PDFBaseline makePDFBaseline
    ZeroBaseline
    LinearBaseline
    '''.split()

from diffpy.srreal import _final_imports
from diffpy.srreal.srreal_ext import PDFBaseline
from diffpy.srreal.srreal_ext import ZeroBaseline, LinearBaseline
from diffpy.srreal.wraputils import propertyFromExtDoubleAttr
from diffpy.srreal.wraputils import _pickle_getstate, _pickle_setstate

# class PDFBaseline ----------------------------------------------------------

# pickling support

def _baseline_create(s):
    from diffpy.srreal.srreal_ext import _PDFBaseline_fromstring
    return _PDFBaseline_fromstring(s)

def _baseline_reduce(self):
    from diffpy.srreal.srreal_ext import _PDFBaseline_tostring
    args = (_PDFBaseline_tostring(self),)
    rv = (_baseline_create, args)
    return rv

def _baseline_reduce_with_state(self):
    rv = _baseline_reduce(self) + (self.__getstate__(),)
    return rv

# inject pickle methods

PDFBaseline.__reduce__ = _baseline_reduce_with_state
PDFBaseline.__getstate__ = _pickle_getstate
PDFBaseline.__setstate__ = _pickle_setstate

ZeroBaseline.__reduce__ = _baseline_reduce
LinearBaseline.__reduce__ = _baseline_reduce

# attribute wrapper

LinearBaseline.slope = propertyFromExtDoubleAttr('slope',
    '''Slope of an unscaled linear baseline.  For crystal structures it
    is preset to (-4 * pi * rho0).''')

# Python functions wrapper

def makePDFBaseline(name, fnc, replace=False, **dbattrs):
    '''Helper function for registering Python function as a PDFBaseline.
    This is required for using Python function as PDFCalculator.baseline.

    name     -- unique string name for registering Python function in the
                global registry of PDFBaseline types.  This will be the
                string identifier for the createByType factory.
    fnc      -- Python function of a floating point argument and optional
                float parameters.  The parameters need to be registered as
                double attributes in the functor class.  The function fnc
                must be picklable and it must return a float.
    replace  -- when set replace any PDFBaseline type already registered
                under the name.  Otherwise raise RuntimeError when the
                name is taken.
    dbattrs  -- optional float parameters of the wrapped function.
                These will be registered as double attributes in the
                functor class.  The wrapped function must be callable as
                fnc(x, **dbattrs).  Make sure to pick attribute names that
                do not conflict with other PDFCalculator attributes.

    Return an instance of the new PDFBaseline class.

    Example:

        # Python baseline function
        def fshiftedline(x, aline, bline):
            return aline * x + bline
        # wrap it as a PDFBaseline and register as a "shiftedline" type
        makePDFBaseline("shiftedline", fshiftedline, aline=-1, bline=0)
        baseline = PDFBaseline.createByType("shiftedline")
        print map(baseline, range(5))
        # use it in PDFCalculator
        pdfc = PDFCalculator()
        pdfc.baseline = baseline
        # or pdfc.baseline = "shiftedline"
    '''
    from diffpy.srreal.wraputils import _wrapAsRegisteredUnaryFunction
    rv = _wrapAsRegisteredUnaryFunction(PDFBaseline, name, fnc,
                                        replace=replace, **dbattrs)
    return rv

# Import delayed tweaks of the extension classes.

_final_imports.import_now()
del _final_imports

# End of file
