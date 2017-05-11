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
Classes for configuring PDF scaling envelope:
    PDFEnvelope, ScaleEnvelope, QResolutionEnvelope,
    SphericalShapeEnvelope, StepCutEnvelope
"""


# exported items
__all__ = '''
   PDFEnvelope makePDFEnvelope
   QResolutionEnvelope
   ScaleEnvelope
   SphericalShapeEnvelope
   StepCutEnvelope
   '''.split()

from diffpy.srreal import _final_imports
from diffpy.srreal.srreal_ext import PDFEnvelope
from diffpy.srreal.srreal_ext import ScaleEnvelope, QResolutionEnvelope
from diffpy.srreal.srreal_ext import SphericalShapeEnvelope, StepCutEnvelope
from diffpy.srreal.wraputils import propertyFromExtDoubleAttr
from diffpy.srreal.wraputils import _pickle_getstate, _pickle_setstate

# class PDFEnvelope ----------------------------------------------------------

# pickling support

def _envelope_create(s):
    from diffpy.srreal.srreal_ext import _PDFEnvelope_fromstring
    return _PDFEnvelope_fromstring(s)

def _envelope_reduce(self):
    from diffpy.srreal.srreal_ext import _PDFEnvelope_tostring
    args = (_PDFEnvelope_tostring(self),)
    rv = (_envelope_create, args)
    return rv

def _envelope_reduce_with_state(self):
    rv = _envelope_reduce(self) + (self.__getstate__(),)
    return rv

# inject pickle methods

PDFEnvelope.__reduce__ = _envelope_reduce_with_state
PDFEnvelope.__getstate__ = _pickle_getstate
PDFEnvelope.__setstate__ = _pickle_setstate

QResolutionEnvelope.__reduce__ = _envelope_reduce
ScaleEnvelope.__reduce__ = _envelope_reduce
SphericalShapeEnvelope.__reduce__ = _envelope_reduce
StepCutEnvelope.__reduce__ = _envelope_reduce

# attribute wrappers

QResolutionEnvelope.qdamp = propertyFromExtDoubleAttr('qdamp',
    '''Dampening parameter in the Gaussian envelope function.
    ''')

ScaleEnvelope.scale = propertyFromExtDoubleAttr('scale',
    '''Overall scale for a uniform scaling envelope.
    ''')

SphericalShapeEnvelope.spdiameter = propertyFromExtDoubleAttr('spdiameter',
    '''Particle diameter in Angstroms for a spherical shape damping.
    ''')

StepCutEnvelope.stepcut = propertyFromExtDoubleAttr('stepcut',
    '''Cutoff for a step-function envelope.
    ''')

# Python functions wrapper

def makePDFEnvelope(name, fnc, replace=False, **dbattrs):
    '''Helper function for registering Python function as a PDFEnvelope.
    This is required for using Python function as PDFCalculator envelope.

    name     -- unique string name for registering Python function in the
                global registry of PDFEnvelope types.  This will be the
                string identifier for the createByType factory.
    fnc      -- Python function of a floating point argument and optional
                float parameters.  The parameters need to be registered as
                double attributes in the functor class.  The function fnc
                must be picklable and it must return a float.
    replace  -- when set replace any PDFEnvelope type already registered
                under the name.  Otherwise raise RuntimeError when the
                name is taken.
    dbattrs  -- optional float parameters of the wrapped function.
                These will be registered as double attributes in the
                functor class.  The wrapped function must be callable as
                fnc(x, **dbattrs).  Make sure to pick attribute names that
                do not conflict with other PDFCalculator attributes.

    Return an instance of the new PDFEnvelope class.

    Example:

        # Python envelope function
        def fexpdecay(x, expscale, exptail):
            from math import exp
            return expscale * exp(-x / exptail)
        # wrap it as a PDFEnvelope and register as a "expdecay" type
        makePDFEnvelope("expdecay", fexpdecay, expscale=5, exptail=4)
        envelope = PDFEnvelope.createByType("expdecay")
        print map(envelope, range(9))
        # use it in PDFCalculator
        pdfc = PDFCalculator()
        pdfc.addEnvelope(envelope)
        # or pdfc.addEnvelope("expdecay")
    '''
    from diffpy.srreal.wraputils import _wrapAsRegisteredUnaryFunction
    rv = _wrapAsRegisteredUnaryFunction(PDFEnvelope, name, fnc,
                                        replace=replace, **dbattrs)
    return rv

# Import delayed tweaks of the extension classes.

_final_imports.import_now()
del _final_imports

# End of file
