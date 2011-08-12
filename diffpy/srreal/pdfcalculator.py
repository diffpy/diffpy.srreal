#!/usr/bin/env python
##############################################################################
#
# diffpy.srreal     by DANSE Diffraction group
#                   Simon J. L. Billinge
#                   (c) 2010 Trustees of the Columbia University
#                   in the City of New York.  All rights reserved.
#
# File coded by:    Pavol Juhas
#
# See AUTHORS.txt for a list of people who contributed.
# See LICENSE.txt for license information.
#
##############################################################################


"""\
class DebyePDFCalculator -- PDF calculator that uses Debye Formula
class PDFCalculator      -- PDF calculator in real space
"""

# module version
__id__ = "$Id$"

# exported items
__all__ = '''DebyePDFCalculator PDFCalculator
    PDFBaseline PDFEnvelope PeakProfile PeakWidthModel
    makePDFBaseline makePDFEnvelope fftftog fftgtof
    '''.split()

from diffpy.srreal.srreal_ext import DebyePDFCalculator
from diffpy.srreal.srreal_ext import PDFCalculator
from diffpy.srreal.srreal_ext import fftftog, fftgtof
from diffpy.srreal.srreal_ext import PDFBaseline, PDFEnvelope
from diffpy.srreal.srreal_ext import PeakProfile, PeakWidthModel
from diffpy.srreal.wraputils import propertyFromExtDoubleAttr
from diffpy.srreal.wraputils import setattrFromKeywordArguments

# ----------------------------------------------------------------------------

def _defineCommonInterface(cls):

    '''This function defines shared properties of PDF calculator classes.
    '''

    cls.scale = propertyFromExtDoubleAttr('scale',
        '''Scale factor of the calculated PDF.  Active for ScaleEnvelope.
        [1.0 unitless]''')

    cls.delta1 = propertyFromExtDoubleAttr('delta1',
        '''Coefficient for (1/r) contribution to the peak sharpening.
        Active for JeongPeakWidth model.
        [0 A]''')

    cls.delta2 = propertyFromExtDoubleAttr('delta2',
        '''Coefficient for (1/r**2) contribution to the peak sharpening.
        Active for JeongPeakWidth model.
        [0 A**2]''')

    cls.qdamp = propertyFromExtDoubleAttr('qdamp',
        '''PDF Gaussian dampening envelope due to limited Q-resolution.
        Not applied when equal to zero.  Active for QResolutionEnvelope.
        [0 1/A]''')

    cls.qbroad = propertyFromExtDoubleAttr('qbroad',
        '''PDF peak broadening from increased intensity noise at high Q.
        Not applied when equal zero.  Active for JeongPeakWidth model.
        [0 1/A]''')

    cls.extendedrmin = propertyFromExtDoubleAttr('extendedrmin',
        '''Low boundary of the extended r-range, read-only.
        [A]''')

    cls.extendedrmax = propertyFromExtDoubleAttr('extendedrmax',
        '''Upper boundary of the extended r-range, read-only.
        [A]''')

    cls.maxextension = propertyFromExtDoubleAttr('maxextension',
        '''Maximum extension of the r-range that accounts for contributions
        from the out of range peaks.
        [10 A]''')

    cls.rmin = propertyFromExtDoubleAttr('rmin',
        '''Lower bound of the r-grid for PDF calculation.
        [0 A]''')

    cls.rmax = propertyFromExtDoubleAttr('rmax',
        '''Upper bound of the r-grid for PDF calculation.
        [10 A]''')

    cls.rstep = propertyFromExtDoubleAttr('rstep',
        '''Spacing in the calculated r-grid.  r-values are at the
        multiples of rstep.
        [0.01 A]''')

    def _call_kwargs(self, structure=None, **kwargs):
        '''Calculate PDF for the given structure as an (r, G) tuple.
        Keyword arguments can be used to configure calculator attributes,
        these override any properties that may be passed from the structure,
        such as spdiameter.

        structure    -- a structure object to be evaluated.  Reuse the last
                        structure when None.
        kwargs       -- optional parameter settings for this calculator

        Example:    pdfcalc(structure, qmax=20, spdiameter=15)

        Return a tuple of (r, G) numpy arrays.
        '''
        setattrFromKeywordArguments(self, **kwargs)
        self.eval(structure)
        # apply kwargs again if structure contained any attribute
        # that may affect the result.
        setattrFromKeywordArguments(self, **kwargs)
        rv = (self.rgrid, self.pdf)
        return rv
    cls.__call__ = _call_kwargs

# _defineCommonInterface

# class DebyePDFCalculator ---------------------------------------------------

# shared interface of the PDF calculator classes

_defineCommonInterface(DebyePDFCalculator)

# Property wrappers to double attributes of the C++ DebyePDFCalculator

DebyePDFCalculator.debyeprecision = propertyFromExtDoubleAttr('debyeprecision',
        '''Cutoff amplitude for the sine contributions to the F(Q).
        [1e-6 unitless]''')

DebyePDFCalculator.qmin = propertyFromExtDoubleAttr('qmin',
        '''Lower bound of the Q-grid for the calculated F(Q).
        Affects the shape envelope.
        [0 1/A]
        ''')

DebyePDFCalculator.qmax = propertyFromExtDoubleAttr('qmax',
        '''Upper bound of the Q-grid for the calculated F(Q).
        Affects the termination ripples.
        [25 1/A]
        ''')

DebyePDFCalculator.qstep = propertyFromExtDoubleAttr('qstep',
        '''Spacing in the Q-grid.  Q-values are at the multiples of qstep.
        [PI/extendedrmax A] unless user overridden.
        See also setOptimumQstep, isOptimumQstep.''')

# method overrides to support optional keyword arguments

def _init_kwargs0(self, **kwargs):
    '''Create a new instance of the DebyePDFCalculator.
    Keyword arguments can be used to configure the calculator properties,
    for example:

    dpc = DebyePDFCalculator(qmax=20, rmin=7, rmax=15)

    Raise ValueError for invalid keyword argument.
    '''
    DebyePDFCalculator.__boostpython__init(self)
    setattrFromKeywordArguments(self, **kwargs)
    return

DebyePDFCalculator.__boostpython__init = DebyePDFCalculator.__init__
DebyePDFCalculator.__init__ = _init_kwargs0

# End of class DebyePDFCalculator

# PDFCalculator --------------------------------------------------------------

# shared interface of the PDF calculator classes

_defineCommonInterface(PDFCalculator)

# Property wrappers to double attributes of the C++ PDFCalculator

PDFCalculator.peakprecision = propertyFromExtDoubleAttr('peakprecision',
        '''Cutoff amplitude of the peak tail relative to the peak maximum.
        [3.33e-6 unitless]''')

PDFCalculator.qmin = propertyFromExtDoubleAttr('qmin',
        '''Lower bound of the experimental Q-range used.
        Affects the shape envelope.
        [0 1/A]''')

PDFCalculator.qmax = propertyFromExtDoubleAttr('qmax',
        '''Upper bound of the experimental Q-range used.
        Affects the termination ripples.  Not used when zero.
        [0 1/A]''')

PDFCalculator.slope = propertyFromExtDoubleAttr('slope',
        '''Slope of the linear PDF background.  Assigned according to
        number density of the evaluated structure at each PDF calculation.
        Active for LinearBaseline.
        [-4*pi*numdensity unitless]''')

PDFCalculator.spdiameter = propertyFromExtDoubleAttr('spdiameter',
        '''Spherical particle diameter for PDF shape damping correction.
        Not used when zero.  Active for SphericalShapeEnvelope.
        [0 A]''')

PDFCalculator.stepcut = propertyFromExtDoubleAttr('stepcut',
        '''r-boundary for a step cutoff of the calculated PDF.
        Not used when negative or zero.  Active for StepCutEnvelope.
        Not used when zero.  Active for StepCutEnvelope.
        [0 A]''')

# method overrides to support optional keyword arguments

def _init_kwargs1(self, **kwargs):
    '''Create a new instance of PDFCalculator.
    Keyword arguments can be used to configure the calculator properties,
    for example:

    pc = PDFCalculator(qmax=20, rmin=7, rmax=15)

    Raise ValueError for invalid keyword argument.
    '''
    PDFCalculator.__boostpython__init(self)
    setattrFromKeywordArguments(self, **kwargs)
    return

PDFCalculator.__boostpython__init = PDFCalculator.__init__
PDFCalculator.__init__ = _init_kwargs1

# End of class PDFCalculator

# class PDFBaseline ----------------------------------------------------------

# pickling support

def _baseline_getstate(self):
    state = (self.__dict__, )
    return state

def _baseline_setstate(self, state):
    if len(state) != 1:
        emsg = ("expected 1-item tuple in call to __setstate__, got " +
                repr(state))
        raise ValueError(emsg)
    self.__dict__.update(state[0])
    return

def _baseline_reduce(self):
    from diffpy.srreal.srreal_ext import _PDFBaseline_tostring
    args = (_PDFBaseline_tostring(self),)
    rv = (_baseline_create, args, self.__getstate__())
    return rv

def _baseline_create(s):
    from diffpy.srreal.srreal_ext import _PDFBaseline_fromstring
    return _PDFBaseline_fromstring(s)

# inject pickle methods

PDFBaseline.__getstate__ = _baseline_getstate
PDFBaseline.__setstate__ = _baseline_setstate
PDFBaseline.__reduce__ = _baseline_reduce

# Python functions wrapper

def makePDFBaseline(name, fnc, **dbattrs):
    '''Helper function for registering Python function as a PDFBaseline.
    This is required for using Python function as PDFCalculator.baseline.

    name     -- unique string name for registering Python function in the
                global registry of PDFBaseline types.  This will be the
                string identifier for the createByType factory.
    fnc      -- Python function of a floating point argument and optional
                float parameters.  The parameters need to be registered as
                double attributes in the functor class.  The function fnc
                must be picklable and it must return a float.
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
        # or pdfc.setBaselineByType("shiftedline")
    '''
    from diffpy.srreal.wraputils import _wrapAsRegisteredUnaryFunction
    return _wrapAsRegisteredUnaryFunction(PDFBaseline, name, fnc, **dbattrs)

# class PDFEnvelope ----------------------------------------------------------

# pickling support

def _envelope_getstate(self):
    state = (self.__dict__, )
    return state

def _envelope_setstate(self, state):
    if len(state) != 1:
        emsg = ("expected 1-item tuple in call to __setstate__, got " +
                repr(state))
        raise ValueError(emsg)
    self.__dict__.update(state[0])
    return

def _envelope_reduce(self):
    from diffpy.srreal.srreal_ext import _PDFEnvelope_tostring
    args = (_PDFEnvelope_tostring(self),)
    rv = (_envelope_create, args, self.__getstate__())
    return rv

def _envelope_create(s):
    from diffpy.srreal.srreal_ext import _PDFEnvelope_fromstring
    return _PDFEnvelope_fromstring(s)

# inject pickle methods

PDFEnvelope.__getstate__ = _envelope_getstate
PDFEnvelope.__setstate__ = _envelope_setstate
PDFEnvelope.__reduce__ = _envelope_reduce

# Python functions wrapper

def makePDFEnvelope(name, fnc, **dbattrs):
    '''Helper function for registering Python function as a PDFEnvelope.
    This is required for using Python function as PDFCalculator envelope.

    name     -- unique string name for registering Python function in the
                global registry of PDFEnvelope types.  This will be the
                string identifier for the createByType factory.
    fnc      -- Python function of a floating point argument and optional
                float parameters.  The parameters need to be registered as
                double attributes in the functor class.  The function fnc
                must be picklable and it must return a float.
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
        # or pdfc.addEnvelopeByType("expdecay")
    '''
    from diffpy.srreal.wraputils import _wrapAsRegisteredUnaryFunction
    return _wrapAsRegisteredUnaryFunction(PDFEnvelope, name, fnc, **dbattrs)

# End of file
