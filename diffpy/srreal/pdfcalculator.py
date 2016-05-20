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
Top-level classes for PDF calculation:
    DebyePDFCalculator -- simulate PDF by evaluating Debye sum in Q-space
    PDFCalculator      -- calculate PDF by peak summation in real space
"""


# exported items
__all__ = '''
    DebyePDFCalculator PDFCalculator
    fftftog fftgtof
    '''.split()

from diffpy.srreal.srreal_ext import DebyePDFCalculator
from diffpy.srreal.srreal_ext import PDFCalculator
from diffpy.srreal.srreal_ext import fftftog, fftgtof
from diffpy.srreal.wraputils import propertyFromExtDoubleAttr
from diffpy.srreal.wraputils import setattrFromKeywordArguments

# silence the pyflakes syntax checker
assert all((fftftog, fftgtof))

# imports for backward compatibility
from diffpy.srreal.pdfbaseline import (PDFBaseline, makePDFBaseline,
        ZeroBaseline, LinearBaseline)
from diffpy.srreal.pdfenvelope import (PDFEnvelope, makePDFEnvelope,
        QResolutionEnvelope, ScaleEnvelope,
        SphericalShapeEnvelope, StepCutEnvelope)
from diffpy.srreal.peakprofile import PeakProfile
from diffpy.srreal.peakwidthmodel import (PeakWidthModel,
        ConstantPeakWidth, DebyeWallerPeakWidth, JeongPeakWidth)

# silence the pyflakes syntax checker
assert all((PDFBaseline, makePDFBaseline, ZeroBaseline, LinearBaseline,
            PDFEnvelope, makePDFEnvelope, QResolutionEnvelope, ScaleEnvelope,
            SphericalShapeEnvelope, StepCutEnvelope, PeakProfile,
            PeakWidthModel, ConstantPeakWidth, DebyeWallerPeakWidth,
            JeongPeakWidth))

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
        '''PDF Gaussian dampening factor due to limited Q-resolution.
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

PDFCalculator.qstep = propertyFromExtDoubleAttr('qstep',
        '''Spacing in the Q-grid.  Q-values are at the multiples of qstep.
        The value is padded by rsteps so that PI/qstep > extendedrmax and
        PI/(qstep * rstep) is a power of 2.  Read-only.
        [PI/(padded extendedrmax) A]''')

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

# End of file
