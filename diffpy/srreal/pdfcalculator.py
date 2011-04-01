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
    PDFBaseline PDFEnvelope PeakProfile
    '''.split()

from diffpy.srreal.srreal_ext import DebyePDFCalculator_ext
from diffpy.srreal.srreal_ext import PDFCalculator_ext
from diffpy.srreal.srreal_ext import PDFBaseline, PDFEnvelope, PeakProfile
from diffpy.srreal.wraputils import propertyFromExtDoubleAttr
from diffpy.srreal.wraputils import setattrFromKeywordArguments

# ----------------------------------------------------------------------------

class PDFCalculatorInterface(object):

    '''Base class for shared properties and methods.
    '''

    scale = propertyFromExtDoubleAttr('scale',
        '''Scale factor of the calculated PDF.  Active for ScaleEnvelope.
        [1.0 unitless]''')

    delta1 = propertyFromExtDoubleAttr('delta1',
        '''Coefficient for (1/r) contribution to the peak sharpening.
        Active for JeongPeakWidth model.
        [0 A]''')

    delta2 = propertyFromExtDoubleAttr('delta2',
        '''Coefficient for (1/r**2) contribution to the peak sharpening.
        Active for JeongPeakWidth model.
        [0 A**2]''')

    qdamp = propertyFromExtDoubleAttr('qdamp',
        '''PDF Gaussian dampening envelope due to limited Q-resolution.
        Not applied when equal to zero.  Active for QResolutionEnvelope.
        [0 1/A]''')

    qbroad = propertyFromExtDoubleAttr('qbroad',
        '''PDF peak broadening from increased intensity noise at high Q.
        Not applied when equal zero.  Active for JeongPeakWidth model.
        [0 1/A]''')

    extendedrmin = propertyFromExtDoubleAttr('extendedrmin',
        '''Low boundary of the extended r-range, read-only.
        [A]''')

    extendedrmax = propertyFromExtDoubleAttr('extendedrmax',
        '''Upper boundary of the extended r-range, read-only.
        [A]''')

    maxextension = propertyFromExtDoubleAttr('maxextension',
        '''Maximum extension of the r-range that accounts for contributions
        from the out of range peaks.
        [10 A]''')

    rmin = propertyFromExtDoubleAttr('rmin',
        '''Lower bound of the r-grid for PDF calculation.
        [0 A]''')

    rmax = propertyFromExtDoubleAttr('rmax',
        '''Upper bound of the r-grid for PDF calculation.
        [10 A]''')

    rstep = propertyFromExtDoubleAttr('rstep',
        '''Spacing in the calculated r-grid.  r-values are at the
        multiples of rstep.
        [0.01 A]''')


    def __call__(self, structure=None, **kwargs):
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

# class PDFCalculatorInterface

# ----------------------------------------------------------------------------

class DebyePDFCalculator(DebyePDFCalculator_ext, PDFCalculatorInterface):

    # Property wrappers to double attributes of the C++ DebyePDFCalculator_ext

    debyeprecision = propertyFromExtDoubleAttr('debyeprecision',
        '''Cutoff amplitude for the sine contributions to the F(Q).
        [1e-6 unitless]''')

    qmin = propertyFromExtDoubleAttr('qmin',
        '''Lower bound of the Q-grid for the calculated F(Q).
        Affects the shape envelope.
        [0 1/A]
        ''')

    qmax = propertyFromExtDoubleAttr('qmax',
        '''Upper bound of the Q-grid for the calculated F(Q).
        Affects the termination ripples.
        [25 1/A]
        ''')

    qstep = propertyFromExtDoubleAttr('qstep',
        '''Spacing in the Q-grid.  Q-values are at the multiples of qstep.
        [PI/extendedrmax A] unless user overridden.
        See also setOptimumQstep, isOptimumQstep.''')

    # Methods

    def __init__(self, **kwargs):
        '''Create a new instance of the DebyePDFCalculator.
        Keyword arguments can be used to configure the calculator properties,
        for example:

        dpc = DebyePDFCalculator(qmax=20, rmin=7, rmax=15)

        Raise ValueError for invalid keyword argument.
        '''
        super(DebyePDFCalculator, self).__init__()
        setattrFromKeywordArguments(self, **kwargs)
        return


# class DebyePDFCalculator

##############################################################################

class PDFCalculator(PDFCalculator_ext, PDFCalculatorInterface):

    # Property wrappers to double attributes of the C++ PDFCalculator_ext

    peakprecision = propertyFromExtDoubleAttr('peakprecision',
        '''Cutoff amplitude of the peak tail relative to the peak maximum.
        [3.33e-6 unitless]''')

    qmin = propertyFromExtDoubleAttr('qmin',
        '''Lower bound of the experimental Q-range used.
        Affects the shape envelope.
        [0 1/A]''')

    qmax = propertyFromExtDoubleAttr('qmax',
        '''Upper bound of the experimental Q-range used.
        Affects the termination ripples.  Not used when zero.
        [0 1/A]''')

    slope = propertyFromExtDoubleAttr('slope',
        '''Slope of the linear PDF background.  Assigned according to
        number density of the evaluated structure at each PDF calculation.
        Active for LinearBaseline.
        [-4*pi*numdensity unitless]''')

    spdiameter = propertyFromExtDoubleAttr('spdiameter',
        '''Spherical particle diameter for PDF shape damping correction.
        Not used when zero.  Active for SphericalShapeEnvelope.
        [0 A]''')

    stepcut = propertyFromExtDoubleAttr('stepcut',
        '''r-boundary for a step cutoff of the calculated PDF.
        Not used when negative or zero.  Active for StepCutEnvelope.
        Not used when zero.  Active for StepCutEnvelope.
        [0 A]''')

    # Methods

    def __init__(self, **kwargs):
        '''Create a new instance of PDFCalculator.
        Keyword arguments can be used to configure the calculator properties,
        for example:

        pc = PDFCalculator(qmax=20, rmin=7, rmax=15)

        Raise ValueError for invalid keyword argument.
        '''
        super(PDFCalculator, self).__init__()
        setattrFromKeywordArguments(self, **kwargs)
        return

# class PDFCalculator

# End of file
