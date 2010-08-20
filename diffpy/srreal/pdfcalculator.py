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
__all__ = ['DebyePDFCalculator', 'PDFCalculator']

from diffpy.srreal.srreal_ext import DebyePDFCalculator_ext
from diffpy.srreal.srreal_ext import PDFCalculator_ext
from diffpy.srreal.wraputils import propertyFromExtDoubleAttr
from diffpy.srreal.wraputils import setattrFromKeywordArguments

# ----------------------------------------------------------------------------

class PDFCalculatorInterface(object):

    '''Base class for shared properties and methods.
    '''

    delta1 = propertyFromExtDoubleAttr('delta1',
        '''Coefficient for (1/r) contribution to the peak sharpening.
        Active for JeongPeakWidth model.
        [0 A]''')

    delta2 = propertyFromExtDoubleAttr('delta2',
        '''Coefficient for (1/r**2) contribution to the peak sharpening.
        Active for JeongPeakWidth model.
        [0 A**2]''')

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


    def __call__(self, structure, **kwargs):
        '''Calculate PDF for the given structure as an (r, G) tuple.
        Keyword arguments can be used to configure calculator attributes,
        these override any properties that may be passed from the structure,
        such as spdiameter.

        structure    -- a structure object to be evaluated
        kwargs       -- optional parameter settings for this calculator

        Example:    pdfcalc(structure, qmax=20, spdiameter=15)

        Return a tuple of (r, G) numpy arrays.
        '''
        setattrFromKeywordArguments(self, **kwargs)
        self.eval(structure)
        # apply kwargs again if structure contained any attribute
        # that may affect the getPDF result.
        setattrFromKeywordArguments(self, **kwargs)
        rv = (self.getRgrid(), self.getPDF())
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

        No return value.
        Raise ValueError for invalid keyword argument.
        '''
        super(DebyePDFCalculator, self).__init__()
        setattrFromKeywordArguments(self, **kwargs)
        return


# class DebyePDFCalculator_ext

##############################################################################

class PDFCalculator(PDFCalculator_ext, PDFCalculatorInterface):

    # Property wrappers to double attributes of the C++ PDFCalculator_ext

    peakprecision = propertyFromExtDoubleAttr('peakprecision',
        '''Cutoff amplitude of the peak tail relative to the peak maximum.
        [3.33e-6 unitless]''')

    qdamp = propertyFromExtDoubleAttr('qdamp',
        '''PDF Gaussian dampening envelope due to limited Q-resolution.
        Not applied when equal to zero.  Active for QResolutionEnvelope.
        [0 1/A]''')

    qmin = propertyFromExtDoubleAttr('qmin',
        '''Lower bound of the experimental Q-range used.
        Affects the shape envelope.
        [0 1/A]''')

    qmax = propertyFromExtDoubleAttr('qmax',
        '''Upper bound of the experimental Q-range used.
        Affects the termination ripples.  Not used when zero.
        [0 1/A]''')

    scale = propertyFromExtDoubleAttr('scale',
        '''Scale factor of the calculated PDF.  Active for ScaleEnvelope.
        [1.0 unitless]''')

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

        No return value.
        Raise ValueError for invalid keyword argument.
        '''
        super(PDFCalculator, self).__init__()
        setattrFromKeywordArguments(self, **kwargs)
        return

# class PDFCalculator

# DebyePDFCalculator_ext pickling support ------------------------------------

def _dbpdfc_getstate(self):
    dbattrs = [(n, self._getDoubleAttr(n))
        for n in self._namesOfWritableDoubleAttributes()]
    # NOTE: convert names below to objects as they get pickle support
    state = (self.__dict__,
            self.getPeakWidthModel().type(),
            self.getScatteringFactorTable(),
            self.usedEnvelopeTypes(),
            dbattrs)
    return state

def _dbpdfc_setstate(self, state):
    if len(state) != 5:
        emsg = ("expected 5-item tuple in call to __setstate__, got %r" +
                repr(state))
        raise ValueError(emsg)
    st = iter(state)
    self.__dict__.update(st.next())
    self.setPeakWidthModelByType(st.next())
    self.setScatteringFactorTable(st.next())
    self.clearEnvelopes()
    for tp in st.next():
        self.addEnvelopeByType(tp)
    for n, v in st.next():
        self._setDoubleAttr(n, v)
    return

# inject pickle methods to PDFCalculator_ext

DebyePDFCalculator_ext.__getstate_manages_dict__ = True
DebyePDFCalculator_ext.__getstate__ = _dbpdfc_getstate
DebyePDFCalculator_ext.__setstate__ = _dbpdfc_setstate

# PDFCalculator_ext pickling support -----------------------------------------

def _pdfc_getstate(self):
    dbattrs = [(n, self._getDoubleAttr(n))
        for n in self._namesOfWritableDoubleAttributes()]
    # NOTE: convert names below to objects as they get pickle support
    state = (self.__dict__,
            self.getPeakWidthModel().type(),
            self.getPeakProfile().type(),
            self.getScatteringFactorTable(),
            self.usedEnvelopeTypes(),
            self.getBaseline().type(),
            dbattrs)
    return state

def _pdfc_setstate(self, state):
    if len(state) != 7:
        emsg = ("expected 7-item tuple in call to __setstate__, got %r" +
                repr(state))
        raise ValueError(emsg)
    st = iter(state)
    self.__dict__.update(st.next())
    self.setPeakWidthModelByType(st.next())
    self.setPeakProfileByType(st.next())
    self.setScatteringFactorTable(st.next())
    self.clearEnvelopes()
    for tp in st.next():
        self.addEnvelopeByType(tp)
    self.setBaselineByType(st.next())
    for n, v in st.next():
        self._setDoubleAttr(n, v)
    return

# inject pickle methods to PDFCalculator_ext

PDFCalculator_ext.__getstate_manages_dict__ = True
PDFCalculator_ext.__getstate__ = _pdfc_getstate
PDFCalculator_ext.__setstate__ = _pdfc_setstate

# End of file
