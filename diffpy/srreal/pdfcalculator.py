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


"""class PDFCalculator    -- highly configurable PDF calculator
"""

# module version
__id__ = "$Id$"


from diffpy.srreal.srreal_ext import PDFCalculator_ext


# Helpers --------------------------------------------------------------------

def _propertyFromExtDoubleAttr(attrname, doc):
    '''Create property wrapper to a DoubleAttr in C++ extension object.

    attrname -- string name of the double attribute
    doc      -- docstring for the Python class property

    Return a property object.
    '''
    def fget(self):
        return self._getDoubleAttr(attrname)
    def fset(self, value):
        self._setDoubleAttr(attrname, value)
        return
    rv = property(fget, fset, doc=doc)
    return rv


# ----------------------------------------------------------------------------

class PDFCalculator(PDFCalculator_ext):

    # Property wrappers to double attributes of the C++ PDFCalculator_ext

    delta1 = _propertyFromExtDoubleAttr('delta1',
        '''Coefficient for (1/r) contribution to the peak sharpening.
        Active for JeongPeakWidth model.
        [0 A]''')

    delta2 = _propertyFromExtDoubleAttr('delta2',
        '''Coefficient for (1/r**2) contribution to the peak sharpening.
        Active for JeongPeakWidth model.
        [0 A**2]''')

    extendedrmin = _propertyFromExtDoubleAttr('extendedrmin',
        '''Low boundary of the extended r-range, read-only.
        [A]''')

    extendedrmax = _propertyFromExtDoubleAttr('extendedrmax',
        '''Upper boundary of the extended r-range, read-only.
        [A]''')

    maxextension = _propertyFromExtDoubleAttr('maxextension',
        '''Maximum extension of the r-range that accounts for contributions
        from the out of range peaks.
        [10 A]''')

    peakprecision = _propertyFromExtDoubleAttr('peakprecision',
        '''Cutoff amplitude of the peak tail relative to the peak maximum.
        [3.33e-6 unitless]''')

    qbroad = _propertyFromExtDoubleAttr('qbroad',
        '''PDF peak broadening from increased intensity noise at high Q.
        Not applied when equal zero.  Active for JeongPeakWidth model.
        [0 1/A]''')

    qdamp = _propertyFromExtDoubleAttr('qdamp',
        '''PDF Gaussian dampening envelope due to limited Q-resolution.
        Not applied when equal to zero.  Active for QResolutionEnvelope.
        [0 1/A]''')

    qmin = _propertyFromExtDoubleAttr('qmin',
        '''Lower bound of the experimental Q-range used.
        Affects the termination ripples.
        [0 1/A]''')

    qmax = _propertyFromExtDoubleAttr('qmax',
        '''Upper bound of the experimental Q-range used.
        Affects the termination ripples.  Not used when zero.
        [0 1/A]''')

    rmin = _propertyFromExtDoubleAttr('rmin',
        '''Lower bound of the r-grid for PDF calculation.
        [0 A]''')

    rmax = _propertyFromExtDoubleAttr('rmax',
        '''Upper bound of the r-grid for PDF calculation.
        [10 A]''')

    rstep = _propertyFromExtDoubleAttr('rstep',
        '''Spacing in the calculated r-grid.  r-values are at the
        multiples of rstep.
        [0.01 A]''')

    scale = _propertyFromExtDoubleAttr('scale',
        '''Scale factor of the calculated PDF.  Active for ScaleEnvelope.
        [1.0 unitless]''')

    slope = _propertyFromExtDoubleAttr('slope',
        '''Slope of the linear PDF background.  Assigned according to
        number density of the evaluated structure at each PDF calculation.
        Active for LinearBaseline.
        [-4*pi*numdensity unitless]''')

    spdiameter = _propertyFromExtDoubleAttr('spdiameter',
        '''Spherical particle diameter for PDF shape damping correction.
        Not used when zero.  Active for SphericalShapeEnvelope.
        [0 A]''')

    stepcut = _propertyFromExtDoubleAttr('stepcut',
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
        for n, v in kwargs.iteritems():
            if not hasattr(self, n):
                emsg = "Unknown attribute %r" % n
                raise ValueError(emsg)
            setattr(self, n, v)
        return


    def __call__(self, structure):
        '''Return PDF of the given structure as an (r, G) tuple.
        '''
        self.eval(structure)
        rv = (self.getRgrid(), self.getPDF())
        return rv

# class PDFCalculator


# End of file
