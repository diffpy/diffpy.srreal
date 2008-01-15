#!/usr/bin/env python

import os, sys
import numpy  

from park.theory.parkTheory import Theory
from park.fit.xmlParameter import XmlParameter
from park.fit.xmlDataArray import XmlDataArray

from park.xmlUtil.xmlHelper import setDefault
import re

PDF_PARAM_NAME_PREFIX = "p"

from diffpy.pdffit2 import PdfFit

class CustomPdfFit(PdfFit):
    """A PdfFit class that works well as a function calculator.
    
    This class suppresses output.
    """

    def __init__(self):
        import os
        from diffpy.pdffit2 import redirect_stdout
        redirect_stdout(os.tmpfile())
        PdfFit.__init__(self)


def _mangleParameter(pname):
    """Mangle a parameter name so it is suitable as an attribute name.

    e.g. "lat(1)" -> "lat_1"
         "pscale" -> "pscale"

    Returns the mangled parameter name.
    """
    mname = re.sub("\(", "_", pname)
    mname = re.sub("\)", "", mname)
    return mname

def _demangleParameter(pname):
    """Demangle a parameter name so it is suitable for pdffit.

    e.g. "lat_1"  -> "lat(1)"
         "pscale" -> "pscale"

    Returns the demangled parameter name.
    """
    dmname = re.sub("_", "(", pname)
    if dmname != pname:
        dmname = re.sub("$", ")", dmname)
    return dmname

class PDFParameter(XmlParameter):

    def __init__(self):
        XmlParameter.__init__(self)
        self._qmax = 0
        self._stype = "X"
        self.lat_1 = 1.0
        self.lat_2 = 1.0
        self.lat_3 = 1.0
        self.lat_4 = 90.0
        self.lat_5 = 90.0
        self.lat_6 = 90.0

def getPDFParameters(P, stype, qmax):
    """Get a list of XmlParameter instances for a partially configured PDF fit.

    P       --  Instance of PdfFit class.
    stype   --  Scattering type, "X" or "N"
    qmax    --  Maximum q value used in the data

    P should already have phases loaded before being used in this function.

    Returns a list of configured PDFParameter instances, one for each phase in
    the fit.
    """
    from diffpy.pdffit2 import pdffit2
    ip = 1
    params = []
    m = _mangleParameter

    while(1):

        # We have no way of asking PdfFit how many phases it has. We have to do
        # it the hard way.
        try:
            P.setphase(ip)
        except pdffit2.unassignedError:
            break

        pm = PDFParameter()
        pm.setName('%s%i'%(PDF_PARAM_NAME_PREFIX,ip-1))

        pm._stype = stype
        pm._qmax = qmax

        ip += 1
        atoms = P.get_atoms()
        # Create the lattice parameters
        for i in range(1,7):
            setDefault(pm, m(P.lat(i)), P.getvar(P.lat(i)))

        # Create the atomic parameters
        for i in range(1,1+P.num_atoms()):
            setDefault(pm, "_atype_%i"%i, atoms[i-1])
            setDefault(pm, m(P.x(i)), P.getvar(P.x(i)))
            setDefault(pm, m(P.y(i)), P.getvar(P.y(i)))
            setDefault(pm, m(P.z(i)), P.getvar(P.z(i)))
            setDefault(pm, m(P.u11(i)), P.getvar(P.u11(i)))
            setDefault(pm, m(P.u22(i)), P.getvar(P.u22(i)))
            setDefault(pm, m(P.u33(i)), P.getvar(P.u33(i)))
            setDefault(pm, m(P.u12(i)), P.getvar(P.u12(i)))
            setDefault(pm, m(P.u13(i)), P.getvar(P.u13(i)))
            setDefault(pm, m(P.u23(i)), P.getvar(P.u23(i)))
            setDefault(pm, m(P.occ(i)), P.getvar(P.occ(i)))

        # Create other structure-related parameters
        setDefault(pm, m(P.pscale()), P.getvar(P.pscale()))
        setDefault(pm, m(P.delta1()), P.getvar(P.delta1()))
        setDefault(pm, m(P.delta2()), P.getvar(P.delta2()))
        setDefault(pm, m(P.sratio()), P.getvar(P.sratio()))
        setDefault(pm, m(P.rcut()), P.getvar(P.rcut()))

        # Non-structure parameters
        setDefault(pm, m(P.dscale()), 1)
        setDefault(pm, m(P.qdamp()), 0.01)
        setDefault(pm, m(P.qbroad()), 0)
        setDefault(pm, m(P.spdiameter()), 0)

        params.append(pm)

    return params
    
class PDFTheory(Theory): 
    """ PDF theory, as implemented in pdfit2.
    """ 

    def __init__(self):
        Theory.__init__(self)
        self._P = None

    def __configureCalculator(self, params, xdata):
        """Configure the calculator based on the parameters and data.
        
        Returns the configured calculator
        """
        from diffpy.Structure.structure import Structure, Atom, Lattice

        P = CustomPdfFit()

        # Create a structure for each parameter set and add it to the
        # calculator.
        for pm in params:

            lattice = Lattice(a = pm.lat_1, b = pm.lat_2, c = pm.lat_3,
                        alpha = pm.lat_4, beta = pm.lat_5, gamma = pm.lat_6)

            # We don't know the number of atoms, so we have to try to count them
            i = 1
            atoms = []
            while(1):

                try:
                    atype = getattr(pm, "_atype_%i"%i)
                except AttributeError:
                    break

                x = getattr(pm, "x_%i"%i)
                y = getattr(pm, "y_%i"%i)
                z = getattr(pm, "z_%i"%i)
                xyz = numpy.array([x,y,z])
                occ = getattr(pm, "occ_%i"%i)
                u11 = getattr(pm, "u11_%i"%i)
                u22 = getattr(pm, "u22_%i"%i)
                u33 = getattr(pm, "u33_%i"%i)
                u12 = getattr(pm, "u12_%i"%i)
                u13 = getattr(pm, "u13_%i"%i)
                u23 = getattr(pm, "u23_%i"%i)
                U = numpy.array([[u11,u12,u13],
                                 [u12,u22,u23],
                                 [u13,u23,u33]])

                A = Atom( atype = atype,
                          xyz = [x,y,z],
                          occupancy=occ,
                          U = U)

                atoms.append(A)
                i += 1

            S = Structure(atoms = atoms, lattice = lattice)

            # Add this structure as a phase to the calculator
            P.read_struct_string(S.writeStr("pdffit"))

            # Get the auxilliary information
            stype = pm._stype
            qmax = pm._qmax

        # Now prepare the calculation
        P.alloc(str(stype), qmax, 0, xdata[0], xdata[-1], len(xdata))

        # We're done!
        return P


    def _getFx(self, data, params):
        """Calculate the pdf function."""  
        if data is None:
            return                   
        try:
            xdata = data[0].getData()
            array = XmlDataArray()
        except:
            xdata = data[0]
            array = None
            
        dm = _demangleParameter

        # configure the PDF calculator based on parameters.
        if self._P is None:
            self._P = self.__configureCalculator(params, xdata)

        # Set all parameters. This is a bit redundant right after the calculator
        # is configures, but the # the overhead is small compared to the
        # calculation time.
        pnum = 1
        for pm in params:
            self._P.setphase(pnum)
            pnum += 1
            for pname in pm.getAttributeNames():
                # Ignore protected names.
                if not pname.startswith("_"):
                    self._P.setvar(dm(pname), getattr(pm,pname))

        self._P.calc()
        y0 = numpy.array(self._P.getpdf_fit())

        if array is None:
            return (y0,)
        else:
            array.setData(y0)      
            return (array, )


    def _getObjectiveFx(self, data, params):
        """ calculate the chisq.
        """
        dy = self._getResidual(data, params)

        try:
            u = data[2].getData()
        except:
            u = data[2]
        # FIXME: Sometimes the arrays are different sizes.
        if len(dy) != len(u):
            u = 1
        r = dy/u
        return numpy.dot(r, r)

    def _getResidual(self, data, params):
        """ Return a vector to calculate norm2. It is required by
            leastsq() method to calculate the error bar of parameters. 
        """
        y0 = self._getFx(data, params)
        try:
            ydata = data[1].getData()
            ycalc = y0[0].getData()
        except:
            ydata = data[1]
            ycalc = y0[0]

            
        dy = ycalc - ydata
        return dy

def main():
    from park.fit.xmlModel import XmlModel
    from pdfDataset import PDFDataset

    
    model = XmlModel()
    model.name='pdf'
    model.theory = 'PDFTheory'

    from diffpy.pdffit2 import PdfFit

    P = PdfFit()
    P.read_struct("examples/Ni.stru")
    
    # Create the parameters from P
    for pm in getPDFParameters(P, "X", 45.0):
        model.addChild(pm)

    dataset = model.getXmlDataset()
    print 'dataset is None:', dataset is None
    model.setXmlDataset(PDFDataset())
    
    theory = PDFTheory()
    x0 = numpy.arange(1.5,15,0.1)
    for i in range(10):
        y0 = theory._getFx((x0,),model.getXmlParameters() )
        
    print 'pdf theory:', y0[0] 
    print 'pdf model', model

    return

if __name__ == '__main__':
    main()
