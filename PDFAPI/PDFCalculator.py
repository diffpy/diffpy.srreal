#!/usr/bin/env python

import numpy

from RefinementAPI.Calculator import Calculator
from diffpy.Structure import Atom, Lattice, Structure
from diffpy.pdffit2 import PdfFit
from adps import AnisotropicAtomicDisplacementFactor as adp
from RefinementAPI.utils import rebinArray

class PDFCalculator(Calculator):
    """Calculator class for generating a PDF signal.

    This calls on pdffit2 to calculate a signal from the configured PDFData and
    CrystalPhase.

    Attributes
    _comp       --  PDFComponent that contains this PDFCalculator. This is set
                    by the PDFComponent on initialization.
    _pdffit     --  An instance of the PdfFit class.
    """

    def __init__(self):
        Calculator.__init__(self)
        self._pdffit = None
        self._comp = None

    def __setstate__(self):
        """Needed for pickling.

        This sets the _pdffit attribute to None, since it is unpickleable.
        """
        state = dict(self.__dict__)
        state["_pdffit"] = None
        return state

    def clear(self):
        """Delete the _pdffit instance."""
        self._pdffit = None
        return

    def calculate(self):
        """The profile equation."""
        if self._pdffit is None:
            self._setupEngine()
        self._updateEngine()
        self._pdffit.calc()
        x = numpy.array(self._pdffit.getR())
        y = numpy.array(self._pdffit.getpdf_fit())
        if 0:
            from pylab import plot, show
            plot(x,y)
            show()
        self._ycal = rebinArray(y, x, self._xcal)
        return

    def _updateEngine(self):
        """Update the engine with the current parameter values."""

        # Get all the information from the component
        comp = self._comp

        # Set up the structures first
        for i in range(comp._model.getNumPhases()):

            self._pdffit.setphase(i+1)

            phase = comp._model.getPhase(i)

            # Scale, etc.
            self._pdffit.setvar("pscale", phase.weight)
            self._pdffit.setvar("delta1", phase.delta1)
            self._pdffit.setvar("delta2", phase.delta2)
            self._pdffit.setvar("sratio", phase.sratio)
            self._pdffit.setvar("rcut", phase.rcut)

            # Lattice
            plat = phase.getLattice()
            self._pdffit.setvar("lat(1)", plat.a)
            self._pdffit.setvar("lat(2)", plat.b)
            self._pdffit.setvar("lat(3)", plat.c)
            self._pdffit.setvar("lat(4)", plat.alpha)
            self._pdffit.setvar("lat(5)", plat.beta)
            self._pdffit.setvar("lat(6)", plat.gamma)

            # Atoms
            for j in range( phase.getNumAtoms() ):
                pa = phase.getAtom(j)

                self._pdffit.setvar(CustomPdfFit.x(j+1), pa.x)
                self._pdffit.setvar(CustomPdfFit.y(j+1), pa.y)
                self._pdffit.setvar(CustomPdfFit.z(j+1), pa.z)

                a = Atom(pa._element)
                u = pa.getADP()
                if isinstance(u, adp):
                    a.B11 = u.B11
                    a.B22 = u.B22
                    a.B33 = u.B33
                    a.B12 = u.B12
                    a.B13 = u.B13
                    a.B23 = u.B23
                else:
                    a.Bisoequiv = u.Biso

                self._pdffit.setvar(CustomPdfFit.u11(j+1), a.U11)
                self._pdffit.setvar(CustomPdfFit.u22(j+1), a.U22)
                self._pdffit.setvar(CustomPdfFit.u33(j+1), a.U33)
                self._pdffit.setvar(CustomPdfFit.u12(j+1), a.U12)
                self._pdffit.setvar(CustomPdfFit.u13(j+1), a.U13)
                self._pdffit.setvar(CustomPdfFit.u23(j+1), a.U23)


        # Now set up the data information
        data = comp.getData()
        qdamp = data.qdamp
        self._pdffit.setvar("qdamp", data.qdamp)
        self._pdffit.setvar("qbroad", data.qbroad)
        self._pdffit.setvar("spdiameter", data.spdiameter)

        return


    def _setupEngine(self):
        """Set up the engine for the calculation."""

        comp = self._comp
        self._pdffit = CustomPdfFit()

        # Set up the structures first
        for i in range(comp._model.getNumPhases()):
            phase = comp._model.getPhase(i)

            plat = phase.getLattice()
            lat = Lattice(plat.a, plat.b, plat.c, plat.alpha, plat.beta,
                    plat.gamma)

            stru= Structure(lattice = lat)

            for j in range( phase.getNumAtoms() ):

                pa = phase.getAtom(j)
                xyz = [pa.x, pa.y, pa.z]
                a = Atom(pa._element, xyz)
                u = pa.getADP()
                if isinstance(u, adp):
                    a.B11 = u.B11
                    a.B22 = u.B22
                    a.B33 = u.B33
                    a.B12 = u.B12
                    a.B13 = u.B13
                    a.B23 = u.B23
                else:
                    a.Bisoequiv = u.Biso

                stru.append(a)

            # Add the structure to pdffit
            self._pdffit.add_structure(stru)

        # Now set up the data information
        data = comp.getData()
        stype = data.getScatteringType()
        qmax = data.getQmax()
        rmin = self._xcal[0]
        rmax = self._xcal[-1]
        bin = len(self._xcal)
        self._pdffit.alloc(stype, qmax, 0, rmin, rmax, bin)

# End of PDFCalculator

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


