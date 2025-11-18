#!/usr/bin/env python

"""Tune the peak precision parameter so that PDFCalculator gives
equivalent results to diffpy.pdffit2.

Usage: tunePeakPrecision.py [qmax] [peakprecision] [createplot]
"""

# global imports
import sys
import time

import numpy

import diffpy.pdffit2
from diffpy.srreal.pdf_ext import PDFCalculator
from diffpy.structure import Structure

# Results:
# Qmax  peakprecision   CPU     Notes
# 15    3.2e-6                  clear minimum
# 15    3.2e-6                  clear minimum
# 20    3.2e-6                  clear minimum
# 25    0.45e-6                 flat shape

# PDF calculation parameters
qmax = 0.0
rmin = 0.01
rmax = 50.0
rstep = 0.01
peakprecision = None
createplot = False

# make PdfFit silent
diffpy.pdffit2.redirect_stdout(open("/dev/null", "w"))

# define nickel structure data
nickel_discus_data = """
title   Ni
spcgr   P1
cell    3.523870,  3.523870,  3.523870, 90.000000, 90.000000, 90.000000
ncell          1,         1,         1,         4
atoms
NI          0.00000000        0.00000000        0.00000000       0.1000
NI          0.00000000        0.50000000        0.50000000       0.1000
NI          0.50000000        0.00000000        0.50000000       0.1000
NI          0.50000000        0.50000000        0.00000000       0.1000
"""

nickel = Structure()
nickel.readStr(nickel_discus_data, format="discus")


def Gpdffit2(qmax):
    """Calculate reference nickel PDF using diffpy.pdffit2.

    Attributes
    ----------
    qmax
        vawevector cutoff value in 1/A

    Return numpy array of (r, g).
    """
    # calculate reference data using pdffit2
    pf2 = diffpy.pdffit2.PdfFit()
    pf2.alloc("X", qmax, 0.0, rmin, rmax, int((rmax - rmin) / rstep + 1))
    pf2.add_structure(nickel)
    pf2.calc()
    rg = numpy.array((pf2.getR(), pf2.getpdf_fit()))
    return rg


def Gsrreal(qmax, peakprecision=None):
    """Calculate nickel PDF using PDFCalculator from diffpy.srreal.

    Attributes
    ----------
    qmax
        vawevector cutoff value in 1/A
    peakprecision
        precision factor affecting peak cutoff,
        keep at default value when None.

    Return numpy array of (r, g).
    """
    pdfcalc = PDFCalculator()
    pdfcalc._setDoubleAttr("qmax", qmax)
    pdfcalc._setDoubleAttr("rmin", rmin)
    pdfcalc._setDoubleAttr("rmax", rmax + rstep * 1e-4)
    pdfcalc._setDoubleAttr("rstep", rstep)
    if peakprecision is not None:
        pdfcalc._setDoubleAttr("peakprecision", peakprecision)
    pdfcalc.eval(nickel)
    rg = numpy.array([pdfcalc.rgrid, pdfcalc.pdf])
    return rg


def comparePDFCalculators(qmax, peakprecision=None):
    """Compare Ni PDF calculations with pdffit2 and PDFCalculator.

    Attributes
    ----------
    qmax
        vawevector cutoff value in 1/A
    peakprecision
        precision factor affecting peak cutoff,
        keep at default value when None.

    Return a dictionary of benchmark results with the following keys:

    Attributes
    ----------
    qmax
        vawevector cutoff value
    peakprecision
        actual peak precision used in PDFCalculator
    r
        common r-grid for the PDF arrays
        g0, g1  -- calculated PDF curves from pdffit2 and PDFCalculator
    gdiff
        PDF difference equal (g0 - g1)
    grmsd
        root mean square value of PDF curves difference
        t0, t1  -- CPU times used by pdffit2 and PDFCalculator calls
    """
    rv = {}
    rv["qmax"] = qmax
    rv["peakprecision"] = (
        peakprecision is None
        and PDFCalculator()._getDoubleAttr("peakprecision")
        or peakprecision
    )
    ttic = time.clock()
    rg0 = Gpdffit2(qmax)
    ttoc = time.clock()
    rv["r"] = rg0[0]
    rv["g0"] = rg0[1]
    rv["t0"] = ttoc - ttic
    ttic = time.clock()
    rg1 = Gsrreal(qmax, peakprecision)
    ttoc = time.clock()
    assert numpy.all(rv["r"] == rg1[0])
    rv["g1"] = rg1[1]
    rv["t1"] = ttoc - ttic
    rv["gdiff"] = rv["g0"] - rv["g1"]
    rv["grmsd"] = numpy.sqrt(numpy.mean(rv["gdiff"] ** 2))
    return rv


def processCommandLineArguments():
    global qmax, peakprecision, createplot
    argc = len(sys.argv)
    if set(["-h", "--help"]).intersection(sys.argv):
        print(__doc__)
        sys.exit()
    if argc > 1:
        qmax = float(sys.argv[1])
    if argc > 2:
        peakprecision = float(sys.argv[2])
    if argc > 3:
        createplot = sys.argv[3].lower() in ("y", "yes", "1", "true")
    return


def plotComparison(cmpdata):
    """Plot comparison of PDF curves.

    Attributes
    ----------
    cmpdata
        dictionary returned from comparePDFCalculators

    No return value.
    """
    import pylab

    pylab.clf()
    pylab.subplot(211)
    pylab.title("qmax=%(qmax)g  peakprecision=%(peakprecision)g" % cmpdata)
    pylab.ylabel("G")
    r = cmpdata["r"]
    g0 = cmpdata["g0"]
    g1 = cmpdata["g1"]
    gdiff = cmpdata["gdiff"]
    pylab.plot(r, g0, r, g1)
    pylab.subplot(212)
    pylab.plot(r, gdiff, "r")
    slope = numpy.sum(r * gdiff) / numpy.sum(r**2)
    pylab.plot(r, slope * r, "--")
    pylab.xlabel("r")
    pylab.ylabel("Gdiff")
    pylab.title("slope = %g" % slope)
    pylab.draw()
    return


def main():
    processCommandLineArguments()
    cmpdata = comparePDFCalculators(qmax, peakprecision)
    print(
        (
            "qmax = %(qmax)g  pkprec = %(peakprecision)g  "
            + "grmsd = %(grmsd)g  t0 = %(t0).3f  t1 = %(t1).3f"
        )
        % cmpdata
    )
    if createplot:
        plotComparison(cmpdata)
        import pylab

        pylab.show()
    return


if __name__ == "__main__":
    main()
