#!/usr/bin/env python
"""\
This script partially demonstrates SrFit UC1-20 and SrFitUC1-100, where a
two-phase model is refined or used to calculate a PDF.
"""

__id__ = "$Id$"

def fitSiNi():
    """Fit a physical mixture of Silicon and Nickel, while finding the mixing
    ratio (pscale).
    """

    import park
    import numpy
    from diffpy.srreal.pdf import PDFModel, phaseFromStructure, structureFromPhase

    # Create a new refinement model
    model = PDFModel("SiNiModel")
    # Set the qmax cutoff for the pdf data
    model.stype = "X"
    model.qmax = 27.0

    # Use diffpy.Structure to read the structure from file
    import diffpy.Structure
    nistru = diffpy.Structure.Structure()
    nistru.read("testdata/ni.stru", "pdffit")
    # Convert the structure to a CrystalStructure object that the PDFModel can
    # use.
    niphase = phaseFromStructure(nistru, "nickel")

    sistru = diffpy.Structure.Structure()
    sistru.read("testdata/si.stru", "pdffit")
    siphase = phaseFromStructure(sistru, "silicon")

    # Add the phase to the model.
    model.addPhase(niphase)
    model.addPhase(siphase)

    # Load the data and select the data points that we want to fit
    from park.modelling.data import Data1D
    dat = Data1D(filename="testdata/si90ni10-q27r60-xray.gr")
    # This selects every-other data point up to r = 10
    import bisect
    idxlo = bisect.bisect(dat.x, 1.5)
    idxhi = bisect.bisect(dat.x, 20)
    dat.select(range(idxlo, idxhi))

    # Create the assembly, which associates the model with the data
    assemb = park.Assembly([(model,dat), (niphase,), (siphase,)])

    # Constrain parameters
    # Scale factors
    model.dscale.value = 1.0
    model.dscale.set((0.2, 2.0))
    niphase.pscale.value = 0.9
    niphase.pscale.set((0.1,0.9))
    siphase.pscale.set("1.0-%s"%niphase.pscale.path)

    # Ni phase
    nilat = niphase.lattice
    nilat.b.set(nilat.a.path)
    nilat.c.set(nilat.a.path)
    nilat.a.set((3.5,3.55))
    niphase.delta2.value = 2
    niphase.delta2.set((0, 10))
    # Loop over atoms and constrain the thermal parameters to be isotropic
    niatoms = niphase.getAtoms()
    u11path = niatoms[0]["u11"].path
    for i, atom in enumerate(niatoms):
        if i != 0: atom.u11.set(u11path)
        atom.u22.set(u11path)
        atom.u33.set(u11path)
    niatoms[0].u11.value = 0.005
    niatoms[0].u11.set( (0.001, 0.01) )

    # Si phase
    silat = siphase.lattice
    silat.b.set(silat.a.path)
    silat.c.set(silat.a.path)
    silat.a.set((5.42,5.46))
    siphase.delta2.value = 2
    siphase.delta2.set((0, 10))
    # Loop over atoms and constrain the thermal parameters to be isotropic
    siatoms = siphase.getAtoms()
    u11path = siatoms[0]["u11"].path
    for i, atom in enumerate(siatoms):
        if i != 0: atom.u11.set(u11path)
        atom.u22.set(u11path)
        atom.u33.set(u11path)
    siatoms[0].u11.value = 0.005
    siatoms[0].u11.set( (0.001, 0.01) )

    # Model parameters
    model.qdamp.value = 0.05
    model.qdamp.set((0.0, 0.1))

    # Configure the fit
    from park.fitting.fitresult import ConsoleUpdate
    from park.optim.fitmc import FitMC
    handler = ConsoleUpdate(improvement_delta=0.1,progress_delta=1)
    fitter = FitMC(start_points=1)

    # Run the fit and save the results
    print "Starting refinement of sini"
    import park.fitting.fit
    result = park.fitting.fit.fit(assemb, handler=handler, fitter=fitter)
    result.print_summary()
    numpy.savetxt("sini.fit", zip(dat.fit_x,dat.calc_y))
    ofile = file("sini.res", 'w')
    result.print_summary(ofile)
    ofile.close()

    # Plot the fit
    if 0:
        diff = dat.fit_y - dat.calc_y + 1.35*min(dat.fit_y)
        import pylab
        pylab.clf()
        pylab.plot(dat.fit_x, dat.fit_y, "bo", label = "data")
        pylab.plot(dat.fit_x, dat.calc_y, "r-", label = "fit", linewidth=2)
        pylab.plot(dat.fit_x, diff, "g-", label = "difference", linewidth=2)
        pylab.legend()
        pylab.title("Park fit to sini using pdffit2 as calculator")
        pylab.xlabel("$r (\AA)$")
        pylab.ylabel("$G (\AA^{-2})$")
        pylab.xlim(0, 20)
        pylab.ylim(-10, 15)
        pylab.savefig("sini.fit.png")

        # Save the refined structures
        nistru = structureFromPhase(niphase)
        nistru.write("ni.rstr", "pdffit")
        sistru = structureFromPhase(siphase)
        sistru.write("si.rstr", "pdffit")

    return

def calcSiNi():
    """Calculate a PDF pattern from a physical mixture of Silicon and Nickel."""

    import park
    import numpy
    from diffpy.srreal.pdf import PDFModel, phaseFromStructure, structureFromPhase

    # Create a new refinement model
    model = PDFModel("SiNiModel")
    # Set the qmax cutoff for the pdf data
    model.stype = "X"
    model.qmax = 27.0

    # Use diffpy.Structure to read the structure from file
    import diffpy.Structure
    nistru = diffpy.Structure.Structure()
    nistru.read("testdata/ni.stru", "pdffit")
    # Convert the structure to a CrystalStructure object that the PDFModel can
    # use.
    niphase = phaseFromStructure(nistru, "nickel")

    sistru = diffpy.Structure.Structure()
    sistru.read("testdata/si.stru", "pdffit")
    siphase = phaseFromStructure(sistru, "silicon")

    # Add the phase to the model.
    model.addPhase(niphase)
    model.addPhase(siphase)

    # Load the data and select the data points that we want to fit
    from park.modelling.data import Data1D
    xvals = numpy.arange(1.5, 10, 0.05)
    yvals = numpy.zeros(len(xvals))
    dat = Data1D(x=xvals, y=yvals)
    # Set the calculation range
    # Create the assembly, which associates the model with the data
    assemb = park.Assembly([(model,dat), (niphase,), (siphase,)])

    # Constrain parameters
    # Scale factors
    model.dscale.set(1)
    niphase.pscale.set(0.1)
    siphase.pscale.set(0.9)

    # Calculate the signal
    assemb.fit_parameters()
    assemb.eval()

    # Save the results
    numpy.savetxt("sini.calc", zip(dat.fit_x,dat.calc_y))

    return


if __name__ == "__main__":

    calcSiNi()
    fitSiNi()

