#!/usr/bin/env python
"""\
This script partially demonstrates SrFit UC1-1 and UC1-2. An instance of
diffpy.Structure.Structure is created by reading strutural information from file
and is then converted to a srreal.CrystalStructure instance that can be refined
by park.  Constraints on parameters are made using mechanisms provided by park.
Introspection capabilities from SrFit are not demonstrated in this script.

Alternate scenerios:
    The diffpy.Structure.Structure instance can be created manually.
"""

__id__ = "$Id$"

def fitNickelFromFile(dataname, stype):

    import park
    import numpy
    from diffpy.srreal.pdf import PDFModel, phaseFromStructure, structureFromPhase

    outname = ".".join(dataname.split(".")[:-1])

    # Create a new refinement model
    model = PDFModel("NiModel")
    # Set the qmax cutoff for the pdf data
    model.stype = stype
    model.qmax = 27.0

    # Use diffpy.Structure to read the structure from file
    import diffpy.Structure
    stru = diffpy.Structure.Structure()
    stru.read("testdata/ni.stru", "pdffit")
    # Convert the structure to a CrystalStructure object that the PDFModel can
    # use.
    phase = phaseFromStructure(stru)

    # Add the phase to the model.
    model.addPhase(phase)

    # Load the data and select the data points that we want to fit
    from park.modelling.data import Data1D
    dat = Data1D(filename="testdata/"+dataname)
    # This selects every-other data point up to r = 10
    import bisect
    idxlo = bisect.bisect(dat.x, 1.8)
    idxhi = bisect.bisect(dat.x, 10)
    dat.select(range(idxlo, idxhi, 2))

    # Create the assembly, which associates the model with the data and
    # specifies which phases are part of the fit
    assemb = park.Assembly([(model,dat), (phase,)])

    # Constrain parameters
    # This refines dscale between 0.2 and 2.0
    model.dscale.set((0.2, 2.0))
    # This sets lattice parameters b and c equal to a
    lat = phase.lattice
    lat.b.set( lat.a.path )
    lat.c.set( lat.a.path )
    lat.a.set((3.50, 3.55))
    # This constrains delta2 and qdamp
    phase.delta2.set((0, 10))
    model.qdamp.set((0.0, 0.1))
    # Loop over atoms and constrain the thermal parameters to be isotropic
    atoms = phase.getAtoms()
    u11path = atoms[0]["u11"].path
    for i, atom in enumerate(atoms):
        if i != 0:
            atom.u11.set(u11path)
        atom.u22.set(u11path)
        atom.u33.set(u11path)
    atoms[0].u11.set( (0.001, 0.01) )

    # Configure the fit
    from park.fitting.fitresult import ConsoleUpdate
    from park.optim.fitmc import FitMC
    handler = ConsoleUpdate(improvement_delta=0.1,progress_delta=1)
    fitter = FitMC(start_points=1)

    # Run the fit and save the results
    print "Starting refinement of", dataname
    import park.fitting.fit
    result = park.fitting.fit.fit(assemb, handler=handler, fitter=fitter)
    result.print_summary()
    numpy.savetxt("%s.fit"%outname, zip(dat.fit_x,dat.calc_y))
    ofile = file("%s.res"%outname, 'w')
    result.print_summary(ofile)
    ofile.close()

    # Plot the fit
    diff = dat.fit_y - dat.calc_y + 1.25*min(dat.fit_y)
    import pylab
    pylab.clf()
    pylab.plot(dat.fit_x, dat.fit_y, "bo", label = "data")
    pylab.plot(dat.fit_x, dat.calc_y, "r-", label = "fit", linewidth=2)
    pylab.plot(dat.fit_x, diff, "g-", label = "difference", linewidth=2)
    pylab.legend()
    pylab.title("Park fit to %s using pdffit2 as calculator"%dataname)
    pylab.xlabel("$r (\AA)$")
    pylab.ylabel("$G (\AA^{-2})$")
    pylab.xlim(0, 10)
    pylab.ylim(-15, 25)
    pylab.savefig("%s.fit.png"%outname)

    # Save the refined
    stru = structureFromPhase(phase)
    stru.write("%s.rstr"%outname, "pdffit")
    return


def fitTwoNickel():

    import park
    import numpy
    from diffpy.srreal.pdf import PDFModel, phaseFromStructure, structureFromPhase

    # Use diffpy.Structure to read the structure from file
    import diffpy.Structure
    stru = diffpy.Structure.Structure()
    stru.read("testdata/ni.stru", "pdffit")
    # Convert the structure to a CrystalStructure object that the PDFModel can
    # use.
    phase = phaseFromStructure(stru)
    pname = phase.name

    # Create a new refinement model
    modelN = PDFModel("NiN")
    # Set the qmax cutoff for the pdf data
    modelN.stype = "N"
    modelN.qmax = 27.0

    modelX = PDFModel("NiX")
    # Set the qmax cutoff for the pdf data
    modelX.stype = "X"
    modelX.qmax = 27.0


    # Add the phase to the models
    modelN.addPhase(phase)
    modelX.addPhase(phase)

    # Load the data and select the data points that we want to fit
    from park.modelling.data import Data1D
    import bisect
    datN = Data1D(filename="testdata/ni-q27r100-neutron.gr")
    # This selects every-other dat1a point up to r = 10
    idxlo = bisect.bisect(datN.x, 1.8)
    idxhi = bisect.bisect(datN.x, 10)
    datN.select(range(idxlo, idxhi, 2))
    datX = Data1D(filename="testdata/ni-q27r60-xray.gr")
    idxlo = bisect.bisect(datX.x, 1.8)
    idxhi = bisect.bisect(datX.x, 10)
    datX.select(range(idxlo, idxhi, 2))

    # Create the assembly, which associates the model with the data
    assemb = park.Assembly([(modelN,datN), (modelX,datX), (phase,)])

    # Constrain parameters
    # This refines dscale between 0.2 and 2.0
    modelN.dscale.set((0.2, 2.0))
    modelX.dscale.set((0.2, 2.0))
    # This sets lattice parameters b and c equal to a
    lat = phase.lattice
    lat.b.set( lat.a.path )
    lat.c.set( lat.a.path )
    lat.a.set((3.50, 3.55))
    # This constrains delta2 and qdamp
    phase.delta2.set((0, 10))
    modelX.qdamp.set((0.0, 0.1))
    modelN.qdamp.set((0.0, 0.1))
    # Loop over atoms and constrain the thermal parameters to be isotropic
    atoms = phase.getAtoms()
    u11path = atoms[0]["u11"].path
    for i, atom in enumerate(atoms):
        if i != 0:
            atom.u11.set(u11path)
        atom.u22.set(u11path)
        atom.u33.set(u11path)
    atoms[0].u11.set( (0.001, 0.01) )

    # Configure the fit
    from park.fitting.fitresult import ConsoleUpdate
    from park.optim.fitmc import FitMC
    handler = ConsoleUpdate(improvement_delta=0.1,progress_delta=1)
    fitter = FitMC(start_points=1)

    # Run the fit and save the results
    print "Starting refinement of both data sets"
    import park.fitting.fit
    result = park.fitting.fit.fit(assemb, handler=handler, fitter=fitter)
    result.print_summary()
    outname = "ni_neutron+xray"
    numpy.savetxt("%s.xray.fit"%outname, zip(datX.fit_x,datX.calc_y))
    numpy.savetxt("%s.neutron.fit"%outname, zip(datN.fit_x,datN.calc_y))
    ofile = file("%s.res"%outname, 'w')
    result.print_summary(ofile)
    ofile.close()

    # Plot the fit
    diffX = datX.fit_y - datX.calc_y + 1.25*min(datX.fit_y)
    diffN = datN.fit_y - datN.calc_y + 1.25*min(datN.fit_y)
    offset = 20*int(1+max(datX.fit_y)/10)
    import pylab
    pylab.clf()
    pylab.plot(datX.fit_x, datX.fit_y, "bo", label = "x-ray data")
    pylab.plot(datX.fit_x, datX.calc_y, "r-", label = "x-ray fit", linewidth=2)
    pylab.plot(datX.fit_x, diffX, "g-", label = "x-ray difference", linewidth=2)
    pylab.plot(datN.fit_x, offset+datN.fit_y, "bo", label = "neutron data")
    pylab.plot(datN.fit_x, offset+datN.calc_y, "r-", label = "neutron fit", linewidth=2)
    pylab.plot(datN.fit_x, offset+diffN, "g-", label = "neutron difference", linewidth=2)
    pylab.legend()
    pylab.title("Park fit to nickel data using pdffit2 as calculator")
    pylab.xlabel("$r (\AA)$")
    pylab.ylabel("$G (\AA^{-2})$")
    pylab.xlim(0, 10)
    pylab.ylim(-15, 100)
    pylab.savefig("%s.fit.png"%outname)

    # Save the refined
    stru = structureFromPhase(phase)
    stru.write("%s.rstr"%outname, "pdffit")

    return


if __name__ == "__main__":

    fitNickelFromFile("ni-q27r100-neutron.gr", "N")
    #fitNickelFromFile("ni-q27r60-xray.gr", "X")
    #fitTwoNickel()

