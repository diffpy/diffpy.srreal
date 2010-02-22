#!/usr/bin/env python
"""\
This script partially demonstrates SrFit UC1-1 and UC1-2. An instance of
diffpy.Structure.Structure is created by reading strutural information from file
and is then converted to a srreal01.CrystalStructure instance that can be refined
by park.  Constraints on parameters are made using mechanisms provided by park.
Introspection capabilities from SrFit are not demonstrated in this script.

Alternate scenerios:
    The diffpy.Structure.Structure instance can be created manually.
"""

__id__ = "$Id$"

def fitNickelFromFile(dataname, stype):
    """Fit nickel data contained in a file.

    See the bottom of this file for invocation examples.
    """

    import park
    import numpy
    from diffpy.srreal01.pdf import PDFModel
    from diffpy.srreal01.pdf import phaseFromStructure
    from diffpy.srreal01.pdf import structureFromPhase

    outname = ".".join(dataname.split(".")[:-1])

    # Create a new refinement model. "NiModel" is the model name.
    model = PDFModel("NiModel")
    # stype is the scattering type, "X" for x-ray and "N" for neutron.
    model.stype = stype
    # Set the qmax cutoff for the pdf data. 
    model.qmax = 27.0

    # Use diffpy.Structure to read the structure from file. We're reading in a
    # structure file in "pdffit" format, but many other formats can be read.
    import diffpy.Structure
    stru = diffpy.Structure.Structure()
    stru.read("data/ni.stru", "pdffit")

    # Convert the structure to a CrystalStructure object that the PDFModel can
    # use.
    phase = phaseFromStructure(stru)

    # Add the phase to the model.
    model.addPhase(phase)

    # Load the data and select the data points that we want to fit
    from park.modelling.data import Data1D
    dat = Data1D(filename="data/"+dataname)
    # This selects every-other data point between r = 1.8 and r = 10.
    import bisect
    idxlo = bisect.bisect(dat.x, 1.8)
    idxhi = bisect.bisect(dat.x, 10)
    dat.select(range(idxlo, idxhi, 2))

    # Create the assembly, which associates the model with the data and
    # specifies which phases are part of the fit. The phase is put in the
    # assembly as well, but without data. This allows us to share the phase
    # between multiple models.
    assemb = park.Assembly([(model,dat), (phase,)])

    # Constrain parameters
    # This says to refine the data scale "dscale" between 0.2 and 2.0
    model.dscale.set((0.2, 2.0))
    # This sets lattice parameters b and c equal to a, and says to refine a
    # between 3.50 and 3.55 Angstroms.
    lat = phase.lattice
    lat.b.set( lat.a.path )
    lat.c.set( lat.a.path )
    lat.a.set((3.50, 3.55))
    # This constrains an atomic vibrational correlation factor, delta2, and
    # and an experimental resolution factor, qdamp.
    phase.delta2.set((0, 10))
    model.qdamp.set((0.0, 0.1))

    # Loop over atoms and constrain the thermal parameters to be isotropic.
    # This will be automated in the future.
    atoms = phase.getAtoms()
    u11path = atoms[0]["u11"].path
    for i, atom in enumerate(atoms):
        if i != 0:
            atom.u11.set(u11path)
        atom.u22.set(u11path)
        atom.u33.set(u11path)
    atoms[0].u11.set( (0.001, 0.01) )

    # Configure the fit.
    from park.fitting.fitresult import ConsoleUpdate
    from park.optim.fitmc import FitMC
    # Select the update method. This will print updates to the screen.
    handler = ConsoleUpdate(improvement_delta=0.1,progress_delta=1)
    # This uses the park Monte-carlo fitter, which performs multiple fits (in
    # this case, just 1) at multiple starting positions. With more start points,
    # we get closer to true global optimization.
    fitter = FitMC(start_points=1)


    # Run the fit and save the results
    print "Starting refinement of", dataname
    import park.fitting.fit
    result = park.fitting.fit.fit(assemb, handler=handler, fitter=fitter)
    result.print_summary()
    # Save the fit.
    numpy.savetxt("%s.fit"%outname, zip(dat.fit_x,dat.calc_y))
    # Save the output to file.
    ofile = file("%s.res"%outname, 'w')
    result.print_summary(ofile)
    ofile.close()

    # Save the refined structure
    stru = structureFromPhase(phase)
    stru.write("%s.rstr"%outname, "pdffit")
    return


def fitTwoNickel():
    """Fit the x-ray and neutron nickel data simultaneously."""

    import park
    import numpy
    from diffpy.srreal01.pdf import PDFModel
    from diffpy.srreal01.pdf import phaseFromStructure
    from diffpy.srreal01.pdf import structureFromPhase

    # Use diffpy.Structure to read the structure from file
    import diffpy.Structure
    stru = diffpy.Structure.Structure()
    stru.read("data/ni.stru", "pdffit")
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
    datN = Data1D(filename="data/ni-q27r100-neutron.gr")
    # This selects every-other dat1a point up to r = 10
    idxlo = bisect.bisect(datN.x, 1.8)
    idxhi = bisect.bisect(datN.x, 10)
    datN.select(range(idxlo, idxhi, 2))
    datX = Data1D(filename="data/ni-q27r60-xray.gr")
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

    return


if __name__ == "__main__":

    fitNickelFromFile("ni-q27r100-neutron.gr", "N")
    #fitNickelFromFile("ni-q27r60-xray.gr", "X")
    #fitTwoNickel()

