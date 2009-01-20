#!/usr/bin/env python
"""The PDFModel class."""

__id__ = "$Id:"

import park
from dynamicmodel import DynamicModel
import numpy
from parameters import PDFTopLevelParameter

class PDFModel(DynamicModel):
    """park.Model for PDF refinement"""

    def __init__(self, name="", **kw):
        """initialize"""
        DynamicModel.__init__(self, name, **kw)
        self.stype = "X"
        self.qmax = 0
        self.parameterset = park.ParameterSet(name=name)

        # The engine
        self._engine = None

        # phases
        self._phases = []

        # Flag indicating whether we need to interpolate
        self._interp = False

        # The x-values from the engine needed for interpolation
        self._ex = []

        # Add the top-level parameters
        self.parameterset.append(PDFTopLevelParameter("dscale", 1.0))
        self.parameterset.append(PDFTopLevelParameter("qdamp", 0.0))
        self.parameterset.append(PDFTopLevelParameter("qbroad", 0.0))

        return

    def eval(self, x):
        """Evaluate the PDF function."""
        x = numpy.asarray(x)
        p = self._getEngine(x)
        #print "starting calculation"
        p.calc()
        y = numpy.asarray(p.getpdf_fit())
        #print "ending calculation"
        # The engine works on a fixed-space grid. We must see if this is the
        # same as the requested grid. If not, then we interpolate on the
        # requesteed grid.
        if self._interp:
            # FIXME - the engine can't handle this right now. There seems to be
            # a bug in getR.
            raise NotImplementedError("Can't interpolate")
        #print self._engine
        return y

    def addPhase(self, phase):
        """Add a phase to the model.

        Do not add atoms to a phase after it has been given to the model, as
        there is no way to add more atoms to the engine once a structure has
        been loaded.
        
        Arguments
        phase   --  phase to add
        """
        self._phases.append(phase)
        return

    def getNumPhases(self):
        """Get a phase count."""
        return len(self._phases)

    def getPhases(self):
        """Get the phases."""
        return self._phases[:]

    ## Internals. Users ignore this stuff. ##

    def _getEngine(self, x):
        """Get a stored engine by key.

        If the requested engine does not exist, this will create a new engine
        and inform all parameters tied to this model about that engine.
        """
        if not numpy.array_equal(x, self._ex):
            self._setupEngine(x)
        return self._engine

    def _setupEngine(self, x):
        """Set up the engine for the calculation."""

        ## Create the engine
        self._engine = _createEngine()
        self._ex = numpy.array(x)

        for i, phase in enumerate(self._phases):
            # If the phase was loaded from a structure, this will save some
            # time when the phase is added to the engine.
            if phase._stru is None:
                import converters
                phase._stru = converters.structureFromPhase(phase)
            self._engine.add_structure(phase._stru)
            phase._addEngine(self._engine, i+1)

        # allocate room in the engine for the calculation
        stype = self.stype
        qmax = self.qmax
        rmin = x[0]
        rmax = x[-1]
        bin = len(x)
        self._engine.alloc(stype, qmax, 0, rmin, rmax, bin)

        # We have to interpolate if the requested calculation points are not
        # equidistant.
        dx = x[1] - x[0]
        toler = 1e-6
        for i in range(2, len(x)):
            if abs(dx - (x[i]-x[i-1])) > toler:
                self._interp = True
                break;

        ## Link the other parameters to this engine.
        for pname in ["dscale", "qdamp", "qbroad"]:
            self.parameterset[pname]._addEngine(self._engine, pname)

        return

# End class PDFModel

def _createEngine():
    """Create the pdffit2 engine instance."""

    from diffpy.pdffit2 import PdfFit
    from diffpy.pdffit2 import redirect_stdout

    class CustomPdfFit(PdfFit):
        """A PdfFit class that works well as a function calculator.
        
        This class suppresses output.
        """

        def __init__(self):
            import os
            redirect_stdout(os.tmpfile())
            PdfFit.__init__(self)

    p = CustomPdfFit()
    return p

## Testing code

def testTwoDataOneModel():

    import park

    m = PDFModel("NiX")
    m.stype = "X"
    m.qmax = 27.0

    m2 = PDFModel("NiN")
    m2.stype = "N"
    m2.qmax = 27.0

    import diffpy.Structure
    S = diffpy.Structure.Structure()
    S.read("tests/testdata/ni.stru", "pdffit")
    import converters
    p = converters.phaseFromStructure(S)

    m.addPhase(p)
    m2.addPhase(p)

    from park.modelling.data import Data1D
    d = Data1D(filename="tests/testdata/ni-q27r60-xray.gr")

    import bisect
    idx = bisect.bisect(d.x, 10)
    d.select(range(0, idx, 2))

    d2 = Data1D(filename="tests/testdata/ni-q27r100-neutron.gr")
    d2.select(range(0, idx, 4))

    a = park.Assembly([(m,d), (m2,d2), (p,)])

    # Data parameters
    m.dscale.set((0.2, 2.0))
    m.qdamp.set((0.0, 0.1))
    m2.dscale.set((0.2, 2.0))
    m2.qdamp.set((0.0, 0.1))

    # Phase parameters
    lat = p.getLattice()
    lat.b.set(lat.a.path)
    lat.c.set(lat.a.path)
    lat.a.set((3.5,3.55))

    p.delta2.set((0, 10))

    atoms = p.getAtoms()
    a1 = atoms[0]
    a1.u11.set( (0.001, 0.01) )
    for i, atom in enumerate(atoms):
        if i != 0:
            atom.u11.set(a1.u11.path)
        atom.u22.set(a1.u11.path)
        atom.u33.set(a1.u11.path)

    from park.fitting.fitresult import ConsoleUpdate
    handler= ConsoleUpdate(improvement_delta=0.1,progress_delta=1)
    from park.optim.fitmc import FitMC
    fitter = FitMC(start_points=1)

    # Now add some parameters
    import park.fitting.fit
    result = park.fitting.fit.fit(a, handler=handler, fitter=fitter)
    result.print_summary()

    numpy.savetxt("ni.pdf.calc", zip(d.fit_x,d.calc_y))
    return

def testSimpleFit(datafile="ni-q27r60-xray.gr"):

    import park

    m = PDFModel("NiModel")
    m.qmax = 45.0

    import diffpy.Structure
    S = diffpy.Structure.Structure()
    S.read("tests/testdata/ni.stru", "pdffit")
    import converters
    p = converters.phaseFromStructure(S)

    m.addPhase(p)

    from park.modelling.data import Data1D
    d = Data1D(filename="tests/testdata/%s"%datafile)

    import bisect
    idx = bisect.bisect(d.x, 10)
    d.select(range(0,idx,2))

    a = park.Assembly([(m,d),(p,)])

    m.dscale.set((0.2, 2.0))
    lat = p.getLattice()
    lat.b.set(lat.a.path)
    lat.c.set(lat.a.path)
    lat.a.set((3.5,3.55))

    p.delta2.set((0, 10))
    m.qdamp.set((0.0, 0.1))

    atoms = p.getAtoms()
    a1 = atoms[0]
    a1.u11.set( (0.001, 0.01) )
    for i, atom in enumerate(atoms):
        if i != 0:
            atom.u11.set(a1.u11.path)
        atom.u22.set(a1.u11.path)
        atom.u33.set(a1.u11.path)

    from park.fitting.fitresult import ConsoleUpdate
    handler= ConsoleUpdate(improvement_delta=0.1,progress_delta=1)
    from park.optim.fitmc import FitMC
    fitter = FitMC(start_points=1)

    # Now run the fit
    import park.fitting.fit
    result = park.fitting.fit.fit(a, handler=handler, fitter=fitter)
    result.print_summary()

    numpy.savetxt("ni.pdf.calc", zip(d.fit_x,d.calc_y))
    return


def testModel():

    C = PDFModel("C")

    import numpy
    x = numpy.arange(1, 20, 0.05)
    import diffpy.Structure
    S = diffpy.Structure.Structure()
    S.read("tests/testdata/ni.stru", "pdffit")
    import converters
    p = converters.phaseFromStructure(S)

    C.addPhase(p)
    # Test access 
    print C.parameterset["dscale"]
    print C["dscale"]
    print C.dscale
    print C.name
    print p.lattice
    print p.getLattice()
    print p.getAtom("Ni1")
    print p.Ni1
    y = C.eval(x)

    numpy.savetxt("ni.pdf.calc", zip(x,y))

if __name__ == "__main__":

    #testModel()
    testSimpleFit()
    testSimpleFit("ni-q27r100-neutron.gr")
    testTwoDataOneModel()
