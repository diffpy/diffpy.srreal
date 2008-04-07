################################################################################
#
#   PDF Refinement of Ni data.
#
################################################################################
__id__ = "$Id$"

from PDFAPI     import *

# Fit
fit   = Fit()
#from RefinementAPI.park.ParkOptimizer import ParkOptimizer
#fit.setOptimizer(ParkOptimizer())
# Set the number of refinement cycles
fit.setNumCycles(-1)

# set component
comp = PDFComponent()
comp.setName("Ni")
fit.addComponent(comp)

### Instantiate the pattern
pat = PDFData()
comp.setData(pat)
pat.loadData("ni.dat", PDFParser())
pat.setQmax(45.0)
pat.setScatteringType('N')
pat.qdamp = 0.001
fit.mapVP("v_qdamp", pat, "qdamp")

# setup the excluded region
comp.setFitRange(1.5, 15.0, 0.05)

### Instantiate the phase
pha = CrystalPhase()
pha.setName("Ni")
comp.addPhase(pha)
fit.mapVP("v_weight", pha, "weight")

# Create a lattice for the phase
lat = Lattice()
pha.setLattice(lat)
lat.a = lat.b = lat.c = 3.524
fit.mapVP("v_a", lat, "a")
fit.mapVP("v_a", lat, "b")
fit.mapVP("v_a", lat, "c")
# Add some atom
for i in range(4):
    a = Atom("Ni")
    a.x = a.y = a.z = 0
    a.setADP( IsotropicAtomicDisplacementFactor() )
    fit.mapVP("v_biso", a.getADP(), "Biso")
    fit.guessV("v_biso", 0.60)
    pha.addAtom(a)
pha.getAtom(1).y = pha.getAtom(1).z = 0.5
pha.getAtom(2).x = pha.getAtom(2).z = 0.5
pha.getAtom(3).x = pha.getAtom(3).y = 0.5
fit.mapVP("v_delta2", pha, "delta2")


# Fitting
fit.refine()
fit.printResults()

from pylab import plot, show
x0,y0,u0 = comp.getDataArrays()
x1,y1 = comp.getFitArrays()
from RefinementAPI.utils import rebinArray
y0i = rebinArray(y0, x0, x1)
plot(x1, y0i, x1, y1)
show()
