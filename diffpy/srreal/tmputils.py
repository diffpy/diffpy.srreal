import numpy

def createSuperCell(stru, mno):
    """Perform supercell expansion for this structure and adjust
    constraints for positions and lattice parameters.  New lattice
    parameters are multiplied and fractional coordinates divided by
    corresponding multiplier.  New atoms are grouped with their source
    in the original cell.
    
    mno -- tuple or list of three positive integer cell multipliers along
    the a, b, c axis
    """
    from diffpy.Structure import Structure, Atom
    if min(mno) < 1:
        raise ValueError, "mno must contain 3 positive integers"
    # back to business
    mnofloats = numpy.array(mno[:3], dtype=float)
    ijklist = [(i,j,k) for i in range(mno[0])
                for j in range(mno[1]) for k in range(mno[2])]
    # build a list of new atoms
    newatoms = []
    for a in stru:
        for ijk in ijklist:
            adup = Atom(a)
            adup.xyz = (a.xyz + ijk)/mnofloats
            newatoms.append(adup)
    # replace original atoms with newatoms
    supercell = Structure(stru)
    supercell[:] = newatoms
    supercell.lattice.setLatPar(
            a=mno[0]*stru.lattice.a,
            b=mno[1]*stru.lattice.b,
            c=mno[2]*stru.lattice.c )
    return supercell


class CostCalculator:

    def __init__(self, y0):
        self.y0 = y0[:]
        return

    def __call__(self, pairhist):
        cost = sum([(y0 - y1)**2 for y0, y1 in zip(self.y0, pairhist.y()) ])
        cost /= pairhist.countBars()
        return cost
