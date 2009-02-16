#!/usr/bin/env python
"""This module contains converters between diffpy.Structure data objects and the
data objects needed by pdfmodel.
"""

__id__ = "$Id$"


phasenum = 0

def phaseFromStructure(S, name=None):
    """Get CrystalPhase object from diffpy.Structure.Structure object."""

    global phasenum
    phasenum += 1

    # Start with a phase
    import phases
    if name is None: name = "phase%i"%phasenum
    _p = phases.CrystalPhase(name)
    import diffpy.Structure
    _p._stru = diffpy.Structure.Structure(S)

    # Start with the lattice
    import lattice
    _l = lattice.Lattice()
    parlist = ["a", "b", "c", "alpha", "beta", "gamma"]
    for name in parlist:
        par = _l[name]
        par.set(getattr(S.lattice, name))
    _p.setLattice(_l)

    # Now the atoms
    import atoms
    namedict = {}
    for a in S:
        # Name the atom as the element plus a number.
        name = a.element.title()

        # Try to get unique names. This will mangle current names.
        if name not in namedict:
            namedict[name] = 0
        namedict[name] += 1
        name += str(namedict[name])

        _a = atoms.Atom(name)
        _a.element = a.element
        _a["x"].set(a.xyz[0])
        _a["y"].set(a.xyz[1])
        _a["z"].set(a.xyz[2])
        _a["occ"].set(a.occupancy)
        _a["u11"].set(a.U11)
        _a["u22"].set(a.U22)
        _a["u33"].set(a.U33)
        _a["u12"].set(a.U12)
        _a["u23"].set(a.U23)
        _a["u13"].set(a.U13)

        _p.addAtom(_a)

    return _p

def structureFromPhase(p):
    """Get diffpy.Structure.Structure object from CrystalPhase object.

    This ignores the _stru member of phase.
    """

    import diffpy.Structure
    # Start with the lattice
    parlist = ["a", "b", "c", "alpha", "beta", "gamma"]
    vallist = []
    _l = p.getLattice()
    for pname in parlist:
        vallist.append( _l[pname].get() )
    L = diffpy.Structure.Lattice(*vallist)

    # Make the structure
    S = diffpy.Structure.Structure(lattice=L)
    S.title = p.name

    # Now the atoms
    import numpy
    for _a in p.getAtoms():

        element = _a.element
        x = _a["x"].get()
        y = _a["y"].get()
        z = _a["z"].get()
        xyz = numpy.array([x,y,z],dtype=float)
        occ = _a["occ"].get()
        U11 = _a["u11"].get()
        U22 = _a["u22"].get()
        U33 = _a["u33"].get()
        U12 = _a["u12"].get()
        U23 = _a["u23"].get()
        U13 = _a["u13"].get()
        U = numpy.array( [
            [U11, U12, U13],
            [U12, U22, U23],
            [U13, U23, U33]],
            dtype = float)
        S.addNewAtom(atype = element, xyz = xyz, U = U)

    return S

def _testConverters():
    """Assert that stru == structureFromPhase(phaseFromStructure(stru))"""
    from diffpy.Structure import Structure
    S1 = Structure()
    S1.read("tests/testdata/ni.stru", "pdffit")

    p = phaseFromStructure(S1)
    S2 = structureFromPhase(p)

    assert(S1.lattice.abcABG() == S2.lattice.abcABG())
    assert(len(S1) == len(S2))
    for i in range(len(S1)):
        assert(S1[i].element == S2[i].element)
        assert(S1[i].occupancy == S2[i].occupancy)
        assert((S1[i].xyz == S2[i].xyz).all())
        assert((S1[i].U == S2[i].U).all())

    return

if __name__ == "__main__":
    _testConverters()
