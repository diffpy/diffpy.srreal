#!/usr/bin/env python
"""Unit tests for the batched pair sums of PDFCalculator and
DebyePDFCalculator, compared to pair-by-pair sums with a Python peak
width model."""

import pickle

import numpy
import pytest
from testutils import (
    DerivedPeriodicStructureAdapter,
    _maxNormDiff,
    loadCrystalStructureAdapter,
    loadDiffPyStructure,
)

from diffpy.srreal.pdfcalculator import DebyePDFCalculator, PDFCalculator
from diffpy.srreal.peakwidthmodel import PeakWidthModel
from diffpy.srreal.structureadapter import (
    AtomicStructureAdapter,
    createStructureAdapter,
    nometa,
)

CALCULATORS = [PDFCalculator, DebyePDFCalculator]


class ScaledWidth(PeakWidthModel):
    """Peak widths of another model times a factor."""

    def __init__(self, model, factor=1):
        PeakWidthModel.__init__(self)
        self.model = model
        self.factor = factor
        return

    def type(self):
        return "scaledwidth"

    def create(self):
        return ScaledWidth(self.model.create(), self.factor)

    def clone(self):
        return ScaledWidth(self.model.clone(), self.factor)

    def calculate(self, bnds):
        return self.factor * self.model.calculate(bnds)

    def maxWidth(self, stru, rmin, rmax):
        return self.factor * self.model.maxWidth(stru, rmin, rmax)


def pair_by_pair(calc):
    """Return a copy of calc that adds its pairs one by one."""
    rv = calc.copy()
    rv.peakwidthmodel = ScaledWidth(calc.peakwidthmodel)
    return rv


@pytest.fixture(scope="module")
def rutile():
    return loadDiffPyStructure("rutile.cif")


@pytest.mark.parametrize("calculator", CALCULATORS)
def test_structures(calculator, rutile):
    """Check the results for every kind of structure object."""
    calc = calculator(rmax=9)
    onebyone = pair_by_pair(calc)
    adapter = createStructureAdapter(rutile)
    structures = [
        rutile,
        adapter,
        nometa(rutile),
        loadCrystalStructureAdapter("rutile.cif"),
    ]
    for stru in structures:
        r, g = calc(stru)
        r, g1 = onebyone(stru)
        assert numpy.any(g)
        assert _maxNormDiff(g1, g) < 1e-12
    # An adapter class defined in Python gives its bonds itself.
    derived = DerivedPeriodicStructureAdapter()
    derived.setLatPar(*adapter.getLatPar())
    derived.setAtomArrays(*adapter.getAtomArrays())
    r, g = calc(derived)
    r, g1 = onebyone(derived)
    assert _maxNormDiff(g1, g) < 1e-12
    r, gadapter = calc(adapter)
    assert _maxNormDiff(gadapter, g) < 1e-12
    return


@pytest.mark.parametrize("calculator", CALCULATORS)
def test_python_peak_width_model(calculator, rutile):
    calc = calculator(rmax=9, peakwidthmodel="constant")
    calc.peakwidthmodel.width = 0.2
    _, g = calc(rutile)
    wide = calc.copy()
    wide.peakwidthmodel.width = 0.4
    r, gwide = wide(rutile)
    assert not numpy.allclose(g, gwide)
    calc.peakwidthmodel = ScaledWidth(calc.peakwidthmodel, 2)
    r, gdouble = calc(rutile)
    assert _maxNormDiff(gwide, gdouble) < 1e-12
    return


@pytest.mark.parametrize("calculator", CALCULATORS)
def test_masks(calculator, rutile):
    calc = calculator(rmax=9)
    r, gall = calc(rutile)
    calc.setTypeMask("Ti", "O", False)
    r, g = calc(rutile)
    r, g1 = pair_by_pair(calc)(rutile)
    assert _maxNormDiff(g1, g) < 1e-12
    assert not numpy.allclose(g, gall)
    return


def test_nearly_equal_distances_straddle_cropped_peak_boundary():
    # One dimer contributes at 2.10 angstroms and the other is cropped.
    xyz = [[0, 0, 0], [2, 0, 0], [0, 30, 0], [numpy.nextafter(2, 3), 30, 0]]
    stru = AtomicStructureAdapter()
    stru.setAtomArrays(
        ["H"] * 4, xyz, numpy.ones(4), numpy.zeros(4), numpy.zeros((4, 3, 3))
    )
    calc = PDFCalculator(rmax=5, rstep=0.01, maxextension=0)
    calc.peakwidthmodel = "constant"
    calc.peakwidthmodel.width = 0.2
    calc.peakprofile = "croppedgaussian"
    calc.peakprecision = 0.5
    reference = pair_by_pair(calc)
    reference(stru)
    calc(stru)
    assert reference.rdf[210] > 0
    numpy.testing.assert_allclose(
        calc.rdf, reference.rdf, rtol=1e-12, atol=1e-14
    )


def test_fast_updates():
    """Check the evaluation used after the change of one atom."""
    ni = loadDiffPyStructure("Ni.stru")
    calc = PDFCalculator(rmax=9)
    # The four atoms of Ni take less work when added all at once.
    for c, used in ((calc, "BASIC"), (pair_by_pair(calc), "OPTIMIZED")):
        basic = c.copy()
        basic.evaluatortype = "BASIC"
        c(ni)
        moved = ni.copy()
        moved[1].xyz[0] += 0.01
        r, g = c(moved)
        assert used == c.evaluatortypeused
        r, gb = basic(moved)
        assert numpy.allclose(gb, g, rtol=0, atol=1e-10)
    return


@pytest.mark.parametrize("calculator", CALCULATORS)
def test_copies(calculator, rutile):
    calc = calculator(rmax=9)
    r, g = calc(rutile)
    for other in (calc.copy(), pickle.loads(pickle.dumps(calc))):
        assert numpy.array_equal(g, other.pdf)
        other.evaluatortype = "BASIC"
        r, g1 = other(rutile)
        assert numpy.array_equal(g, g1)
    return


@pytest.mark.parametrize("calculator", CALCULATORS)
def test_large_cpu_batch(calculator):
    # Unequal occupancies and mixed atom types exercise the Debye cutoff.
    # The cluster has enough pairs to cross batch and worker boundaries.
    n = 377
    rng = numpy.random.default_rng(151)
    atoms = AtomicStructureAdapter()
    atoms.setAtomArrays(
        ["Ni", "O"] * (n // 2) + ["Ni"],
        rng.uniform(0, 12, (n, 3)),
        rng.choice([0.4, 0.7, 1.0], n),
        numpy.zeros(n),
        numpy.tile(numpy.eye(3) * 0.005, (n, 1, 1)),
    )
    calc = calculator(rmax=14, qmax=18)
    calc.evaluatortype = "BASIC"
    calc.setTypeMask("Ni", "O", False)
    _, reference = pair_by_pair(calc)(atoms)
    _, batch = calc(atoms)
    batch = batch.copy()
    assert numpy.any(reference)
    assert _maxNormDiff(reference, batch) < 1e-12
    numpy.testing.assert_array_equal(calc(atoms)[1], batch)
