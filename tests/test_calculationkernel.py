#!/usr/bin/env python
"""Unit tests for diffpy.srreal.calculationkernel."""

import os
import pickle

import numpy
import pytest
from testutils import _maxNormDiff, loadDiffPyStructure

from diffpy.srreal.calculationkernel import (
    availableCalculationKernels,
    getCalculationKernel,
    setCalculationKernel,
)
from diffpy.srreal.pdfcalculator import DebyePDFCalculator, PDFCalculator
from diffpy.srreal.structureadapter import AtomicStructureAdapter

CUDA_KERNELS = ["cuda", "cuda-mixed"]


@pytest.fixture
def restore_kernel():
    """Leave the tests with the kernel that was active before them."""
    name = getCalculationKernel()
    yield name
    setCalculationKernel(name)
    return


@pytest.fixture
def cuda_device(restore_kernel):
    names = availableCalculationKernels()
    if "cuda" not in names:
        pytest.skip("CUDA kernels unavailable")
    setCalculationKernel(names[0])


def test_available(restore_kernel):
    names = availableCalculationKernels()
    assert len(names) == len(set(names))
    # CPU kernels come first, CUDA kernels are never the default.
    cpunames = [n for n in names if n not in CUDA_KERNELS]
    assert "scalar" == cpunames[-1]
    assert names == cpunames + CUDA_KERNELS or names == cpunames
    known = {"avx512", "avx2", "sse2", "neon", "scalar"}
    assert known.issuperset(cpunames)
    assert restore_kernel in names
    return


def test_set_kernel(restore_kernel):
    for name in availableCalculationKernels():
        setCalculationKernel(name)
        assert name == getCalculationKernel()
    return


def test_set_unavailable_kernel(restore_kernel):
    with pytest.raises(ValueError, match="scalar"):
        setCalculationKernel("invalid")
    with pytest.raises(ValueError):
        setCalculationKernel("")
    with pytest.raises(TypeError):
        setCalculationKernel(None)
    assert restore_kernel == getCalculationKernel()
    return


def test_cpu_calculation(restore_kernel):
    stru = loadDiffPyStructure("TiO2_rutile-fit.stru")
    calc = PDFCalculator(rmax=12, evaluatortype="BASIC")
    setCalculationKernel("scalar")
    reference = calc(stru)[1].copy()
    assert numpy.max(numpy.abs(reference)) > 0.1
    setCalculationKernel(availableCalculationKernels()[0])
    assert _maxNormDiff(reference, calc(stru)[1]) < 1e-10


def test_cropped_peak_on_absolute_grid(restore_kernel):
    # A single dimer gives an RDF of (r / distance) times its peak profile.
    distance, width = 2.0, 0.12
    atoms = AtomicStructureAdapter()
    atoms.setAtomArrays(
        ["Ni", "Ni"],
        [[0, 0, 0], [distance, 0, 0]],
        [1, 1],
        [False, False],
        numpy.zeros((2, 3, 3)),
    )
    calc = PDFCalculator(
        rmin=1.8,
        rmax=2.2,
        rstep=0.01,
        maxextension=0,
        peakwidthmodel="constant",
        peakprofile="croppedgaussian",
        peakprecision=0.5,
    )
    calc.peakwidthmodel.width = width
    for kernel in availableCalculationKernels():
        setCalculationKernel(kernel)
        r, _ = calc(atoms)
        expected = [
            x / distance * calc.peakprofile(x - distance, width) for x in r
        ]
        rtol = 1e-6 if kernel == "cuda-mixed" else 1e-12
        numpy.testing.assert_allclose(
            calc.rdf, expected, rtol=rtol, atol=0, err_msg=kernel
        )


def test_cuda_missing(restore_kernel):
    names = availableCalculationKernels()
    if "cuda" in names and os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        pytest.skip("CUDA kernels available")
    for name in CUDA_KERNELS:
        assert name not in names
        with pytest.raises(ValueError, match=f"'{name}' is not available: .+"):
            setCalculationKernel(name)
    assert restore_kernel == getCalculationKernel()
    return


@pytest.mark.parametrize("cls", [PDFCalculator, DebyePDFCalculator])
def test_cuda(cuda_device, cls):
    rutile = loadDiffPyStructure("rutile.cif")
    cpu = cls(rmax=9)
    r, g = cpu(rutile)
    setCalculationKernel("cuda")
    calc = cls(rmax=9)
    r, gcuda = calc(rutile)
    assert _maxNormDiff(g, gcuda) < 1e-11
    assert _maxNormDiff(cpu.value, calc.value) < 1e-12
    # the sums in single precision are other ones
    setCalculationKernel("cuda-mixed")
    mixed = cls(rmax=9)
    r, gmixed = mixed(rutile)
    assert 0 < _maxNormDiff(cpu.value, mixed.value) < 1e-5
    assert _maxNormDiff(g, gmixed) < 1e-4


@pytest.mark.parametrize("cls", [PDFCalculator, DebyePDFCalculator])
def test_pickled_mixed_precision(cuda_device, cls):
    stru = loadDiffPyStructure("rutile.cif")
    setCalculationKernel("cuda-mixed")
    calc = cls(rmax=9)
    _, mixed = calc(stru)
    restored = pickle.loads(pickle.dumps(calc))

    setCalculationKernel("cuda")
    _, expected = cls(rmax=9)(stru)
    assert not numpy.array_equal(mixed, expected)
    numpy.testing.assert_array_equal(restored(stru)[1], expected)


def test_cuda_in_refinement(cuda_device):
    """Check refinements with numerical derivatives on the device."""
    least_squares = pytest.importorskip("scipy.optimize").least_squares
    stru = loadDiffPyStructure("Ni.stru")
    ptrue = numpy.array([3.53, 0.006, 0.8, 2.5])

    def pdf(calc, p):
        a, uiso, scale, delta2 = p
        stru.lattice.set_latt_parms(a, a, a, 90, 90, 90)
        stru.Uisoequiv = uiso
        calc.scale = scale
        calc.delta2 = delta2
        return calc(stru)[1]

    gobs = pdf(PDFCalculator(rmin=1.5, rmax=9), ptrue)
    # The default step, 1.5e-8 of a parameter, changes the peaks by less
    # than the rounding errors of the mixed precision.
    for kernel, step, tolerance in (
        ("cuda", None, 1e-7),
        ("cuda-mixed", 1e-4, 1e-5),
    ):
        setCalculationKernel(kernel)
        calc = PDFCalculator(rmin=1.5, rmax=9)
        fit = least_squares(
            lambda p: pdf(calc, p) - gobs,
            [3.52, 0.005, 1.0, 1.5],
            x_scale="jac",
            diff_step=step,
        )
        assert numpy.allclose(fit.x, ptrue, rtol=tolerance)
    return
