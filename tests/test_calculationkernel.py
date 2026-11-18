#!/usr/bin/env python
"""Unit tests for diffpy.srreal.calculationkernel."""

import numpy
import pytest
from testutils import _maxNormDiff, loadDiffPyStructure

from diffpy.srreal.calculationkernel import (
    availableCalculationKernels,
    getCalculationKernel,
    setCalculationKernel,
)
from diffpy.srreal.pdfcalculator import PDFCalculator


@pytest.fixture
def restore_kernel():
    """Leave the tests with the kernel that was active before them."""
    name = getCalculationKernel()
    yield name
    setCalculationKernel(name)
    return


def test_available(restore_kernel):
    names = availableCalculationKernels()
    assert "scalar" == names[-1]
    assert len(names) == len(set(names))
    known = {"avx512", "avx2", "sse2", "neon", "scalar"}
    assert known.issuperset(names)
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
