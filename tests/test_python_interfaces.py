"""Public Python entry points used by calculations and saved objects."""

import gc
import pickle
import subprocess
import sys
from importlib import import_module

import numpy as np
import pytest
from test_structure_lifetimes import make_structure

from diffpy.srreal import srreal_ext as ext

CALCULATOR_MODULES = {
    "PDFCalculator": "pdfcalculator",
    "DebyePDFCalculator": "pdfcalculator",
    "BondCalculator": "bondcalculator",
    "BVSCalculator": "bvscalculator",
    "OverlapCalculator": "overlapcalculator",
}


@pytest.mark.parametrize("name,module", CALCULATOR_MODULES.items())
def test_extension_calculator_public_interface(name, module):
    public_class = getattr(import_module("diffpy.srreal." + module), name)
    assert getattr(ext, name) is public_class
    calc = getattr(ext, name)(rmax=3)
    result = calc(make_structure(), rmax=4)
    assert calc.rmax == 4
    assert isinstance(result, (tuple, np.ndarray))
    restored = pickle.loads(pickle.dumps(calc))
    assert type(restored) is public_class
    np.testing.assert_allclose(restored(), result)


@pytest.mark.parametrize(
    "first_import",
    [
        "srreal_ext",
        "structureadapter",
        *sorted(set(CALCULATOR_MODULES.values())),
    ],
)
def test_calculator_exports_for_each_import_order(first_import):
    code = f"""
from importlib import import_module
import_module('diffpy.srreal.' + {first_import!r})
from diffpy.srreal import srreal_ext as ext
for name, module in {CALCULATOR_MODULES!r}.items():
    cls = getattr(import_module('diffpy.srreal.' + module), name)
    assert getattr(ext, name) is cls
    calc = getattr(ext, name)(rmax=3)
    calc()
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True
    )


@pytest.mark.parametrize(
    "cls",
    [
        ext.AtomicStructureAdapter,
        ext.PeriodicStructureAdapter,
        ext.CrystalStructureAdapter,
    ],
)
def test_direct_diff_retains_both_independent_structures(cls):
    first, second = make_structure(cls), make_structure(cls)
    second[0].xc = 10
    diff = first.diff(second)
    assert diff.stru0 is first
    assert diff.stru1 is second
    assert diff.pop0 == diff.add1 == [0]
    del first, second
    gc.collect()
    assert diff.stru0[0].xc == 1
    assert diff.stru1[0].xc == 10


@pytest.mark.parametrize(
    "cls",
    [
        ext.StructureAdapter,
        ext.AtomicStructureAdapter,
        ext.PeriodicStructureAdapter,
        ext.CrystalStructureAdapter,
    ],
)
def test_diff_override_can_call_base(cls):
    class Derived(cls):
        calls = 0

        def countSites(self):
            return 0

        def diff(self, other):
            self.calls += 1
            return super().diff(other)

    first, second = Derived(), Derived()
    diff = first.diff(second)
    assert first.calls == 1
    assert diff.stru0 is first
    assert diff.stru1 is second
    cls.diff(first, second)
    assert first.calls == 1


@pytest.mark.parametrize(
    "factory,method,prefix",
    [
        (ext.LinearBaseline, "__call__", ()),
        (ext.ScaleEnvelope, "__call__", ()),
        (ext.SFTXray, "lookup", ("C",)),
    ],
)
def test_scalar_and_array_return_types(factory, method, prefix):
    function = getattr(factory(), method)
    for value in [2, 2.0, np.float64(2)]:
        assert isinstance(function(*prefix, value), float)
    for value in [np.float32(2), np.int64(2), np.array(2.0), [2.0]]:
        result = function(*prefix, value)
        assert isinstance(result, np.ndarray)
        assert result.shape == np.asarray(value).shape
