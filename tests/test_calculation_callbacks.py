"""Calculation hooks must execute each contribution exactly once."""

import numpy as np
import pytest

import diffpy.srreal.srreal_ext as ext


def test_parallel_data_super_calls_override_once():
    class Derived(ext.PairQuantity):
        calls = 0

        def _getParallelData(self):
            self.calls += 1
            return b"custom:" + super()._getParallelData()

    calc = Derived()
    calc._getParallelData()
    assert calc.calls == 1


@pytest.mark.parametrize(
    "method",
    [
        "_resizeValue",
        "_resetValue",
        "_configureBondGenerator",
        "_addPairContribution",
        "_executeParallelMerge",
        "_finishValue",
        "_stashPartialValue",
        "_restorePartialValue",
    ],
)
def test_pairquantity_super_and_explicit_base_do_not_repeat_override(method):
    class Derived(ext.PairQuantity):
        calls = 0

    def override(self, *args):
        self.calls += 1
        return getattr(super(Derived, self), method)(*args)

    setattr(Derived, method, override)
    calc = Derived()
    generator = ext.AtomicStructureAdapter().createBondGenerator()
    args = {
        "_resizeValue": (1,),
        "_configureBondGenerator": (generator,),
        "_addPairContribution": (generator, 1),
        "_executeParallelMerge": (calc._getParallelData(),),
    }.get(method, ())
    for function in (
        getattr(calc, method),
        lambda *args: getattr(ext.PairQuantity, method)(calc, *args),
    ):
        if method in ("_stashPartialValue", "_restorePartialValue"):
            with pytest.raises(RuntimeError, match="not defined"):
                function(*args)
        else:
            function(*args)
        assert calc.calls == 1


def test_finish_value_customization_applies_once_from_python_and_cpp():
    class Derived(ext.PairQuantity):
        calls = 0

        def _finishValue(self):
            super()._finishValue()
            self.calls += 1
            self._value[0] *= 2

    calc = Derived()
    calc._resizeValue(1)
    calc._value[0] = 1
    calc._finishValue()
    assert calc._value[0] == 2
    assert calc.calls == 1
    calc.eval()
    assert calc.calls == 2


@pytest.mark.parametrize(
    "cls",
    [
        ext.StructureAdapter,
        *[
            ext.AtomicStructureAdapter,
            ext.PeriodicStructureAdapter,
            ext.CrystalStructureAdapter,
        ],
    ],
)
def test_custom_config_super_runs_once(cls):
    class Derived(cls):
        calls = 0

        def countSites(self):
            return 0

        def _customPQConfig(self, calc):
            self.calls += 1
            super()._customPQConfig(calc)

    adapter = Derived()
    calc = ext.PairQuantity()
    adapter._customPQConfig(calc)
    assert adapter.calls == 1
    calc.setStructure(adapter)
    assert adapter.calls == 2


class CallbackAnisotropyAdapter(ext.StructureAdapter):
    anisotropy = False

    def clone(self):
        return self

    def countSites(self):
        return 2

    def createBondGenerator(self):
        return ext.BaseBondGenerator(self)

    def siteCartesianPosition(self, index):
        return [index, 0, 0]

    def siteAtomType(self, index):
        return "C"

    def siteAnisotropy(self, index):
        return self.anisotropy

    def siteCartesianUij(self, index):
        return np.eye(3) * 0.005


@pytest.mark.parametrize(
    "flag", [False, True, 0, 1, np.bool_(True), np.int64(0)]
)
def test_python_adapter_truth_values_produce_the_same_pdf(flag):
    adapter = CallbackAnisotropyAdapter()
    adapter.anisotropy = flag
    actual = ext.PDFCalculator(rmax=2)(adapter)
    adapter.anisotropy = bool(flag)
    expected = ext.PDFCalculator(rmax=2)(adapter)
    np.testing.assert_allclose(actual, expected)
