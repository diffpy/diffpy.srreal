"""Calculation hooks must execute each contribution exactly once."""

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
