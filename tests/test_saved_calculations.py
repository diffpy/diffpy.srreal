"""Saved calculations retain structures, components, and Python
state."""

import base64
import copy
import json
import pickle
import struct
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pytest
from test_python_interfaces import CALCULATOR_MODULES
from test_structure_lifetimes import make_structure

from diffpy.srreal import srreal_ext as ext
from diffpy.srreal.pdfcalculator import PDFCalculator


class NamedAdapter(ext.AtomicStructureAdapter):
    def __init__(self, label):
        super().__init__()
        self.label = label

    def __getinitargs__(self):
        return (self.label,)


class NamedProfile(ext.GaussianProfile):
    def __init__(self, precision):
        super().__init__()
        self.peakprecision = precision

    def __getinitargs__(self):
        return (self.peakprecision,)


class CalculatorWithState(PDFCalculator):
    __getstate_manages_dict__ = True

    def __init__(self, label):
        super().__init__(rmax=3)
        self.label = label

    def __getinitargs__(self):
        return (self.label,)

    def __getstate__(self):
        return super().__getstate__(), self.__dict__

    def __setstate__(self, state):
        super().__setstate__(state[0])
        self.__dict__.update(state[1])


@pytest.mark.parametrize(
    "clone",
    [copy.copy, copy.deepcopy, lambda o: pickle.loads(pickle.dumps(o))],
)
def test_subclass_pickle_hooks(clone):
    adapter = NamedAdapter("sample")
    adapter.append(make_structure()[2])
    restored = clone(adapter)
    assert type(restored) is NamedAdapter
    assert restored.label == "sample"
    assert restored[0].xc == 3
    assert clone(NamedProfile(1e-7)).peakprecision == 1e-7
    calc = CalculatorWithState("saved")
    calc.setStructure(adapter)
    restored = clone(calc)
    assert type(restored) is CalculatorWithState
    assert restored.label == "saved"
    assert restored.rmax == 3
    assert restored.getStructure()[0].xc == 3


def test_unmanaged_dictionary_is_still_rejected():
    calc = PDFCalculator()
    calc.label = "not serialized"
    with pytest.raises(RuntimeError, match="Incomplete pickle support"):
        pickle.dumps(calc)
    assert calc.getStructure() is not None


_legacy = json.loads(
    (Path(__file__).parent / "testdata/boost_python_pickles.json").read_text()
)


@pytest.fixture(scope="module")
def legacy_component_registry():
    # Registered-function pickles require their type to be registered in the
    # receiving process under both bindings.
    from test_pdfbaseline import parabola_baseline
    from test_pdfenvelope import parabola_envelope

    from diffpy.srreal.pdfbaseline import makePDFBaseline
    from diffpy.srreal.pdfenvelope import makePDFEnvelope

    makePDFBaseline("legacy_parabola", parabola_baseline, a=1, b=2, c=3)
    makePDFEnvelope("legacy_envelope", parabola_envelope, a=1, b=2, c=3)
    yield
    ext.PDFBaseline._deregisterType("legacy_parabola")
    ext.PDFEnvelope._deregisterType("legacy_envelope")


@pytest.mark.parametrize("name,case", _legacy["cases"].items())
def test_boost_python_pickle_compatibility(
    name, case, legacy_component_registry
):
    if (sys.byteorder, struct.calcsize("P"), struct.calcsize("l")) != (
        _legacy["byteorder"],
        _legacy["pointer_size"],
        _legacy["long_size"],
    ):
        pytest.skip("Boost binary archive uses a different native ABI")
    obj = pickle.loads(base64.b64decode(case["pickle"]))
    # The restored object must itself remain serializable with this binding.
    for restored in [obj, pickle.loads(pickle.dumps(obj))]:
        if "rmax" in case:
            assert restored.rmax == case["rmax"]
            assert callable(restored)
            adapter = restored.getStructure()
            if "component" in case:
                component = getattr(restored, case["component"])
                assert component.type() == case["component_type"]
            if name == "Python_envelope":
                assert restored.getEnvelope("legacy_envelope")(2) == 11
            if "value" in case:
                np.testing.assert_allclose(
                    restored.value, case["value"], atol=1e-12
                )
                np.testing.assert_allclose(
                    restored.eval(), case["value"], atol=1e-12
                )
                # Old pickles name srreal_ext classes and must also retain
                # their normal public calling interface after restoration.
                module = import_module(
                    "diffpy.srreal." + CALCULATOR_MODULES[name]
                )
                assert type(restored) is getattr(module, name)
                result = restored()
                if "PDFCalculator" in name:
                    np.testing.assert_allclose(result[0], restored.rgrid)
                    np.testing.assert_allclose(result[1], restored.pdf)
                else:
                    expected = (
                        restored.sitesquareoverlaps
                        if name == "OverlapCalculator"
                        else restored.value
                    )
                    np.testing.assert_allclose(result, expected)
        else:
            adapter = restored
        assert adapter.countSites() == case["sites"]
        if case["sites"]:
            assert adapter.siteAtomType(0) == "Ni"
            np.testing.assert_allclose(
                adapter.siteCartesianPosition(0), [1, 2, 3]
            )


class NamedNoMeta(ext.NoMetaStructureAdapter):
    def __init__(self, source, name):
        super().__init__(source)
        self.source = source
        self.name = name

    def __getinitargs__(self):
        return self.source, self.name


class NamedNoSymmetry(ext.NoSymmetryStructureAdapter):
    def __init__(self, source, name):
        super().__init__(source)
        self.source = source
        self.name = name

    def __getinitargs__(self):
        return self.source, self.name


class StatefulNoMeta(NamedNoMeta):
    def __getstate__(self):
        return "custom proxy state", super().__getstate__()

    def __setstate__(self, state):
        assert state[0] == "custom proxy state"
        super().__setstate__(state[1])
        self.restored = True


@pytest.mark.parametrize("cls", [NamedNoMeta, NamedNoSymmetry, StatefulNoMeta])
@pytest.mark.parametrize(
    "clone",
    [copy.copy, copy.deepcopy, lambda obj: pickle.loads(pickle.dumps(obj))],
)
def test_proxy_subclass_pickle_hooks(cls, clone):
    original = cls(make_structure(), "named source")
    restored = clone(original)
    assert type(restored) is cls
    assert restored.name == "named source"
    assert restored.countSites() == 4
    assert restored.siteCartesianPosition(0)[0] == 1
    if cls is StatefulNoMeta:
        assert restored.restored


@pytest.mark.parametrize(
    "cls", [ext.NoMetaStructureAdapter, ext.NoSymmetryStructureAdapter]
)
@pytest.mark.parametrize("metadata", [False, True])
def test_proxy_default_pickle_hooks(cls, metadata):
    source = make_structure()
    proxy = cls(source)
    assert proxy.__getinitargs__() == (source,)
    if metadata:
        proxy.label = "source metadata"
    restored = pickle.loads(pickle.dumps(proxy))
    assert type(restored) is cls
    assert restored.countSites() == source.countSites()
    assert restored.__dict__ == proxy.__dict__


class AtomWithCustomState(ext.Atom):
    def __getstate__(self):
        return self.xc

    def __setstate__(self, state):
        self.xc = state


class CalculatorWithCustomState(PDFCalculator):
    def __getstate__(self):
        return self.rmax

    def __setstate__(self, state):
        self.rmax = state


@pytest.mark.parametrize(
    "cls", [AtomWithCustomState, CalculatorWithCustomState]
)
@pytest.mark.parametrize(
    "clone",
    [copy.copy, copy.deepcopy, lambda obj: pickle.loads(pickle.dumps(obj))],
)
def test_custom_state_cannot_silently_drop_metadata(cls, clone):
    original = cls()
    original.label = "must not disappear silently"
    with pytest.raises(RuntimeError, match="Incomplete pickle support"):
        clone(original)


def test_false_dictionary_policy_does_not_opt_out_of_guard():
    original = AtomWithCustomState()
    original.__getstate_manages_dict__ = False
    original.label = "metadata"
    with pytest.raises(RuntimeError, match="Incomplete pickle support"):
        pickle.dumps(original)
