"""Regenerate fixtures using an installed Boost.Python diffpy.srreal
1.4.0.

Run this script with the reference environment's Python, without placing
the migration checkout on PYTHONPATH. The fixture records the native
archive ABI.
"""

import base64
import json
import pickle
import struct
import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np

import diffpy.srreal.srreal_ext as ext
from diffpy.srreal.bondcalculator import BondCalculator
from diffpy.srreal.bvscalculator import BVSCalculator
from diffpy.srreal.overlapcalculator import OverlapCalculator
from diffpy.srreal.pdfcalculator import DebyePDFCalculator, PDFCalculator
from diffpy.srreal.structureadapter import EMPTY
from diffpy.srreal.structureconverters import convertDiffPyStructure
from diffpy.structure import Atom, Structure

assert any("Boost.Python" in str(c) for c in ext.Atom.__mro__)

fixtures = {
    "producer": f"Boost.Python diffpy.srreal {version('diffpy.srreal')}",
    "byteorder": sys.byteorder,
    "pointer_size": struct.calcsize("P"),
    "long_size": struct.calcsize("l"),
    "cases": {},
}


def record(name, obj, **expected):
    fixtures["cases"][name] = {
        "pickle": base64.b64encode(pickle.dumps(obj, protocol=4)).decode(),
        **expected,
    }


record("EMPTY", EMPTY, sites=0)
for cls in (
    ext.AtomicStructureAdapter,
    ext.PeriodicStructureAdapter,
    ext.CrystalStructureAdapter,
):
    adapter = cls()
    atom = ext.Atom()
    atom.atomtype = "Ni"
    atom.xyz_cartn = [1, 2, 3]
    adapter.append(atom)
    if hasattr(adapter, "setLatPar"):
        adapter.setLatPar(4, 5, 6, 90, 90, 90)
    if hasattr(adapter, "addSymOp"):
        adapter.addSymOp(np.eye(3), np.zeros(3))
    record(cls.__name__, adapter, sites=1)
    # Native C++ clones use a different reconstruction protocol.
    record(cls.__name__ + "Clone", adapter.clone(), sites=1)

structure = Structure([Atom("Ni", [1, 2, 3]), Atom("Ni", [2, 2, 3])])
structure.Uisoequiv = 0.005
record("DiffPyAdapter", convertDiffPyStructure(structure), sites=2)
for cls in (
    BondCalculator,
    BVSCalculator,
    OverlapCalculator,
    PDFCalculator,
    DebyePDFCalculator,
):
    calc = cls(rmax=3)
    record(cls.__name__ + "Empty", calc, sites=0, rmax=3)
    calc.eval(structure)
    record(cls.__name__, calc, sites=2, rmax=3, value=calc.value.tolist())


def record_python_components():
    # Reuse importable Python components already exercised by the test suite.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from test_pdfbaseline import parabola_baseline
    from test_pdfenvelope import parabola_envelope
    from test_peakprofile import MySawTooth
    from test_peakwidthmodel import MyPWM
    from test_scatteringfactortable import LocalTable

    from diffpy.srreal.atomradiitable import CovalentRadiiTable
    from diffpy.srreal.pdfbaseline import makePDFBaseline
    from diffpy.srreal.pdfenvelope import makePDFEnvelope

    baseline = makePDFBaseline(
        "legacy_parabola", parabola_baseline, a=1, b=2, c=3
    )
    for attribute, component in (
        ("peakprofile", MySawTooth()),
        ("peakwidthmodel", MyPWM()),
        ("scatteringfactortable", LocalTable()),
        ("baseline", baseline),
    ):
        calc = PDFCalculator(rmax=3)
        setattr(calc, attribute, component)
        record(
            "Python_" + attribute,
            calc,
            sites=0,
            rmax=3,
            component=attribute,
            component_type=component.type(),
        )
    calc = OverlapCalculator(rmax=3)
    calc.atomradiitable = CovalentRadiiTable()
    record(
        "Python_atomradiitable",
        calc,
        sites=0,
        rmax=3,
        component="atomradiitable",
        component_type="covalent",
    )
    calc = PDFCalculator(rmax=3)
    calc.envelopes = [
        makePDFEnvelope("legacy_envelope", parabola_envelope, a=1, b=2, c=3)
    ]
    record("Python_envelope", calc, sites=0, rmax=3)


record_python_components()

path = (
    Path(__file__).resolve().parents[1]
    / "tests/testdata/boost_python_pickles.json"
)
path.write_text(json.dumps(fixtures, indent=2) + "\n")
