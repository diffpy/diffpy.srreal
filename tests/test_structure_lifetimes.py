"""Writable atoms and safe NumPy snapshots during structure editing."""

import gc
import pickle

import numpy as np
import pytest

from diffpy.srreal import srreal_ext as ext
from diffpy.srreal.bondcalculator import BondCalculator

ADAPTERS = [
    ext.AtomicStructureAdapter,
    ext.PeriodicStructureAdapter,
    ext.CrystalStructureAdapter,
]


def make_structure(cls=ext.AtomicStructureAdapter):
    adapter = cls()
    adapter.reserve(16)
    for x in (1, 2, 3, 4):
        atom = ext.Atom()
        atom.atomtype = "Ni"
        atom.xc = x
        adapter.append(atom)
    return adapter


@pytest.mark.parametrize("cls", ADAPTERS)
def test_indexed_and_iterated_atoms_follow_survivors(cls):
    adapter = make_structure(cls)
    indexed = adapter[2]
    iterated = list(adapter)[2]
    adapter.insert(0, ext.Atom())
    indexed.xc = 30
    assert adapter[3].xc == iterated.xc == 30
    adapter.pop(1)
    iterated.xc = 40
    assert adapter[2].xc == indexed.xc == 40
    del adapter[:2]
    indexed.xc = 50
    assert adapter[0].xc == 50
    del adapter[0]
    indexed.xc = 60
    assert [atom.xc for atom in adapter] == [4]
    assert iterated.xc == 60


@pytest.mark.parametrize("cls", ADAPTERS)
@pytest.mark.parametrize("field", ["xyz_cartn", "uij_cartn"])
@pytest.mark.parametrize("operation", ["append", "reserve", "delete", "clear", "replace", "load"])
def test_views_become_safe_snapshots_before_structural_edits(cls, field, operation):
    adapter = make_structure(cls)
    atom = adapter[0]
    view = getattr(atom, field)
    retained_slice = view[..., :1]
    view[...] = 7
    np.testing.assert_array_equal(getattr(adapter[0], field), 7)
    if operation == "append":
        adapter.extend([ext.Atom()] * 100)
    elif operation == "reserve":
        adapter.reserve(100)
    elif operation == "delete":
        del adapter[0]
    elif operation == "clear":
        adapter.clear()
    elif operation == "replace":
        adapter[0] = ext.Atom()
    else:
        # Python-created adapters carry their native payload in __getstate__;
        # native objects normally restore it through their constructor.
        class Replacement(cls):
            pass

        adapter.__setstate__(make_structure(Replacement).__getstate__())
    if len(adapter):
        expected = getattr(adapter[0], field).copy()
        retained_slice[...] = 9
        np.testing.assert_array_equal(getattr(adapter[0], field), expected)
    del adapter, atom, view
    gc.collect()
    retained_slice[...] = 11
    np.testing.assert_array_equal(retained_slice, 11)


def test_retained_atoms_write_through_after_growth():
    adapter = make_structure()
    atom = adapter[0]
    adapter.extend([atom] * 100)
    atom.xc = 12
    assert adapter[0].xc == 12
    assert pickle.loads(pickle.dumps(atom)).xc == 12


@pytest.mark.parametrize("index", [1, slice(1, 3), slice(None, None, 2)])
def test_replaced_atoms_detach_without_affecting_survivors(index):
    adapter = make_structure()
    old = list(adapter)
    if isinstance(index, slice):
        adapter[index] = [ext.Atom()] * len(range(4)[index])
    else:
        adapter[index] = ext.Atom()
    assert [atom.xc for atom in old] == [1, 2, 3, 4]
    assert adapter[1].xc == (2 if index == slice(None, None, 2) else 0)


def test_slice_editing_and_invalid_replacements():
    adapter = make_structure()
    with pytest.raises(TypeError):
        adapter[1:3] = [ext.Atom(), object()]
    assert [a.xc for a in adapter] == [1, 2, 3, 4]
    adapter[1:3] = [adapter[0]]
    adapter.extend(adapter)
    assert [a.xc for a in adapter] == [1, 1, 4, 1, 1, 4]
    adapter[::-2] = [ext.Atom()] * 3
    del adapter[::2]
    assert [a.xc for a in adapter] == [0, 0, 0]


def test_calculation_uses_mutations_through_retained_atoms():
    adapter = make_structure()
    atom = adapter[2]
    adapter.insert(0, ext.Atom())
    adapter.pop(1)
    atom.xc = 8
    fresh = ext.AtomicStructureAdapter()
    for x in (0, 2, 8, 4):
        value = ext.Atom()
        value.xc = x
        value.atomtype = "Ni"
        fresh.append(value)
    np.testing.assert_allclose(
        BondCalculator(rmax=10)(adapter), BondCalculator(rmax=10)(fresh)
    )


@pytest.mark.parametrize("cls", ADAPTERS)
def test_structure_clone_override_calls_base_once(cls):
    class Derived(cls):
        def clone(self):
            result = super().clone()
            result[0].xc += 1
            return result

    adapter = Derived()
    adapter.append(ext.Atom())
    assert adapter.clone()[0].xc == 1
