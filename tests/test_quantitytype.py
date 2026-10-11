"""Numeric sequence behavior used by Python pair calculators."""

import numpy as np
import pytest

from diffpy.srreal.srreal_ext import PairQuantity, QuantityType


def test_quantity_slice_edits_are_atomic_and_support_aliasing():
    values = PairQuantity()._value
    values[:] = [1, 2, 3]
    values[1:2] = [4, 5]
    assert list(values) == [1, 4, 5, 3]
    values[::-1] = values
    assert list(values) == [3, 5, 4, 1]
    with pytest.raises(TypeError):
        values[:] = [1, object()]
    assert list(values) == [3, 5, 4, 1]

    def failing_values():
        yield 1
        raise RuntimeError("iteration failed")

    with pytest.raises(RuntimeError, match="iteration failed"):
        values[:] = failing_values()
    assert list(values) == [3, 5, 4, 1]
    values[:] = []
    assert len(values) == 0


@pytest.mark.parametrize("value", [1, 1.0, np.int64(1), np.float32(1)])
def test_quantity_membership_matches_numeric_assignment(value):
    values = QuantityType([value])
    assert value in values
    assert 1 in values
    assert 2 not in values


def test_mutable_quantity_uses_value_equality_without_hashing():
    values = QuantityType([1, 2])
    assert values == QuantityType([1, 2])
    assert values != QuantityType([2, 1])
    with pytest.raises(TypeError):
        hash(values)


def test_quantity_slices_preserve_swapped_and_reversed_values():
    values = np.arange(4.0).astype(np.dtype(float).newbyteorder("S"))[::-1]
    quantity = QuantityType()
    quantity[:] = values
    assert list(quantity) == values.tolist()
    quantity[::-1] = values
    assert list(quantity) == values[::-1].tolist()


def test_result_arrays_own_independent_writable_storage():
    quantity = PairQuantity()
    quantity._value[:] = [1, 2, 3]
    first, second = quantity.value, quantity.value
    assert first.flags.owndata and first.flags.writeable
    assert not np.shares_memory(first, second)
    first[:] = -1
    assert list(quantity._value) == [1, 2, 3]
    quantity._value[:] = [4, 5]
    del quantity
    np.testing.assert_array_equal(second, [1, 2, 3])
    first.resize((4,), refcheck=False)
