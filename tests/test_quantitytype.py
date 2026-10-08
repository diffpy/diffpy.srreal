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
