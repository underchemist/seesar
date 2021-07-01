#!/usr/bin/env python

"""Tests for `seesar` package."""

import pytest
import numpy as np
from numpy.testing import assert_array_equal

from seesar import seesar
from seesar.enums import NormalizationKind


@pytest.fixture
def simple_arr_3x3_float32_1():
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32) * 1000


@pytest.mark.parametrize(
    'args,kwargs',
    [
        [([], None, 1), dict()],
        [([], 1, None), dict()],
        [([], None, None, None, 1), dict()],
        [([], None, None, 1, None), dict()],
        [([], 0, 1, 2), dict()]
    ]
)
def test_rescale_args_valid(args, kwargs):
    with pytest.raises(ValueError):
        seesar.rescale(*args, **kwargs)


def test_rescale_linear(simple_arr_3x3_float32_1):
    arr = simple_arr_3x3_float32_1
    brr = seesar.rescale(arr, kind=NormalizationKind.linear)
    assert_array_equal(brr, np.array([[0, 31, 63],[95, 127, 159], [191, 223, 255]]))