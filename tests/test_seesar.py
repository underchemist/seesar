#!/usr/bin/env python

"""Tests for `seesar` package."""

import pytest
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

from seesar import seesar
from seesar.enums import NormalizationKind


@pytest.fixture
def arr_3x3_float32():
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32) * 1000


@pytest.fixture
def arr_2x3x3_float32(arr_3x3_float32):
    arr = arr_3x3_float32
    return np.broadcast_to(arr, (2, *arr.shape))


@pytest.mark.parametrize(
    "args,kwargs",
    [
        [([], None, 1), dict()],
        [([], 1, None), dict()],
        [([], None, None, None, 1), dict()],
        [([], None, None, 1, None), dict()],
        [([], 0, 1, 2), dict()],
        [([], 2, 1), dict()],
    ],
)
def test_rescale_args_valid(args, kwargs):
    with pytest.raises(ValueError):
        seesar.rescale(*args, **kwargs)


def test_rescale_linear(arr_3x3_float32):
    arr = arr_3x3_float32
    brr = seesar.rescale(arr, kind=NormalizationKind.linear)
    assert_array_equal(brr, np.array([[0, 31, 63], [95, 127, 159], [191, 223, 255]]))


def test_rescale_log(arr_3x3_float32):
    arr = arr_3x3_float32
    brr = seesar.rescale(arr, src_min=-20, src_max=0, kind=NormalizationKind.log)
    assert brr.min() >= 0
    assert brr.max() <= 255


def test_rescale_return_shape(arr_3x3_float32, arr_2x3x3_float32):
    arr = arr_3x3_float32
    brr = arr_2x3x3_float32
    assert seesar.rescale(arr).shape == arr.shape
    assert seesar.rescale(brr).shape == brr.shape


@pytest.mark.parametrize(
    "dst_min,dst_max,dtype",
    [(2, 4, np.uint8), (0, 65535, np.uint16), (-32767, 32767, np.int16)],
)
def test_rescale_return_values_and_dtypes(dst_min, dst_max, dtype, request):
    arr = request.getfixturevalue("arr_3x3_float32")
    out = seesar.rescale(arr, dst_min=dst_min, dst_max=dst_max, dtype=dtype)
    assert out.min() >= dst_min
    assert out.max() <= dst_max
    assert out.dtype == dtype
