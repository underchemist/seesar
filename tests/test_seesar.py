#!/usr/bin/env python

"""Tests for `seesar` package."""

import pytest
import numpy as np

from seesar import seesar
from seesar.enums import NormalizationKind

def test_arr_rescale():
    arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]).astype(np.float32) * 1000
    brr = seesar.rescale(arr, kind=NormalizationKind.linear)
    assert brr.dtype is np.uint8
    assert brr == np.array([[0, 31, 63],[95, 127, 159], [191, 223, 255]])