"""Main module."""
from typing import Optional, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike, DTypeLike
import rasterio

from seesar.enums import NormalizationKind


def rescale(
    arr: ArrayLike,
    src_min: Optional[float] = None,
    src_max: Optional[float] = None,
    src_min_per: Optional[float] = None,
    src_max_per: Optional[float] = None,
    dst_min: Union[float, int] = 0,
    dst_max: Union[float, int] = 255,
    kind: NormalizationKind = NormalizationKind.linear,
    dtype: DTypeLike = np.uint8,
) -> np.ndarray:
    """
    Rescale an input array to a specified dtype using a normalization method.

    Parameters
    ----------
    arr : ArrayLike
        Input array to be rescaled.
    src_min, src_max : float, optional
        The minimum and maximum values of the input array to map to dtype.
        Defaults to np.nanmin(arr) and np.nanmax(arr), respectively.
    src_min_per, src_max_per : int or float, optional
        The minimum and maximum values of the input array, expressed as a percentile
        (value between 0 and 100), map to dtype. Exclusive with src_min, src_max.
    dst_min, dst_max : int or float, optional
        The minimum and maximum values of the rescaled array. Defaults to 0 and 255,
        respectively.
    kind : NormalizationKind
        The normalization algorithm used for rescaling. Defaults to
        NormalizationKind.Linear.
    dtype : DTypeLike
        The dtype of the rescaled array. Defaults to np.uint8.

    Returns
    -------
    np.ndarray
        Rescaled array

    """
    if (src_min and not src_max) or (not src_min and src_max):
        raise ValueError("src_min and src_max must be specified together")
    if (src_min_per and not src_max_per) or (not src_min_per and src_max_per):
        raise ValueError("src_min_per and src_max_per must be specified together")
    if src_min and src_min_per:
        raise ValueError(
            "src_min, src_max and src_min_per, src_max_per are mutually exclusive"
        )

    arr = np.asarray(arr)
    axis: Optional[Tuple[int, int]] = None
    is_multiband: bool = False
    if len(arr.shape) > 2:
        # multiband, assume last two dimensions are width/height
        is_multiband = True
        axis = (-2, -1)
    
    if kind is NormalizationKind.log:
        # avoid runtime error
        arr = np.clip(arr, np.finfo(arr.dtype).eps, None)
        arr = np.log10(arr)

    if not (src_min_per and src_max_per):
        src_min = src_min or np.nanmin(arr, axis=axis)
        src_max = src_max or np.nanmax(arr, axis=axis)
    else:
        src_min = np.nanpercentile(arr, src_min_per, axis=axis)
        src_max = np.nanpercentile(arr, src_max_per, axis=axis)

    if not is_multiband and src_min > src_max:
        raise ValueError("src_max must be greater than src_min")

    # normalize to between [0.0, 1.0]
    arr = arr.T
    arr = (arr - src_min) / (src_max - src_min)
    arr = dst_min + (arr * dst_max)
    return arr.T.astype(dtype)
