##########################################################################################
# psiops/__init__.pyi
#
# Type stub for the public API of the psiops package, i.e. the names exported from
# psiops/__init__.py. Signatures mirror the implementation modules.
##########################################################################################

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

##########################################################################################
# Spatial transforms
##########################################################################################

def ishift(
    image: np.ndarray,
    offset: int | tuple[int, int],
    mask: np.ndarray | None = ...,
    *,
    maskval: float | None = ...,
    weights: np.ndarray | None = ...,
    nans: bool = ...,
    mode: str = ...,
    cval: float | None = ...,
    returns: str | None = ...,
) -> np.ndarray | list[np.ndarray]: ...
def resample(
    image: np.ndarray,
    zoom_: float | tuple[float, float],
    mask: np.ndarray | None = ...,
    *,
    maskval: float | None = ...,
    weights: np.ndarray | None = ...,
    nans: bool = ...,
    origin: tuple[float, float] | None = ...,
    center: tuple[float, float] | None = ...,
    shape: tuple[int, int] | None = ...,
    minweight: float = ...,
    returns: str | None = ...,
) -> np.ndarray | list[np.ndarray]: ...
def reshape(
    image: np.ndarray,
    shape: int | tuple[int, int],
    mask: np.ndarray | None = ...,
    *,
    maskval: float | None = ...,
    weights: np.ndarray | None = ...,
    nans: bool = ...,
    returns: str | None = ...,
) -> np.ndarray | list[np.ndarray]: ...
def rotate(
    image: np.ndarray,
    angle: float,
    mask: np.ndarray | None = ...,
    *,
    maskval: float | None = ...,
    weights: np.ndarray | None = ...,
    nans: bool = ...,
    origin: tuple[float, float] | None = ...,
    center: tuple[float, float] | None = ...,
    shape: tuple[int, int] | None = ...,
    minweight: float = ...,
    eps: float = ...,
    returns: str | None = ...,
    _debug: dict | None = ...,
) -> np.ndarray | list[np.ndarray]: ...
def shift(
    image: np.ndarray,
    offset: float | tuple[float, float],
    mask: np.ndarray | None = ...,
    *,
    maskval: float | None = ...,
    weights: np.ndarray | None = ...,
    nans: bool = ...,
    mode: str = ...,
    cval: float | None = ...,
    returns: str | None = ...,
) -> np.ndarray | list[np.ndarray]: ...
def unzoom(
    image: np.ndarray,
    unzoom_: int | tuple[int, int],
    mask: np.ndarray | None = ...,
    *,
    maskval: float | None = ...,
    weights: np.ndarray | None = ...,
    nans: bool = ...,
    returns: str | None = ...,
) -> np.ndarray | list[np.ndarray]: ...
def zoom(
    image: np.ndarray,
    zoom_: int | tuple[int, int],
    mask: np.ndarray | None = ...,
    *,
    maskval: float | None = ...,
    weights: np.ndarray | None = ...,
    nans: bool = ...,
    returns: str | None = ...,
) -> np.ndarray | list[np.ndarray]: ...

##########################################################################################
# Stack operations and spatial filters
##########################################################################################

def gaussian_filter(
    image: np.ndarray,
    sigma: float | tuple[float, float],
    mask: np.ndarray | None = ...,
    *,
    maskval: float | None = ...,
    weights: np.ndarray | None = ...,
    nans: bool = ...,
    mode: str = ...,
    cval: float = ...,
    order: int | tuple[int, int] = ...,
    returns: str | None = ...,
) -> np.ndarray | list[np.ndarray]: ...
def maximum(
    image: np.ndarray,
    mask: np.ndarray | None = ...,
    *,
    maskval: float | None = ...,
    weights: np.ndarray | None = ...,
    nans: bool = ...,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: bool = ...,
    returns: str | None = ...,
) -> np.ndarray | list[np.ndarray]: ...
def maximum_filter(
    image: np.ndarray,
    footprint: npt.ArrayLike | int | tuple[int, int],
    *,
    mask: np.ndarray | None = ...,
    maskval: float | None = ...,
    weights: np.ndarray | None = ...,
    nans: bool = ...,
    returns: str | None = ...,
) -> np.ndarray | list[np.ndarray]: ...
def mean(
    image: np.ndarray,
    mask: np.ndarray | None = ...,
    *,
    maskval: float | None = ...,
    weights: np.ndarray | None = ...,
    nans: bool = ...,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: bool = ...,
    factors: npt.ArrayLike | None = ...,
    returns: str | None = ...,
) -> np.ndarray | list[np.ndarray]: ...
def mean_filter(
    image: np.ndarray,
    footprint: npt.ArrayLike | int | tuple[int, int],
    *,
    mask: np.ndarray | None = ...,
    maskval: float | None = ...,
    weights: np.ndarray | None = ...,
    nans: bool = ...,
    returns: str | None = ...,
) -> np.ndarray | list[np.ndarray]: ...
def median(
    image: np.ndarray,
    mask: np.ndarray | None = ...,
    *,
    maskval: float | None = ...,
    weights: np.ndarray | None = ...,
    nans: bool = ...,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: bool = ...,
    factors: npt.ArrayLike | None = ...,
    omit: int = ...,
    returns: str | None = ...,
) -> np.ndarray | list[np.ndarray]: ...
def median_filter(
    image: np.ndarray,
    footprint: npt.ArrayLike | int | tuple[int, int],
    *,
    mask: np.ndarray | None = ...,
    maskval: float | None = ...,
    weights: np.ndarray | None = ...,
    nans: bool = ...,
    omit: int = ...,
    returns: str | None = ...,
) -> np.ndarray | list[np.ndarray]: ...
def minimum(
    image: np.ndarray,
    mask: np.ndarray | None = ...,
    *,
    maskval: float | None = ...,
    weights: np.ndarray | None = ...,
    nans: bool = ...,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: bool = ...,
    returns: str | None = ...,
) -> np.ndarray | list[np.ndarray]: ...
def minimum_filter(
    image: np.ndarray,
    footprint: npt.ArrayLike | int | tuple[int, int],
    *,
    mask: np.ndarray | None = ...,
    maskval: float | None = ...,
    weights: np.ndarray | None = ...,
    nans: bool = ...,
    returns: str | None = ...,
) -> np.ndarray | list[np.ndarray]: ...
def stdev(
    image: np.ndarray,
    mask: np.ndarray | None = ...,
    *,
    maskval: float | None = ...,
    weights: np.ndarray | None = ...,
    nans: bool = ...,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: bool = ...,
    factors: npt.ArrayLike | None = ...,
    stdtype: str = ...,
    returns: str | None = ...,
) -> np.ndarray | list[np.ndarray]: ...
def stdev_filter(
    image: np.ndarray,
    footprint: npt.ArrayLike | int | tuple[int, int],
    *,
    mask: np.ndarray | None = ...,
    maskval: float | None = ...,
    weights: np.ndarray | None = ...,
    nans: bool = ...,
    stdtype: str = ...,
    returns: str | None = ...,
) -> np.ndarray | list[np.ndarray]: ...
def variance(
    image: np.ndarray,
    mask: np.ndarray | None = ...,
    *,
    maskval: float | None = ...,
    weights: np.ndarray | None = ...,
    nans: bool = ...,
    axis: int | tuple[int, ...] | None = ...,
    keepdims: bool = ...,
    factors: npt.ArrayLike | None = ...,
    vartype: str = ...,
    returns: str | None = ...,
) -> np.ndarray | list[np.ndarray]: ...
def variance_filter(
    image: np.ndarray,
    footprint: npt.ArrayLike | int | tuple[int, int],
    *,
    mask: np.ndarray | None = ...,
    maskval: float | None = ...,
    weights: np.ndarray | None = ...,
    nans: bool = ...,
    vartype: str = ...,
    returns: str | None = ...,
) -> np.ndarray | list[np.ndarray]: ...

##########################################################################################
# Modeling
##########################################################################################

class ImageModel:
    def transform(
        self,
        shape: tuple[int, int],
        center: tuple[float, float],
        expand: float = ...,
        rotate: float = ...,
    ) -> np.ndarray: ...

class ArrayModel(ImageModel):
    def __init__(
        self,
        array: np.ndarray,
        origin: tuple[float, float] | None = ...,
        outside: float = ...,
    ) -> None: ...
    def transform(
        self,
        shape: tuple[int, int],
        center: tuple[float, float],
        expand: float = ...,
        rotate: float = ...,
    ) -> np.ndarray: ...

class Gaussian(ImageModel):
    def __init__(self, sigma: float = ..., integral: float = ...) -> None: ...
    def transform(
        self,
        shape: tuple[int, int],
        center: tuple[float, float],
        expand: float = ...,
        rotate: float = ...,
    ) -> np.ndarray: ...

class SmearedModel(ImageModel):
    def __init__(
        self,
        model: ImageModel,
        smear: npt.ArrayLike,
        maxstep: float = ...,
    ) -> None: ...
    def transform(
        self,
        shape: tuple[int, int],
        center: tuple[float, float],
        expand: float = ...,
        rotate: float = ...,
    ) -> np.ndarray: ...

class SummedModel(ImageModel):
    def __init__(
        self,
        models: Sequence[ImageModel],
        factors: npt.ArrayLike,
    ) -> None: ...
    def transform(
        self,
        shape: tuple[int, int],
        center: tuple[float, float],
        expand: float = ...,
        rotate: float = ...,
    ) -> np.ndarray: ...

class Stretch:
    orders: tuple[int, ...]
    ranks: tuple[int, ...]
    ncoeffs: int
    coeffs: np.ndarray | None
    image: np.ndarray | None
    image_mask: np.ndarray | None
    shape: tuple[int, int] | None
    target: np.ndarray | None
    target_mask: np.ndarray | None
    target_weights: np.ndarray | None
    mask: np.ndarray | None
    dof: int | None
    weight_sum: float | None
    chi_sq: float | None
    rms: float | None
    covar: np.ndarray | None
    def __init__(
        self,
        orders: Sequence[int],
        coeffs: npt.ArrayLike | None = ...,
        *,
        image: np.ndarray | None = ...,
        mask: np.ndarray | None = ...,
        maskval: float | None = ...,
        weights: np.ndarray | None = ...,
        nans: bool = ...,
    ) -> None: ...
    def set_coeffs(self, coeffs: npt.ArrayLike) -> None: ...
    def set_image(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = ...,
        maskval: float | None = ...,
        nans: bool = ...,
    ) -> None: ...
    def set_target(
        self,
        target: np.ndarray,
        *,
        mask: np.ndarray | None = ...,
        maskval: float | None = ...,
        weights: np.ndarray | None = ...,
        nans: bool = ...,
    ) -> None: ...
    def fit(self) -> None: ...
    @property
    def model(self) -> np.ndarray: ...
    @property
    def background(self) -> np.ndarray: ...
    @property
    def scaling(self) -> np.ndarray: ...
    @property
    def residuals(self) -> np.ndarray: ...
    @property
    def residuals_1d(self) -> np.ndarray: ...
    @property
    def m_sigma(self) -> np.ndarray: ...
    @property
    def b_sigma(self) -> np.ndarray: ...
    @property
    def s_sigma(self) -> np.ndarray: ...

class Fitting:
    imagemodel: ImageModel
    stretch: Stretch
    target: np.ndarray
    mask: np.ndarray | None
    weights: np.ndarray | None
    corner: tuple[int, int]
    shape: tuple[int, int]
    params: np.ndarray
    transformed: np.ndarray
    x: float
    y: float
    zoom: float
    rotate: float
    guesses: np.ndarray
    flags: np.ndarray
    limits: np.ndarray
    nparams: int
    lsq_dict: dict
    dof: int
    weight_sum: float
    chi_sq: float
    rms: float
    covar: np.ndarray
    dx: float
    dy: float
    corr: float
    def __init__(self, model: ImageModel, stretch: Stretch) -> None: ...
    def set_target(
        self,
        target: np.ndarray,
        *,
        mask: np.ndarray | None = ...,
        maskval: float | None = ...,
        weights: np.ndarray | None = ...,
        nans: bool = ...,
        corner: tuple[int, int] = ...,
        shape: tuple[int, int] | None = ...,
    ) -> None: ...
    def remask(self, mask: np.ndarray) -> None: ...
    def fit(
        self,
        params: npt.ArrayLike,
        flags: Sequence[bool] = ...,
        limits: Sequence[float] = ...,
        lsq_dict: dict | None = ...,
    ) -> None: ...
    @property
    def model(self) -> np.ndarray: ...
    @property
    def background(self) -> np.ndarray: ...
    @property
    def scaling(self) -> np.ndarray: ...
    @property
    def residuals(self) -> np.ndarray: ...
    @property
    def m_sigma(self) -> np.ndarray: ...
    @property
    def b_sigma(self) -> np.ndarray: ...
    @property
    def s_sigma(self) -> np.ndarray: ...

##########################################################################################
# Other operations
##########################################################################################

def fft(image: np.ndarray, *, retile: bool = ..., real: bool = ...) -> np.ndarray: ...
def ifft(image: np.ndarray, *, retile: bool = ..., real: bool = ...) -> np.ndarray: ...
def fft_power(image: np.ndarray, retile: bool = ...) -> np.ndarray: ...
def correlate(
    image: np.ndarray,
    reference: np.ndarray,
    *,
    normalize: bool = ...,
    retile: bool = ...,
) -> np.ndarray: ...
def autocorrelate(image: np.ndarray, retile: bool = ...) -> np.ndarray: ...
def ialign(
    image: np.ndarray,
    reference: np.ndarray,
    sigma: float,
) -> tuple[int, int]: ...

##########################################################################################
