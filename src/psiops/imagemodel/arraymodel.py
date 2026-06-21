##########################################################################################
# psiops/imagemodel/arraymodel.py
##########################################################################################

import numpy as np
import numpy.typing as npt

from ..resample import resample
from ..rotate import rotate as ops_rotate
from . import ImageModel


class ArrayModel(ImageModel):
    """An ImageModel defined by a 2-D image array."""

    def __init__(
        self,
        array: npt.ArrayLike,
        origin: tuple[float, float] | None = None,
        outside: float = 0.,
    ) -> None:
        """Constructor for an ArrayModel.

        Parameters:
            array: A 2-D array describing the model.
            origin: Two floating-point coordinates defining the origin coordinates within
                `array`. If not specified, the midpoint of the array is used. Note that
                integers refer to the corners between pixels and half-integers refer to
                pixel centers. In other words, (0,0) is the lower corner of the array and
                (0.5,0.5) is the center of the first pixel.
            outside: The value to use outside the array.
        """

        self._array = np.asarray(array, dtype=np.float64)
        self._outside = float(outside)

        if origin is None:
            self._origin = np.asarray(self._array.shape) / 2.
        else:
            self._origin = np.asarray(origin, dtype=np.float64)

        # Determine how much space we need around the center in a case of arbitrary
        # rotation.
        xradius = max(abs(self._origin[0]), abs(self._array.shape[0] - self._origin[0]))
        yradius = max(abs(self._origin[1]), abs(self._array.shape[1] - self._origin[1]))
        self._radius = np.sqrt(xradius**2 + yradius**2)

    def transform(
        self,
        shape: tuple[int, int],
        center: tuple[float, float],
        expand: float = 1.,
        rotate: float = 0.,
    ) -> np.ndarray:
        """This ArrayModel re-sampled for a particular grid of pixels while preserving its
        integral.

        Parameters:
            shape: Two integers defining the shape of the returned array.
            center: Two floating-point coordinates defining the model's origin coordinates
                within the returned image array. Note that integers refer to the corners
                between pixels and half-integers refer to pixel centers. In other words,
                (0,0) is the lower corner of the image array and (0.5,0.5) is the center
                of the first pixel.
            expand: An expansion (zoom) factor to apply to the ImageModel. Values greater
                than one increase the size of the ImageModel in both directions, but leave
                the center location unchanged. Note that the model's amplitude scales with
                1/expand**2 in order to preserve the integral.
            rotate: The angle in radians by which to rotate the ImageModel. Rotations are
                counterclockwise and are applied about the center of the ImageModel after
                it has been expanded.

        Returns:
            A 2-D array of the specified shape, containing the ImageModel as centered,
            expanded, and rotated.
        """

        # resample() and rotate() require at least three dimensions, so operate on the
        # 2-D model array with a leading unit axis and then drop it afterward.
        source = self._array[np.newaxis]

        if rotate:
            # Determine a temporary center and shape for a resample operation such that
            # pixels will be truncated during rotation.
            iradius = int(np.ceil(self._radius * expand))
            temp_center = (iradius + center[0] % 1, iradius + center[1] % 1)
            temp_shape  = (int(np.ceil(temp_center[0] + iradius)),
                           int(np.ceil(temp_center[1] + iradius)))

            # Resample
            array, mask = resample(source, expand, origin=self._origin,
                                   center=temp_center, shape=temp_shape, returns='im')

            # Rotate and place into final grid
            array, mask = ops_rotate(array, rotate, mask=mask, origin=temp_center,
                                     center=center, shape=tuple(shape), returns='im')

        # Without rotation, we can just use resample(); this is a common occurrence
        else:
            array, mask = resample(source, expand, origin=self._origin,
                                   center=center, shape=tuple(shape), returns='im')

        # resample() multiplies pixel values by expand**2, so divide it back out to
        # preserve the model's integral, as documented.
        array = array[0] / expand**2

        # Fill perimeter and return
        array[mask[0]] = self._outside
        return array

##########################################################################################
