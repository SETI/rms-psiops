##########################################################################################
# psiops/imagemodel/gaussian.py
##########################################################################################

import numpy as np
import scipy

from . import ImageModel


class Gaussian(ImageModel):
    """A ImageModel defined by a symmetric 2-D Gaussian."""

    _SQRT_HALF = np.sqrt(0.5)

    def __init__(self, sigma=1., integral=1.):
        """Constructor for a Gaussian ImageModel.

        Parameters:
            sigma (float, optional): The standard deviation of the Gaussian in units of
                pixels.
            integral (float, optional): The integral ("volume") of the ImageModel.
        """

        self._sigma = sigma
        self._integral = integral

    def transform(self, shape, center, expand=1., rotate=0.):
        """This Gaussian ImageModel re-sampled for a particular grid of pixels while
        preserving its integral.

        Parameters:
            shape (tuple of two ints): Two integers defining the shape of the returned
                array.
            center (tuple of two floats): Two floating-point coordinates defining the
                model's origin coordinates within the returned image array. Note that
                integers refer to the corners between pixels and half-integers refer to
                pixel centers. In other words, (0,0) is the lower corner of the image
                array and (0.5,0.5) is the center of the first pixel.
            expand (float, optional): An expansion (zoom) factor to apply to the
                ImageModel. Values greater than one increase the size of the ImageModel in
                both directions, but leave the center location unchanged. Note that the
                model's amplitude scales with 1/expand**2 in order to preserve the
                integral.
            rotate (float, optional): The angle in radians by which to rotate the
                ImageModel. Rotations are counterclockwise and are applied about the
                center of the ImageModel after it has been expanded.

        Returns:
            A 2-D array of the specified shape, containing the ImageModel as centered,
            expanded, and rotated.
        """

        i = np.arange(shape[0])[:, np.newaxis]
        j = np.arange(shape[1])
        sigma = expand * self._sigma
        return self._integral * Gaussian._gaussian_psf(i, j, center[0], center[1],
                                                       sigma=sigma)

    @staticmethod
    def _gaussian_integral(xmin, xmax, xcenter=0., sigma=1.):
        """The integral of a unit-area, 1-D Gaussian function.

        This uses the fact that the integral of a Gaussian from -infinity to x is equal
        to::

            (1 + erf((x - xcenter) / (sqrt(2)*sigma)) / 2

        This function uses scipy.special.erf(), which works for scalar and numpy array
        inputs.

        Parameters:
            xmin (array-like): The lower limit of the integration.
            xmax (array-like): The upper limit of the integration.
            xcenter (array-like, optional): The center coordinate of the Gaussian.
            sigma (array-like, optional): The standard deviation of the Gaussian.

        Returns:
            The value(s) of the Gaussian integral. This is a scalar if the inputs are
            all scalars; otherwise, it is an array with a shape defined by broadcasting
            together the shapes of all the inputs.
        """

        scale = Gaussian._SQRT_HALF / sigma
        umin = (xmin - xcenter) * scale
        umax = (xmax - xcenter) * scale
        return 0.5 * (scipy.special.erf(umax) - scipy.special.erf(umin))

    @staticmethod
    def _gaussian_psf(i, j, x0, y0, sigma):
        """Pixel value(s) of a 2-D, unit-volume Gaussian PSF.

        Note that pixel coordinates are integers at the corners of the pixel grid and
        half-integers at the centers. A Gaussian at (x0,y0) = (0,0) is centered at the
        corner of the pixel grid; a Gaussian at (x0,y0) = (0.5,0.5) is centered at the
        middle of the first pixel.

        Parameters:
            i (array-like): Integer pixel coordinate on the first axis.
            j (array-like): Integer pixel coordinate on the second axis.
            x0 (float): Center coordinate of the Gaussian along the first axis.
            y0 (float): Center coordinate of the Gaussian along the second axis.
            sigma (float): The standard deviation of the Gaussian.

        Returns:
            The value(s) of the 2-D Gaussian. This is a scalar if the inputs are all
            scalars; otherwise, it is an array with a shape defined by broadcasting
            together the shapes of all the inputs.
        """

        xmodel = Gaussian._gaussian_integral(i, i + 1., x0, sigma)
        ymodel = Gaussian._gaussian_integral(j, j + 1., y0, sigma)
        return xmodel * ymodel

##########################################################################################
