##########################################################################################
# image_ops/imagemodel/summedmodel.py
##########################################################################################

import numpy as np

from . import ImageModel


class SummedModel(ImageModel):
    """An ImageModel defined by a sum of two or more ImageModels."""

    def __init__(self, models, factors):
        """Constructor for a SummedModel ImageModel.

        Parameters:
            models (list or tuple): A list of two or more ImageModels.
            factors (array-like): Scale factors to apply to these models.
        """

        self._nmodels = len(models)
        self._models = models
        self._factors = factors

    def transform(self, shape, center, expand=1., rotate=0.):
        """This SummedModel re-sampled for a particular grid of pixels, while preserving
        its integral.

        Parameters:
            shape (array-like): Two integers defining the shape of the returned array.
            center (array-like): Two floating-point coordinates defining the model's
                origin coordinates within the returned image array. Note that integers
                refer to the corners between pixels and half-integers refer to pixel
                centers. In other words, (0,0) is the lower corner of the image array and
                (0.5,0.5) is the center of the first pixel.
            expand (scalar): An expansion (zoom) factor to apply to the ImageModel. Values
                greater than one increase the size of the ImageModel in both directions,
                but leave the center location unchanged. Note that the model's amplitude
                scales with 1/expand**2 in order to preserve the integral.
            rotate (scalar): The angle in radians by which to rotate the ImageModel.
                Rotations are counterclockwise and are applied about the center of the
                ImageModel after it has been expanded.

        Returns:
            (array): A 2-D array of the specified shape, containing the ImageModel as
                centered, expanded, and rotated.
        """

        images = np.empty((self._nmodels,) + shape)
        k = 0
        for model, factor in zip(self._models, self._factors):
            images[k] = factor * model.transform(shape, center, expand, rotate)
            k += 1

        return np.sum(images, axis=0)

##########################################################################################
