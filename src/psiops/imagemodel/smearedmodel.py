##########################################################################################
# psiops/imagemodel/smearedmodel.py
##########################################################################################

import numpy as np

from . import ImageModel


class SmearedModel(ImageModel):
    """An ImageModel defined by smearing another ImageModel."""

    def __init__(self, model, smear, maxstep=0.5):
        """Constructor for a SmearedModel.

        The new model will be centered on the same coordinates as the original model.

        Parameters:
            model (ImageModel): The model to be smeared.
            smear (array-like): (dx,dy) coordinates defining the full smear of the model.
            maxstep (float, optional): The largest pixel offset allowed between evaluated
                positions of the original model.
        """

        self._model = model

        smear = np.asarray(smear, dtype=np.float64)
        if smear.shape != (2,):
            raise ValueError(f'invalid smear {smear.tolist()}; two values required')
        distance = np.sqrt(smear[0]**2 + smear[1]**2)
        nsteps = max(int(np.ceil(distance/maxstep)), 1)

        self._nsteps = nsteps
        self._offsets = smear/nsteps * np.arange(-nsteps/2 + 0.5, nsteps/2)[:,np.newaxis]

    def transform(self, shape, center, expand=1., rotate=0.):
        """This SmearedModel re-sampled for a particular grid of pixels while preserving
        its integral.

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

        center = np.array(center)
        cos = np.cos(rotate)
        sin = np.sin(rotate)
        images = np.zeros(shape)
        for offset in self._offsets:
            # Rotate (counterclockwise) and scale each smear offset so the smear direction
            # and extent respond to `rotate` and `expand`, then place the sub-model there.
            di, dj = offset
            rotated = np.array([cos * di - sin * dj, sin * di + cos * dj])
            placed = center + expand * rotated
            images += self._model.transform(shape, placed, expand, rotate)

        return images / self._nsteps

##########################################################################################
