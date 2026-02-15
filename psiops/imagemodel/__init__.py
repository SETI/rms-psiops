##########################################################################################
# image_ops/imagemodel/__init__.py
##########################################################################################
"""Definition of the abstract ImageModel class and its various subclasses:

All ImageModels have a transform() method that generates a 2-D image based on shift, zoom,
and rotate parameters, combined with an optional Scaling. The fit() method can be used to
determine the optimal transformation and Scaling to match the model to an image or portion
of an image. This can be used for applications such as PSF astrometry and removing glare.
"""

class ImageModel:
    """Abstract class that defines a 2-D pattern that can be re-sampled into an arbitrary
    grid of pixels, given an origin, expansion factor and rotation angle.

    The integrated "volume" under an ImageModel is unchanged during these transformations.
    """

    def transform(self, shape, center, expand=1., rotate=0.):
        """This ArrayModel re-sampled for a particular grid of pixels while preserving its
        integral.

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

        raise NotImplementedError(f'{type(self).__name__}.transform() is not implemented')

##########################################################################################

