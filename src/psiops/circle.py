##########################################################################################
# psiops/circle.py
##########################################################################################

import numpy as np


def circle(
    radius: float,
) -> np.ndarray:
    """A circular footprint (boolean array) of the specified radius.

    The shape is always odd to ensure that using it in a filter does not contribute an
    offset to the image geometry.

    Parameters:
        radius: The radius in pixels.

    Returns:
        A 2-D boolean array containing True wherever the distance from the footprint
        center to the center of a pixel is less than the given `radius`.
    """

    halfsize = int(radius)
    x = np.arange(-halfsize, halfsize+1)
    array = (x**2 + x[:,np.newaxis]**2 <= radius**2)
    return array

##########################################################################################
