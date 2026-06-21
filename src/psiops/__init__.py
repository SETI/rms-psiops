##########################################################################################
# psiops/__init__.py
##########################################################################################
"""\
psiops: Photometrically accurate Science Image Operations on 2-D images.

* Spatial transforms: zoom, unzoom, shift, ishift, resample, reshape, rotate.
* Spatial filters: gaussian, mean, median, minimum, maximum, variance, standard deviation.
* Stack operations: mean, median, minimum, maximum, variance, standard deviation.
* Other operations: FFT, inverse FFT, correlation, autocorrelation, image alignment and,
  least-squares fitting, PSF modeling and fitting.

General properties:

- Relative to similar functions such as those provided by SciPy, these spatial transforms
  preserve photometric integrity. In other words, the sum of the pixels in a given region
  of an image will be preserved almost exactly under all shift and rotate operations. As
  one would expect under "zoom" operations, pixel sums change in proportion to the product
  of the zoom factors being applied along the two image axes. SciPy functions do not have
  these properties, making them unreliable for many scientific purposes.

  All of the underlying algorithms are designed for numerical efficiency and should
  generally operate quickly. Unnecessary loops are rigorously avoided.

- Most operations support boolean masks of invalid pixels. Wherever the mask is True, the
  array value is excluded from scientific analysis. Masked pixels remain masked under
  spatial transforms and do not participate in "filter" and "stack" operations.

- Most operations can be performed on an arbitrarily-shaped array of images all at once.
  The last two axes of the array are assumed to be the spatial axes of the image.

Important note: There are many places in this library where one can specify a fractional
location within the pixel grid of an image. Throughout, we follow the convention that
integer coordinates refer to the corners between pixels and half-integers refer to pixel
centers. In other words, (0,0) is the lower corner of the image array and (0.5,0.5) is the
center of the first pixel.
"""

import os
import sys
import warnings

# Turn off all RuntimeWarnings by default. User can override if they wish.
if not sys.warnoptions:
    warnings.simplefilter('ignore', category=RuntimeWarning)
    os.environ['PYTHONWARNINGS'] = 'ignore::RuntimeWarning'  # also affect subprocesses

__all__ = [
    # Spatial transforms
    'ishift',
    'resample',
    'reshape',
    'rotate',
    'shift',
    'unzoom',
    'zoom',
    # Stack operations and spatial filters
    'gaussian_filter',
    'maximum',
    'maximum_filter',
    'mean',
    'mean_filter',
    'median',
    'median_filter',
    'minimum',
    'minimum_filter',
    'stdev',
    'stdev_filter',
    'variance',
    'variance_filter',
    # Modeling
    'ArrayModel',
    'Gaussian',
    'ImageModel',
    'SmearedModel',
    'Stretch',
    'SummedModel',
    # Other operations
    'autocorrelate',
    'correlate',
    'fft',
    'fft_power',
    'ialign',
    'ifft',
]

# Spatial transforms
# Stack operations and spatial filters
from psiops.gaussian_filter import gaussian_filter
from psiops.ishift   import ishift
from psiops.maximum  import maximum, maximum_filter
from psiops.mean     import mean, mean_filter
from psiops.median   import median, median_filter
from psiops.minimum  import minimum, minimum_filter
from psiops.resample import resample
from psiops.reshape  import reshape
from psiops.rotate   import rotate
from psiops.shift    import shift
from psiops.stdev    import stdev, stdev_filter
from psiops.unzoom   import unzoom
from psiops.variance import variance, variance_filter
from psiops.zoom     import zoom

# Modeling
from psiops.imagemodel              import ImageModel
from psiops.imagemodel.arraymodel   import ArrayModel
from psiops.imagemodel.gaussian     import Gaussian
from psiops.imagemodel.smearedmodel import SmearedModel
from psiops.imagemodel.summedmodel  import SummedModel
from psiops.stretch                 import Stretch
# from psiops.fitting  import Fitting
# from psiops.scaling  import Scaling

# Other operations
from psiops.fft import fft, ifft, fft_power, correlate, autocorrelate, ialign

##########################################################################################
