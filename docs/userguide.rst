User's Guide
============

``psiops`` (Photometrically accurate Science Image Operations) is a library of
image-processing routines built on NumPy and SciPy. Unlike general-purpose image
tools, every operation is designed to preserve the *photometric* content of an
image: the total signal in any region is conserved (or scaled in a precisely
known way) as the image is shifted, rotated, zoomed, or resampled. This makes the
library suitable for quantitative scientific work where the number of photons in a
feature matters, not just its appearance.

This guide introduces the core concepts and then walks through each family of
operations with examples. For the full signature and options of every function,
see the :doc:`module` reference.

.. contents:: On this page
   :local:
   :depth: 2


Installation
------------

Install the package and its runtime dependencies with ``pip``:

.. code-block:: bash

   pip install rms-psiops

Then import the public functions from the top-level package:

.. code-block:: python

   import numpy as np
   import psiops


Core concepts
-------------

Image arrays and spatial axes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every operation treats the **last two axes** of an array as the spatial (row,
column) axes of an image. Any leading axes are interpreted as a stack of images.
For example, an array of shape ``(10, 512, 512)`` is a stack of ten 512×512
images, and an array of shape ``(3, 4, 512, 512)`` is a 3×4 grid of images.

Stack operations (:func:`~psiops.mean`, :func:`~psiops.median`,
:func:`~psiops.minimum`, :func:`~psiops.maximum`, :func:`~psiops.variance`, and
:func:`~psiops.stdev`) require the array to be at least three-dimensional,
because they need at least one non-spatial axis to combine across.

Spatial transforms (:func:`~psiops.shift`, :func:`~psiops.ishift`,
:func:`~psiops.zoom`, :func:`~psiops.unzoom`, :func:`~psiops.rotate`,
:func:`~psiops.resample`, and :func:`~psiops.reshape`) accept a plain 2-D
image. If an image stack is provided, they operate on each image in the stack.


The coordinate convention
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many functions accept fractional positions within the pixel grid (an ``origin``,
a ``center``, or a sub-pixel ``offset``). Throughout the library:

* **Integer** coordinates refer to the corners *between* pixels.
* **Half-integer** coordinates refer to pixel *centers*.

So ``(0.0, 0.0)`` is the lower corner of the image array and ``(0.5, 0.5)`` is the
center of the first pixel. Keeping this in mind avoids half-pixel surprises when
specifying rotation centers or resampling origins.


Photometric accuracy
~~~~~~~~~~~~~~~~~~~~~~

The defining property of this library is conservation of signal:

* Under :func:`~psiops.shift` and :func:`~psiops.rotate`, the sum of the pixels in
  a region is preserved almost exactly. Sub-pixel shifts use linear interpolation
  that redistributes — but does not create or destroy — flux.
* Under :func:`~psiops.zoom` by a factor of ``k``, each pixel's signal is spread
  over ``k × k`` output pixels, so the total sum is preserved while individual
  pixel values scale accordingly. :func:`~psiops.unzoom` averages blocks of pixels
  back down.

By contrast, the equivalent SciPy routines optimize for visual quality and do not
guarantee these properties, which is why this library reimplements them.


Identifying invalid pixels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Real scientific images contain bad pixels: saturated values, cosmic-ray hits,
detector defects, or gaps. Almost every function accepts several mutually
compatible ways to mark pixels as invalid so they are excluded from the
computation and remain flagged in the output:

``mask``
    A boolean array, ``True`` wherever the corresponding image value should be
    ignored. It is broadcast to the image shape, so a single 2-D mask can be
    applied to a whole stack.

``maskval``
    A scalar value that should be treated as invalid wherever it appears in the
    image (for example ``maskval=0`` to ignore zero-filled gaps).

``weights``
    A floating-point array of per-pixel weights. A weight of zero is equivalent to
    masking, while positive weights allow unequal weighting in averages and
    variances. Weights must never be negative.

``nans``
    Pass ``nans=True`` to treat any ``NaN`` in the image as an invalid pixel.

You can also pass a NumPy :class:`~numpy.ma.MaskedArray` directly; its mask is
honored and the result is returned as a ``MaskedArray``.

.. code-block:: python

   image = np.array([[1., 2., 3.],
                     [4., 5., 6.],
                     [7., 8., 9.]])
   mask = np.zeros(image.shape, dtype=bool)
   mask[1, 1] = True              # ignore the central pixel

   stack = image[np.newaxis]      # shape (1, 3, 3)
   result, result_mask = psiops.mean(stack, mask=mask[np.newaxis])


Controlling the return values
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each function returns the processed image and, optionally, an updated mask and/or
weight array. By default:

* If you supplied **no** mask or weights, the function returns just the image
  array.
* If you supplied a **mask** (or used a masking ``mode``), it returns a list
  ``[image, new_mask]``.
* If you supplied **weights**, it returns a list ``[image, new_weights]``.

You can override this with the ``returns`` keyword, which takes a short string:

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - ``returns``
     - Result
   * - ``'i'``
     - the image only
   * - ``'im'``
     - ``[image, mask]``
   * - ``'iw'``
     - ``[image, weights]``
   * - ``'imw'``
     - ``[image, mask, weights]``

The ``new_mask`` is ``True`` wherever every pixel that would have contributed to an
output value was invalid. The ``new_weights`` array is the summed weight behind
each output pixel — or, when no input weights were given, the integer count of
valid pixels that contributed.

.. note::

   :func:`~psiops.resample` and :func:`~psiops.rotate` can also report the
   transformed center coordinate. Append ``'c'`` to the ``returns`` string (for
   example ``returns='imc'``) to include it as the final element of the returned
   list. By default the center is included whenever it was not supplied as input.


Spatial transforms
------------------------------------------

Shifting
~~~~~~~~

:func:`~psiops.shift` translates an image by an arbitrary, possibly fractional,
offset along the two spatial axes, using flux-conserving linear interpolation.
:func:`~psiops.ishift` is a faster path for purely integer offsets.

.. code-block:: python

   image = np.random.default_rng(0).random((1, 64, 64))

   # Sub-pixel shift; pixels exposed at the boundary are masked by default
   shifted, shifted_mask = psiops.shift(image, (0.5, -1.5))

   # Integer shift, filling exposed pixels with a constant instead of masking
   moved = psiops.ishift(image, (2, -3), mode='constant', cval=0.)

The ``mode`` keyword controls how the boundary is handled. The default,
``'masked'``, marks pixels that move in from outside the array as invalid. The
other modes — ``'constant'`` (fill with ``cval``), ``'nearest'``, ``'wrap'``,
``'reflect'``, and ``'mirror'`` — match the familiar SciPy boundary conventions.


Zooming
~~~~~~~

:func:`~psiops.zoom` magnifies an image by an integer factor along each axis,
splitting each input pixel's signal across the corresponding block of output
pixels. :func:`~psiops.unzoom` is the inverse, averaging blocks back down to a
smaller image.

.. code-block:: python

   image = np.ones((1, 32, 32))

   big = psiops.zoom(image, 2)            # shape (1, 64, 64)
   small = psiops.unzoom(image, (2, 4))   # shape (1, 16, 8)

Either operation accepts a single factor (applied to both axes) or a
``(row_factor, col_factor)`` tuple.


Resampling
~~~~~~~~~~

:func:`~psiops.resample` is the general-purpose, flux-conserving transform that
combines a (possibly non-integer) zoom and a shift in a single pass. It is more
efficient and more precise than chaining separate operations, and it can place the
image into an output array of a chosen shape and center.

.. code-block:: python

   image = np.random.default_rng(1).random((100, 100))

   # Enlarge by 1.5x along both axes
   resampled, new_center = psiops.resample(image, 1.5)

   # Full control: image, mask, and the transformed center
   out, out_mask, out_center = psiops.resample(image, (2.0, 1.25), returns='imc')

Use ``origin`` to choose the point in the input image about which the resampling
is anchored, ``center`` to place that point in the output, and ``shape`` to fix the
output dimensions.


Reshaping
~~~~~~~~~

:func:`~psiops.reshape` resizes an image to an exact target shape, choosing the
zoom factors needed to map the input onto the requested number of rows and
columns. It is a convenience wrapper around :func:`~psiops.resample`.

.. code-block:: python

   image = np.random.default_rng(2).random((100, 100))
   resized = psiops.reshape(image, (50, 75))   # shape (50, 75)


Rotating
~~~~~~~~

:func:`~psiops.rotate` rotates each image by an arbitrary angle (in radians) about
a chosen center, conserving flux. By default it returns the rotated image together
with the new center coordinate; the output array is enlarged as needed so that no
content is clipped.

.. code-block:: python

   image = np.random.default_rng(3).random((1, 64, 64))

   rotated, center = psiops.rotate(image, np.pi / 6)

   # Rotate about a specific point and keep only the image
   rotated = psiops.rotate(image, np.pi / 6, origin=(32., 32.), returns='i')


Spatial filters
------------------------------------------

Spatial filters replace each pixel with a statistic computed over a small
neighborhood — its *footprint*. The footprint can be given as an integer (a square
window), a ``(rows, cols)`` tuple (a rectangular window), or an explicit boolean
array for an arbitrary shape. Masked pixels are excluded from each
neighborhood, and an output pixel is masked only if every pixel in its footprint
was invalid.

The available filters are :func:`~psiops.mean_filter`,
:func:`~psiops.median_filter`, :func:`~psiops.minimum_filter`,
:func:`~psiops.maximum_filter`, :func:`~psiops.variance_filter`, and
:func:`~psiops.stdev_filter`.

.. code-block:: python

   image = np.random.default_rng(4).random((1, 128, 128))
   mask = np.zeros(image.shape, dtype=bool)
   mask[:, ::16, ::16] = True            # a sparse grid of bad pixels

   # 3x3 mean, ignoring masked pixels
   smoothed, smoothed_mask = psiops.mean_filter(image, 3, mask=mask)

   # Median over a cross-shaped footprint
   cross = np.array([[0, 1, 0],
                     [1, 1, 1],
                     [0, 1, 0]], dtype=bool)
   despeckled = psiops.median_filter(image, cross)

:func:`~psiops.gaussian_filter` smooths with a Gaussian kernel of a given
``sigma`` (a scalar or per-axis tuple) and supports the standard boundary
``mode`` options ``'nearest'``, ``'constant'``, ``'reflect'``, ``'mirror'``, and
``'wrap'``:

.. code-block:: python

   blurred = psiops.gaussian_filter(image, 2.0, mode='nearest')


Stack operations
------------------------------------------

Stack operations reduce a stack of images across one or more non-spatial axes,
producing a single image (or a smaller stack). They mirror the NumPy reductions
but honor masks and weights and report how many pixels contributed to each result.
The available reductions are :func:`~psiops.mean`, :func:`~psiops.median`,
:func:`~psiops.minimum`, :func:`~psiops.maximum`, :func:`~psiops.variance`, and
:func:`~psiops.stdev`.

.. code-block:: python

   # A stack of 20 dithered exposures of the same field
   stack = np.random.default_rng(5).random((20, 256, 256))
   mask = np.random.default_rng(6).random((20, 256, 256)) < 0.05

   # Mask-aware average frame, plus the mask of pixels that were bad everywhere
   average, average_mask = psiops.mean(stack, mask=mask)

   # Robust combination via the median
   combined = psiops.median(stack, mask=mask, returns='i')

By default a reduction operates over **all** the non-spatial axes. Use ``axis`` to
reduce only a chosen axis or axes (negative indices count back from the first
spatial axis), and ``keepdims=True`` to retain the reduced axes as length-one
dimensions for broadcasting.

.. code-block:: python

   grid = np.random.default_rng(7).random((4, 5, 64, 64))

   per_column = psiops.mean(grid, axis=0)    # shape (5, 64, 64)
   per_row = psiops.mean(grid, axis=1)       # shape (4, 64, 64)

The :func:`~psiops.variance` and :func:`~psiops.stdev` reductions additionally
accept a ``vartype`` / ``stdtype`` keyword selecting the estimator: ``'biased'``
(divide by ``N``), ``'unbiased'`` / ``'frequency'`` (divide by ``N − 1``), or
``'reliability'`` (the default, an unbiased estimate when weights describe
measurement uncertainties).


A worked example: stacking dithered frames
------------------------------------------

The following example ties the pieces together. We take a stack of exposures that
are slightly offset from one another, register them onto a common grid with
sub-pixel shifts, and combine them with a mask-aware median to reject bad pixels.

.. code-block:: python

   import numpy as np
   import psiops

   rng = np.random.default_rng(42)

   # Ten 128x128 exposures, each with a known sub-pixel dither and bad pixels
   exposures = rng.random((10, 128, 128))
   offsets = rng.uniform(-2.0, 2.0, size=(10, 2))
   bad = rng.random((10, 128, 128)) < 0.02

   # Register every exposure back to the reference frame
   registered = np.empty_like(exposures)
   registered_mask = np.zeros(exposures.shape, dtype=bool)
   for i, (dy, dx) in enumerate(offsets):
       frame, frame_mask = psiops.shift(exposures[i:i + 1], (-dy, -dx),
                                        mask=bad[i:i + 1])
       registered[i] = frame
       registered_mask[i] = frame_mask

   # Combine with a mask-aware median; pixels bad in every frame stay masked
   combined, combined_mask = psiops.median(registered, mask=registered_mask)

The result, ``combined``, is a single 128×128 image in which the dither has been
removed and transient bad pixels have been rejected, while ``combined_mask`` flags
any location that was invalid in every exposure.


Fourier transforms and correlation
------------------------------------------

:func:`~psiops.fft` and :func:`~psiops.ifft` compute the forward and inverse 2-D FFT
over the last two axes of an array, and :func:`~psiops.fft_power` returns the power
spectrum. Pass ``retile=True`` to move the zero-frequency term to the center of the
output (convenient for display), and ``real=True`` to keep only the real component.

.. code-block:: python

   image = np.random.default_rng(8).random((128, 128))

   spectrum = psiops.fft(image)
   restored = psiops.ifft(spectrum, real=True)         # recovers the original image
   power = psiops.fft_power(image, retile=True)         # zero frequency at the center

:func:`~psiops.correlate` cross-correlates an image against a reference (with
``normalize=True`` scaling the result to the range −1 to 1), and
:func:`~psiops.autocorrelate` is the special case of an image with itself.
:func:`~psiops.ialign` uses the correlation peak to report the integer offset that
best aligns two images — handy for registering frames before stacking.

.. code-block:: python

   reference = psiops.shift(image, (5, -3), mode='wrap', returns='i')

   corr = psiops.correlate(image, reference, normalize=True)
   offset = psiops.ialign(image, reference, sigma=2.0)   # -> (5, -3)

The ``sigma`` argument to :func:`~psiops.ialign` unsharp-masks both images before
correlating, which sharpens the peak when the images contain large-scale gradients.


Image models
------------------------------------------

An :class:`~psiops.ImageModel` describes a continuous, photometrically normalized
source that can be rendered onto any pixel grid. Every model exposes one method,
``transform(shape, center, expand=1.0, rotate=0.0)``, which returns a 2-D array of the
requested ``shape`` with the model centered at ``center`` (in the pixel-corner/center
convention above), optionally magnified by ``expand`` and rotated by ``rotate``
radians. The model's integral is preserved.

:class:`~psiops.Gaussian` is a symmetric 2-D Gaussian of a given ``sigma`` and total
``integral``:

.. code-block:: python

   psf = psiops.Gaussian(sigma=2.0).transform((31, 31), center=(15.5, 15.5))
   # psf.sum() == 1.0  (unit integral by default)

:class:`~psiops.ArrayModel` wraps an arbitrary 2-D array, resampling it (flux-
conserving) onto the output grid. Use ``origin`` to choose which point of the source
array is placed at ``center``, and ``outside`` to set the fill value beyond the
array's footprint:

.. code-block:: python

   template = np.zeros((11, 11))
   template[5, 5] = 1.0
   model = psiops.ArrayModel(template, origin=(5.5, 5.5))
   rendered = model.transform((64, 64), center=(40.0, 25.0), expand=1.5, rotate=0.3)

:class:`~psiops.SmearedModel` averages a model along a ``(dx, dy)`` smear vector to
emulate motion blur, and :class:`~psiops.SummedModel` adds several models with
per-model scale factors (for example a narrow core plus a broad halo):

.. code-block:: python

   trailed = psiops.SmearedModel(psiops.Gaussian(sigma=1.5), smear=(6.0, 0.0))
   trail = trailed.transform((41, 41), center=(20.5, 20.5))

   combo = psiops.SummedModel(
       [psiops.Gaussian(sigma=1.0), psiops.Gaussian(sigma=4.0)],
       factors=[1.0, 0.3],
   )
   psf = combo.transform((41, 41), center=(20.5, 20.5))


Stretching to match an image
------------------------------------------

A :class:`~psiops.Stretch` finds the photometric transformation — a per-pixel
background plus a scaling of an image — that best matches a target, using fast linear
least squares. The ``orders`` argument gives the polynomial order of each component:
``[0, 0]`` is a constant background plus a constant scale factor, while higher orders
let the background and scaling vary smoothly across the image. (An order of ``-1``
omits a component.)

.. code-block:: python

   image = np.random.default_rng(9).random((50, 50))
   target = 3.0 * image + 1.5                     # a known scale and offset

   stretch = psiops.Stretch([0, 0])               # constant background + constant scale
   stretch.set_image(image)
   stretch.set_target(target)
   stretch.fit()

   stretch.scaling       # ~ 3.0
   stretch.background    # ~ 1.5
   stretch.model         # the best-fit reconstruction of target
   stretch.rms           # residual RMS (~ 0 for this exact example)

``set_target`` accepts the usual ``mask``, ``maskval``, ``weights``, and ``nans``
options, so invalid target pixels are excluded from the fit. The fitted coefficients
and their uncertainties are available as ``coeffs``, ``covar``, and the ``m_sigma`` /
``b_sigma`` / ``s_sigma`` properties.


Fitting a model to an image
------------------------------------------

A :class:`~psiops.Fitting` pairs an :class:`~psiops.ImageModel` with a
:class:`~psiops.Stretch` and solves for the geometric transformation — pixel offset
``(x, y)``, ``zoom``, and ``rotate`` — that, together with the stretch, best matches a
target image. It is the tool for measuring the position (and optionally the scale and
orientation) of a source whose shape you can model, such as centroiding a star against
a PSF.

.. code-block:: python

   # A star observed at a sub-pixel location, with some background
   truth = psiops.Gaussian(sigma=2.0)
   target = 5.0 * truth.transform((64, 64), center=(31.2, 28.7)) + 0.1

   model = psiops.Gaussian(sigma=2.0)
   stretch = psiops.Stretch([0, 0])               # background + amplitude
   # Seed the stretch with a same-shape render so it knows the target geometry.
   stretch.set_image(model.transform((64, 64), center=(31.0, 29.0)))

   fitting = psiops.Fitting(model, stretch)
   fitting.set_target(target)

   # Initial guess (x, y, zoom, rotate); by default only x and y are fitted.
   fitting.fit([31.0, 29.0, 1.0, 0.0])

   fitting.x, fitting.y          # best-fit center -> (31.2, 28.7)
   fitting.rms                   # residual RMS
   fitting.dx, fitting.dy        # one-sigma uncertainties on x and y
   fitting.model                 # the best-fit model image

The ``flags`` argument to ``fit`` selects which of the four parameters are free (the
default fits only ``x`` and ``y``), and ``limits`` bounds how far each may move from
its initial guess. Use ``set_target(..., corner=..., shape=...)`` to fit only a
sub-region of a larger image.


Where to go next
------------------------------------------

* The :doc:`module` page documents every public function, its full set of
  keyword options, and its return values.
* Each function's docstring describes its photometric behavior and edge cases in
  detail; they are reproduced in the API reference.
