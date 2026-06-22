##########################################################################################
# psiops/resample.py
##########################################################################################

import numpy as np

from psiops._filter import _use_shortcuts
from psiops._utils import _check_tuple, _merge_weights
from psiops._validation import _check_image, _check_return
from psiops.shift import shift


def resample(image, zoom_, mask=None, *, maskval=None, weights=None, nans=False,
             origin=None, center=None, shape=None, minweight=1.e-6, returns=None):
    """General function for integer or non-integer zoom and shift.

    This function is more efficient and somewhat more precise than combining separate
    zoom()/unzoom() and shift() operations. It also supports non-integer zoom factors.

    Parameters:
        image (array): Image array, in which the last two axes are the spatial
            dimensions. This can be a MaskedArray.
        zoom_ (float or tuple of two floats): The zoom factor or tuple of two zoom
            factors. Values > 1 expand the image, while values < 1 shrink the image.
        mask (array, optional): Boolean mask array, equal to True where the values in
            `image` are to be ignored. It is broadcasted to the shape of `image` if
            necessary.
        maskval (float, optional): A value that should be masked wherever it appears in
            `image`. This can be used instead of or in addition to the `mask`.
        weights (array, optional): Weight array specifying the possibly unequal weights
            associated with the pixels in `image`. A weight of zero is equivalent to a
            `mask` value of True. This can be provided in addition to or instead of the
            `mask` or `maskval`. It is broadcasted to the shape of `image` if necessary.
            Values should never be negative.
        nans (bool, optional): True to check `image` for NaNs and interpret them as
            masked values.
        origin (tuple of two floats, optional): Two coordinates defining the center
            location within the image array around which the resampling is performed. If
            not specified, the midpoint of `image` is used. Note that integer coordinates
            refer to the corners between pixels and half-integers refer to pixel centers.
            In other words, (0.,0.) is the lower corner of the image array and (0.5,0.5)
            is the center of the first pixel.
        center (tuple of two floats, optional): Two coordinates defining the transformed
            location of the origin in the returned image array. If not specified, the
            center will be determined to align (0,0) in the input `image` to (0,0) in the
            returned image. Note that integer coordinates refer to the corners between
            pixels and half-integers refer to pixel centers.
        shape (tuple of two ints, optional): The shape of the returned image. If not
            specified, the resampled image is large enough to encompass the entire
            content of the input `image`.
        minweight (float, optional): The minimum weight within a pixel in the returned
            image that is treated as significant. If the total weight (fraction of a
            source pixel) in a new pixel is less than this value, it will be treated as
            masked.
        returns (str, optional): Used to override the default quantity or quantities to
            return, one of "i" (image only), "im" (image and mask), "iw" (image and
            weight array), or "imw" (image, mask, and weight array). Append "c" to
            include the new center coordinates of the returned image.

    Returns:
        `resampled` or (`resampled`[, `new_mask`][, `new_weights`][, `new_center`]):

        * `resampled`: The floating-point, resampled image array. If `image` is a
          MaskedArray, this will also be a MaskedArray. If `maskval` was specified, any
          masked elements in this array will be filled with this value. Otherwise, if
          `nans` is True, masked pixels will be filled with NaN.
        * `new_mask`: The new mask array, True wherever all the pixels contributing to the
          image are masked. By default, this is returned if `mask` is provided; use
          `returns` to override the default behavior.
        * `new_weights`: The weight array, equal to the sum of the weights of the elements
          that contributed to each pixel in `resampled`. By default, this is returned if
          `weights` is provided; use `returns` to override this default behavior. If
          `weights` is not provided, this is the number of unmasked pixels that
          contributed to each new pixel's value.
        * `new_center`: The two coordinates of the center of `resampled`, corresponding
          to the `origin` in the input image. By default, this is included unless `center`
          was specified as input (in which case `new_center` is equal to `center`); use
          `returns` to override this default.

    Raises:
        ValueError: If any inputs are invalid or incompatible.
        TypeError: If `image` dtype is not numeric.
    """

    # Interpret the image and mask
    image, mask, weights, info = _check_image(image, mask, maskval, weights, nans=nans,
                                              comps=True, returns=returns, extra_char='c',
                                              extra_by_default=(center is None))
    weights = _merge_weights(mask, weights)

    # Interpret origin and zoom
    front = image.shape[:-2]
    old_xy_shape = image.shape[-2:]
    origin = _check_tuple(origin, 'origin coordinates', floats=True, negs=True,
                          default=(old_xy_shape[0]/2., old_xy_shape[1]/2.))
    zoom_ = _check_tuple(zoom_, 'zoom factor', floats=True, negs=False)

    # Check the new shape (and note that None returns None)
    new_xy_shape = _check_tuple(shape, 'new shape', floats=False, negs=False, nones=True)

    # Fill in a missing shape
    if new_xy_shape is None:
        if center is None:
            # Define new shape as large enough to hold entire resampled image
            new_xy_shape = (int(np.ceil(old_xy_shape[0] * zoom_[0] - minweight)),
                            int(np.ceil(old_xy_shape[1] * zoom_[1] - minweight)))
        else:
            # Given new center, define new shape to enclose remainder of image
            dx = (old_xy_shape[0] - origin[0]) * zoom_[0]
            dy = (old_xy_shape[1] - origin[1]) * zoom_[1]
            new_xy_shape = (int(np.ceil(center[0] + dx - minweight)),
                            int(np.ceil(center[1] + dy - minweight)))

    # Check and fill in the new center
    new_center = _check_tuple(center, 'new center', floats=True, negs=True,
                              default=(origin[0] * zoom_[0], origin[1] * zoom_[1]))

    # Use a shortcut if zoom_ == (1,1)

    # Use shift() if there's no zoom, because it's much faster. Note that zoom_==1 can be
    # a common occurrence.
    if zoom_ == (1, 1) and _use_shortcuts():
        offset = np.array(new_center) - np.array(origin)

        # Start by copying the image into the new output array within one pixel of its
        # destination.
        ioffset = offset.astype(np.int_)    # round toward zero to minimize truncation
        ijfrac = offset - ioffset

        # Expand destination array as needed to accommodate any extra fractional pixels
        temp_xy_shape = np.array(new_xy_shape) + np.ceil(np.abs(ijfrac)).astype(np.int_)
        temp_xy_shape = tuple(temp_xy_shape)

        # Define the boundaries of the original array in new coordinates
        old_ijmin = np.array((0, 0))
        old_ijmax = old_ijmin + old_xy_shape
        new_ijmin = ioffset
        new_ijmax = ioffset + old_xy_shape

        # Truncate limits to valid range; confirm overlap
        overlap = True
        for k in range(2):
            if new_ijmin[k] >= temp_xy_shape[k]:
                overlap = False
            elif new_ijmin[k] < 0:
                old_ijmin[k] = -new_ijmin[k]
                new_ijmin[k] = 0

            if new_ijmax[k] <= 0:
                overlap = False
            elif new_ijmax[k] > temp_xy_shape[k]:
                old_ijmax[k] -= (new_ijmax[k] - temp_xy_shape[k])
                new_ijmax[k] = temp_xy_shape[k]

        # Create and initialize new arrays
        resampled = np.empty(front + temp_xy_shape, dtype=image.dtype)
        resampled.fill(info.fill_value or 0.)
        if overlap:
            resampled[..., new_ijmin[0]:new_ijmax[0], new_ijmin[1]:new_ijmax[1]] = \
                image[..., old_ijmin[0]:old_ijmax[0], old_ijmin[1]:old_ijmax[1]]

        # Build the coverage/weight array, pre-zeroed so the boundary rows or columns left
        # uncovered by the integer offset are flagged (weight 0). Within the placed region
        # use the supplied weights, or 1 when unweighted. (Previously this was built only
        # when weights were supplied, dropping the boundary mask for unweighted input.)
        new_weights = np.zeros(resampled.shape, dtype=image.dtype)
        if overlap:
            fill = 1. if weights is None else \
                weights[..., old_ijmin[0]:old_ijmax[0], old_ijmin[1]:old_ijmax[1]]
            new_weights[..., new_ijmin[0]:new_ijmax[0], new_ijmin[1]:new_ijmax[1]] = fill

        # At this point our arrays have the required integer shift and possibly one extra
        # row or column. Apply the fractional shift as needed; the masked-mode shift
        # attenuates the coverage weights at the shifted boundary.
        resampled, new_weights = shift(resampled, offset=ijfrac, weights=new_weights,
                                       mode='masked', maskval=maskval, nans=nans,
                                       returns='iw')

        # Now trim off any extra pixels
        resampled = resampled[..., :new_xy_shape[0], :new_xy_shape[1]]
        new_weights = new_weights[..., :new_xy_shape[0], :new_xy_shape[1]]

        # shift() renormalizes the partially-covered boundary, so edge pixels keep their
        # intensity (equivalent to assuming an off-edge pixel equals the nearest in-image
        # pixel, not zero). Unweighted input returns a boolean coverage mask; weighted
        # input returns the propagated weights and lets _check_return derive any mask.
        if weights is None:
            new_mask = new_weights == 0.
            resampled[new_mask] = 0.        # fully-uncovered pixels are masked; zero them
            return _check_return(resampled, new_mask, None, info=info, extra=new_center)
        return _check_return(resampled, None, new_weights, info=info, extra=new_center)

    # Handle weighting along x-axis and then the y-axis

    # Define index arrays in old and new coordinates
    axis_info = []
    for axis in (0, 1):
        if zoom_[axis] >= 1:

            # Define the maximum number of new pixels that might overlap an original pixel
            max_pixels = int(np.ceil(zoom_[axis])) + 1

            # Get the indexes along the shorter of the new and original axes
            old_index = np.arange(old_xy_shape[axis])
            new_coord = (old_index - origin[axis]) * zoom_[axis] + new_center[axis]
            new_index = np.floor(new_coord).astype(np.int_)

            # Create the sequence of weights to apply
            axis_weights = np.ones((old_xy_shape[axis], max_pixels))
            axis_weights[:, -1] = 0.

            # Weight of the first pixel
            w0 = 1. - (new_coord - new_index)
            axis_weights[:, 0] = w0

            # Total weight in subsequent new pixels
            remainder = zoom_[axis] - w0

            # Number of fully weighted pixels
            fully = np.floor(remainder).astype(np.int_)

            # Weight of the last new pixel
            w1 = zoom_[axis] - fully - w0
            axis_weights[(old_index, fully+1)] = w1

            # Identify which of the new and original array axes requires iteration
            new_step = 1
            old_step = 0

        else:
            zoom_inv = 1. / zoom_[axis]

            # Define the maximum number of original pixels that might overlap a new pixel
            max_pixels = int(np.ceil(zoom_inv)) + 1

            # Get the index of the first weighted item along the new axis associated
            # with each item along the original x-axis.
            new_index = np.arange(new_xy_shape[axis])
            old_coord = (new_index - new_center[axis]) * zoom_inv + origin[axis]
            old_index = np.floor(old_coord).astype(np.int_)

            # Create the sequence of weights to apply, starting from the first.
            # The total of all these weights will equal 1./zoom_[axis].
            axis_weights = np.ones((new_xy_shape[axis], max_pixels))
            axis_weights[:, -1] = 0.

            # Weight of the first pixel
            w0 = 1. - (old_coord - old_index)
            axis_weights[:, 0] = w0

            # Total weight in subsequent new pixels
            remainder = zoom_inv - w0

            # Number of fully weighted pixels
            fully = np.floor(remainder).astype(np.int_)

            # Weight of the last new pixel
            w1 = zoom_inv - fully - w0
            axis_weights[(new_index, fully+1)] = w1

            # Now, re-scale the weights so the sum is unity
            axis_weights *= zoom_[axis]

            # Identify which of the new and original array axes requires iteration
            old_step = 1
            new_step = 0

        # Suppress roundoff errors in weights
        axis_weights[axis_weights < minweight] = 0.
        axis_info.append((old_index, new_index, old_step, new_step, axis_weights))

    (old_x, new_x, old_dx, new_dx, xweight) = axis_info[0]
    (old_y, new_y, old_dy, new_dy, yweight) = axis_info[1]

    # Create empty buffers for the arrays to be resampled. Weighted accumulation is
    # floating-point, so integer images must use a float buffer.
    out_dtype = image.dtype if image.dtype.kind in 'fc' else np.float64
    new_shape = image.shape[:-2] + new_xy_shape
    if weights is None:
        arrays = image
        buffer = np.zeros(new_shape, dtype=out_dtype)
        wbuffer = np.zeros(new_xy_shape, dtype=out_dtype)
        count = 1
    else:
        arrays = np.empty((2, *image.shape), dtype=out_dtype)
        arrays[1] = weights
        arrays[0] = image * weights
        buffer = np.zeros((2, *new_shape), dtype=out_dtype)
        new_mask = None         # derived from new_weights by _check_return()
        count = 2

    # The buffered scatter below silently drops duplicate write indices, so the write
    # ("new") index on each axis must stay unique; when an index falls outside the grid
    # it is zero-weighted and reassigned to a spare slot from these sets. Read ("old")
    # indices only need to be in range, so they are simply clamped (weight is already 0).
    new_xset = set(range(new_xy_shape[0]))
    new_yset = set(range(new_xy_shape[1]))

    # Construct the new arrays
    for i in range(xweight.shape[1]):
        ox = old_x + i * old_dx
        nx = new_x + i * new_dx
        wx = xweight[:, i].copy()

        if old_dx == 0:
            # Expanding: the write index `nx` iterates and must remain unique, so move
            # out-of-range entries to spare slots (always available since new >= old).
            xmask = (nx < 0) | (nx >= new_xy_shape[0])
            wx[xmask] = 0.
            unweighted = np.array(list(new_xset - set(nx)))
            nx[xmask] = unweighted[:xmask.sum()]
        else:
            # Shrinking: the write index `nx` is already unique (it is `arange`); only
            # the read index `ox` may stray out of range, so clamp it to a valid,
            # zero-weighted slot. (Spare read indices may not exist when the output is
            # much larger than the source, so a clamp is used, not a unique reassignment.)
            xmask = (ox < 0) | (ox >= old_xy_shape[0])
            wx[xmask] = 0.
            ox[xmask] = 0

        for j in range(yweight.shape[1]):
            oy = old_y + j * old_dy
            ny = new_y + j * new_dy
            wy = yweight[:, j].copy()

            if old_dy == 0:
                # Expanding along y: keep the write index `ny` unique (see above).
                ymask = (ny < 0) | (ny >= new_xy_shape[1])
                wy[ymask] = 0.
                unweighted = np.array(list(new_yset - set(ny)))
                ny[ymask] = unweighted[:ymask.sum()]
            else:
                # Shrinking along y: clamp the out-of-range, zero-weighted read index.
                ymask = (oy < 0) | (oy >= old_xy_shape[1])
                wy[ymask] = 0.
                oy[ymask] = 0

            wxy = wx[:, np.newaxis] * wy
            buffer[..., nx[:, np.newaxis], ny] += wxy * arrays[..., ox[:, np.newaxis], oy]

            if count == 1:
                # Accumulate the coverage weight, used below to renormalize and to mask
                wbuffer[nx[:, np.newaxis], ny] += wxy

    if count == 1:
        # Renormalize so partially-covered edge pixels keep their intensity (equivalent to
        # assuming an off-edge pixel equals the nearest in-image pixel, not zero). Fully
        # uncovered pixels give 0/0 -> NaN, as in the weighted and shortcut paths.
        new_mask = wbuffer == 0.
        with np.errstate(invalid='ignore', divide='ignore'):
            resampled = buffer / wbuffer
        resampled[..., new_mask] = 0.       # fully-uncovered pixels are masked; zero them
        new_weights = None
        if new_mask.shape != resampled.shape:
            new_mask = np.broadcast_to(new_mask, resampled.shape)

    else:
        (resampled, new_weights) = buffer
        # Fully masked pixels produce 0/0 -> NaN, which is expected and handled later
        with np.errstate(invalid='ignore', divide='ignore'):
            resampled = resampled / new_weights

    return _check_return(resampled, new_mask, new_weights, info=info, extra=new_center)

##########################################################################################
