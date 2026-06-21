##########################################################################################
# psiops/rotate.py
##########################################################################################

import numpy as np

from psiops._filter import _use_shortcuts
from psiops._utils import _check_tuple, _merge_weights
from psiops._validation import _check_image, _check_return
from psiops.resample import resample

HALFPI = np.pi/2.
TWOPI = np.pi * 2.


def rotate(image, angle, mask=None, *, maskval=None, weights=None, nans=False,
           origin=None, center=None, shape=None, minweight=1.e-6, eps=1.e-12,
           returns=None, _debug=None):
    """Rotate this image while retaining photometric precision.

    This function calculates the area of overlap between each pixel (assumed square) of
    the input image and each pixel of a new, rotated grid of pixels of the same size. Each
    pixel of the returned, rotated image is an average of the overlapping pixel values
    from the input image, weighted by the area of overlap. In this way, the photometric
    content of the input is preserved, albeit at a slightly reduced spatial resolution.

    Parameters:
        image (array): Image array, in which the last two axes are the spatial
            dimensions. This can be a MaskedArray.
        angle (float): The rotation angle in radians. If we regard the spatial
            coordinates as (x,y), where the x-axis points rightward and the y-axis points
            upward, then the rotation is counterclockwise.
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
            location within the image array around which the rotation is performed. If not
            specified, the midpoint of `image` is used. Note that integer coordinates
            refer to the corners between pixels and half-integers refer to pixel centers.
            In other words, (0,0) is the lower corner of the image array and (0.5,0.5) is
            the center of the first pixel.
        center (tuple of two floats, optional): Two coordinates defining the transformed
            location of the origin in the returned image array. If not specified, the
            center will be determined to align (0,0) in the input `image` to (0,0) in the
            returned image. Note that integer coordinates refer to the corners between
            pixels and half-integers refer to pixel centers.
        shape (tuple of two ints, optional): The shape of the returned image. If not
            specified, the resampled image is large enough to encompass the entire content
            of the input `image`.
        minweight (float, optional): The minimum weight within a pixel in the returned
            image that is treated as significant. If the total weight (fraction of a
            source pixel) in a new pixel is less than this value, it will be treated as
            masked.
        eps (float, optional): The positional error tolerance in units of pixels when
            determining the overlap between the original pixels of `image` and the new,
            rotated pixels.
        returns (str, optional): Used to override the default quantity or quantities to
            return, one of "i" (image only), "im" (image and mask), "iw" (image and weight
            array), or "imw" (image, mask, and weight array). Append "c" to include the
            new center coordinates of the returned image.
        _debug (dict, optional): If provided, this dict is populated with internal state
            after the call, useful for testing and debugging. Keys: 'area_list',
            'imod_list', 'jmod_list' (parallel lists of per-pixel overlap areas and source
            row/column indices from the inner loop, or None when the angle is an exact
            multiple of π/2), 'new_center', 'new_mask', and 'new_weights'.

    Returns:
        `rotated` or (`rotated`[, `new_mask`][, `new_weights`][, `new_center`]):

        * `rotated`: The floating-point, rotated image array. If `image` is a MaskedArray,
          this will also be a MaskedArray. If `maskval` was specified, any masked elements
          in this array will be filled with this value. Otherwise, if `nans` is True,
          masked pixels will be filled with NaN.
        * `new_mask`: The new mask array, True wherever all the pixels contributing to the
          image are masked. By default, this is returned if `mask` is provided; use
          `returns` to override the default behavior.
        * `new_weights`: The weight array, equal to the sum of the weights of the elements
          that contributed to each pixel in `rotated`. By default, this is returned if
          `weights` is provided; use the `returns` to override this default behavior. If
          `weights` is not provided, this is the number of unmasked pixels that
          contributed to each new pixel's value.
        * `new_center`: The two coordinates of the center of `rotated`, corresponding to
          the `origin` in the input image. By default, this is included unless `center`
          was specified as input (in which case `new_center` is equal to `center`); use
          `returns` to override this default.

    Raises:
        ValueError: If any inputs are invalid.
        TypeError: If `image` dtype is not numeric.
        ZeroDivisionError: If the rotation angle is exactly a multiple of π/2 but differs
            from the nearest multiple by more than `eps`.
    """

    # Intepret array and mask
    image, mask, weights, info = _check_image(image, mask, maskval, weights, nans=nans,
                                              comps=True, returns=returns,
                                              extra_char='c',
                                              extra_by_default=(center is None))

    # Preserve a floating-point input dtype (e.g. float32, complex64); integer inputs
    # become float64 because the area-weighted accumulation is floating-point.
    out_dtype = image.dtype if image.dtype.kind in 'fc' else None

    # Interpret origin
    old_xy_shape = image.shape[-2:]
    origin = _check_tuple(origin, 'origin coordinates', floats=True, negs=True,
                          default=(old_xy_shape[0]/2., old_xy_shape[1]/2.))

    # Check the new center and new shape (and note that None returns None)
    new_center = _check_tuple(center, 'new center', floats=True, negs=True, nones=True)
    new_xy_shape = _check_tuple(shape, 'new shape', floats=False, negs=False, nones=True)

    # Locate four old corners in new grid
    old_xlim = np.array([0., old_xy_shape[0], old_xy_shape[0], 0.])
    old_ylim = np.array([0., 0., old_xy_shape[1], old_xy_shape[1]])
    shifted_x = old_xlim - origin[0]
    shifted_y = old_ylim - origin[1]
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    new_xlim = shifted_x * cos_angle - shifted_y * sin_angle
    new_ylim = shifted_x * sin_angle + shifted_y * cos_angle
    new_xy_min = (new_xlim.min(), new_ylim.min())
    new_xy_max = (new_xlim.max(), new_ylim.max())

    # Fill in the new shape if missing
    if new_xy_shape is None:
        # Define new shape as large enough to hold entire rotated image
        new_xy_shape = [int(np.ceil(new_xy_max[0] - new_xy_min[0] - minweight)),
                        int(np.ceil(new_xy_max[1] - new_xy_min[1] - minweight))]

        # If the new center is undefined, make sure there is room for its fractional part
        # to match the fractional part of the origin. This allows for cleaner rotations,
        # with less mixing of pixels, especially if the angle is near a multiple of pi/2.
        if new_center is None:
            for j in (0, 1):
                if (origin[j] - new_xy_shape[j]/2.) % 1. != 0.:
                    new_xy_shape[j] += 1

        new_xy_shape = tuple(new_xy_shape)

    # Fill in the new center if missing
    if new_center is None:
        # Place the new center as close as possible to the center of the new shape without
        # leaving behind any unnecessary unused space.
        new_center = [0, 0]
        for j in (0, 1):
            # Start at the center of the new shape
            new_center[j] = new_xy_shape[j] / 2.

            # Shift the center by up to a half-pixel in either direction to match the
            # fractional part of the origin.
            mod_diff = (origin[j] - new_center[j] + 0.5) % 1. - 0.5
            new_center[j] += mod_diff

            # Identify unused space and shift the new_center by integer steps to use it
            unused_below = new_center[j] + new_xy_min[j]
            unused_above = new_xy_shape[j] - new_center[j] - new_xy_max[j]
            if unused_below > 0 and unused_above < 0:
                new_center[j] -= np.floor(min(unused_below, -unused_above) + 0.5)
            elif unused_below < 0 and unused_above > 0:
                new_center[j] += np.floor(min(-unused_below, unused_above) + 0.5)

        new_center = tuple(new_center)

    # Prepare the image and weight array. Keep an un-weighted copy of the image and the
    # merged weights for the exact-multiple-of-pi/2 fallback, which delegates to
    # resample() and does its own weighting.
    weights = _merge_weights(mask, weights)
    orig_image = image
    orig_weights = weights
    if weights is None:
        weights = np.ones(image.shape, dtype=image.dtype)
    else:
        image = image * weights

    # Locate all the new pixel vertices in the old grid
    # new_xy indexing is [new_i, new_j, corner 0-3, 0 for x or 1 for y]
    new_x = np.arange(new_xy_shape[0] + 1) - new_center[0]
    new_y = np.arange(new_xy_shape[1] + 1) - new_center[1]

    old_x =  cos_angle * new_x[:, np.newaxis] + sin_angle * new_y + origin[0]
    old_y = -sin_angle * new_x[:, np.newaxis] + cos_angle * new_y + origin[1]

    old_corners = np.empty(new_xy_shape + (4, 2))
    old_corners[:, :, 0, 0] = old_x[:-1, :-1]
    old_corners[:, :, 1, 0] = old_x[:-1, 1: ]
    old_corners[:, :, 2, 0] = old_x[1: , 1: ]
    old_corners[:, :, 3, 0] = old_x[1: , :-1]

    old_corners[:, :, 0, 1] = old_y[:-1, :-1]
    old_corners[:, :, 1, 1] = old_y[:-1, 1: ]
    old_corners[:, :, 2, 1] = old_y[1: , 1: ]
    old_corners[:, :, 3, 1] = old_y[1: , :-1]

    # Find the minimum index in the new grid that straddles each old pixel
    old_imin = np.floor(np.min(old_corners[..., 0], axis=-1)).astype(np.int_)
    old_jmin = np.floor(np.min(old_corners[..., 1], axis=-1)).astype(np.int_)

    # Initialize the buffers; always float64 since _intersection produces float64 areas
    sum1 = np.zeros(image.shape[:-2] + new_xy_shape, dtype=np.float64)
    sum0 = np.zeros(image.shape[:-2] + new_xy_shape, dtype=np.float64)

    # Below, _intersection raises a ZeroDivisionError if the rotation angle is too close
    # to a multiple of pi/2. In this case, we need to catch this exception and use a
    # simpler procedure to rotate by the exact multiple of pi/2 and then resample.
    try:

        # Loop through the potentially overlapping pairs of pixels
        grid = np.empty(new_xy_shape + (4, 2), dtype=np.int_)
        if _debug is not None:
            area_list = []
            imod_list = []
            jmod_list = []
        for di in range(3):
            i = old_imin + di
            imod = i % old_xy_shape[0]
            imask = i != imod

            grid[..., 0, 0] = imod
            grid[..., 1, 0] = imod+1
            grid[..., 2, 0] = imod+1
            grid[..., 3, 0] = imod

            for dj in range(3):
                j = old_jmin + dj
                jmod = j % old_xy_shape[1]
                jmask = j != jmod

                grid[..., 0, 1] = jmod
                grid[..., 1, 1] = jmod
                grid[..., 2, 1] = jmod+1
                grid[..., 3, 1] = jmod+1

                areas = _intersection(grid, old_corners, eps)

                # Make sure out-of-range indices are unweighted
                areas[imask | jmask] = 0.

                # Accumulate weighted sums
                sum1 += areas * weights[..., imod, jmod] * image[..., imod, jmod]
                sum0 += areas * weights[..., imod, jmod]

                if _debug is not None:
                    area_list.append(areas)
                    imod_list.append(imod)
                    jmod_list.append(jmod)

        new_weights = sum0
        new_mask = (sum0 < minweight)
        new_weights[new_mask] = 0
        rotated = np.divide(sum1, new_weights, out=np.zeros_like(sum1),
                            where=~new_mask)

        if _debug is not None:
            _debug['area_list'] = area_list
            _debug['imod_list'] = imod_list
            _debug['jmod_list'] = jmod_list

    # Angle is too close to a multiple of pi/2
    except ZeroDivisionError as err:

        # Make sure angle is a multiple of pi/2; if not, raise this error
        angle %= TWOPI
        steps = int(angle/HALFPI + 0.5)     # round to nearest
        diff = angle - steps * HALFPI
        if abs(diff) > eps:
            raise err

        # Work from the original, un-weighted image and merged weights; resample() applies
        # its own weighting.
        rot_image = orig_image
        rot_weights = orig_weights
        rot_mask = mask

        # Use rot90 to rotate the image, mask, and weights together
        if steps != 0:
            rot_image = np.rot90(rot_image, steps, axes=(-2, -1))
            if rot_mask is not None:
                rot_mask = np.rot90(rot_mask, steps, axes=(-2, -1))
            if rot_weights is not None:
                rot_weights = np.rot90(rot_weights, steps, axes=(-2, -1))

        # Update the origin
        if steps == 1:
            origin = (rot_image.shape[-2] - origin[1], origin[0])
        elif steps == 2:
            origin = (rot_image.shape[-2] - origin[0],
                      rot_image.shape[-1] - origin[1])
        elif steps == 3:
            origin = (origin[1], rot_image.shape[-1] - origin[0])

        # resample() requires at least three dimensions, so temporarily add a leading
        # axis for 2-D inputs and squeeze it back off the results afterward.
        squeeze = rot_image.ndim < 3
        if squeeze:
            rot_image = rot_image[np.newaxis]
            if rot_mask is not None:
                rot_mask = rot_mask[np.newaxis]
            if rot_weights is not None:
                rot_weights = rot_weights[np.newaxis]

        # Use resample to create the new image. Force the general (non-shortcut) resample
        # path: the zoom==1 shortcut is unsuitable here because it does not return the new
        # mask that this branch requires. The flag is restored in all cases.
        saved_shortcuts = _use_shortcuts()
        try:
            _use_shortcuts(False)
            rotated, new_mask, new_weights = resample(rot_image, 1, mask=rot_mask,
                                                      maskval=maskval,
                                                      weights=rot_weights, origin=origin,
                                                      center=new_center,
                                                      shape=new_xy_shape,
                                                      minweight=minweight, returns='imw')
        finally:
            _use_shortcuts(saved_shortcuts)

        if squeeze:
            rotated = rotated[0]
            if new_mask is not None and new_mask.ndim > 2:
                new_mask = new_mask[0]
            if new_weights is not None and new_weights.ndim > 2:
                new_weights = new_weights[0]

        if _debug is not None:
            _debug['area_list'] = None
            _debug['imod_list'] = None
            _debug['jmod_list'] = None

    if _debug is not None:
        _debug['new_center'] = new_center
        _debug['new_mask'] = new_mask
        _debug['new_weights'] = new_weights

    return _check_return(rotated, new_mask, new_weights, info, extra=new_center,
                         return_dtype=out_dtype)


def _intersection(grid, corners, eps):
    """The intersection areas of two grids of unit squares.

    The input arrays have shape (...,4,2), where the second-to-last axis identifies
    corners of the square indexed 0-3, and the last coordinate is 0 for x, 1 for y.

    Parameters:
        grid (array): The (x,y) coordinates of the corners of unit squares. These squares
            must be oriented with sides parallel to the coordinate system. Shape is
            (..., 4, 2).
        corners (array): The (x,y) coordinates of the corners of a second set of unit
            squares. These can be oriented arbitrarily but must all have the same
            orientation. Shape is (..., 4, 2).
        eps (float): The positional error tolerance in units of pixels when determining
            the overlap between the original pixels of `image` and the new, rotated
            pixels.

    Returns:
        The area of overlap of each square in the first grid with its counterpart in the
        second grid.

    Raises:
        ZeroDivisionError: If any edge is parallel to the coordinate axes (angle is a
            multiple of π/2).
    """

    grid, corners = np.broadcast_arrays(grid, corners)

    # Locate all relevant intersections between the edges
    edge_points = _edge_points(grid, corners, eps)

    # Locate corner points inside the grid squares
    inside_points, inside_count = _inside_points(grid, corners, eps)

    # `edge_points` is guaranteed to start with a crossing from outside the rotated square
    # to inside, so it is usually safe to insert the interior points after them. There are
    # two issues:
    #   - If there are two interior points, we need to insert them in the correct order.
    #     We need to start with the one _closest_ to the last edge point.
    #   - If the interior point roughly coincides with an edge point, we need to omit it.

    # Identify inside points that are very close to edge points
    dsq = np.sum((inside_points[..., np.newaxis, :]
                  - edge_points[..., np.newaxis, :, :])**2, axis=-1)
    close = np.any(dsq <= 2*eps, axis=-1)
    mask = close[..., 0] & np.logical_not(close[..., 1])
    inside_points[mask, 0] = inside_points[mask, 1]
    mask = close[..., 1] & np.logical_not(close[..., 0])
    inside_points[mask, 1] = inside_points[mask, 0]
    inside_count[np.any(close, axis=-1)] -= 1

    # Merge the lists of edge points and inside points
    points = np.empty(grid.shape[:-2] + (11, 2))
    points[..., :8, :] = edge_points
    points[..., -1, :] = points[..., 0, :]      # close the loop!

    # First, just duplicate the last edge point
    points[..., 8:10, :] = points[..., 7:8, :]

    # Overwrite the last two points if they are inside points
    valid = inside_count > 0
    points[valid, 8:10] = inside_points[valid]

    # Check the distances from the 8th point to the last two; swap if necessary
    dsq = np.sum((points[..., 8:10, :] - points[..., 7:8, :])**2, axis=-1)
    swap = dsq[..., 0] > dsq[..., 1]
    points[swap, 8:10] = points[swap, 9:7:-1]

    # Calculate the area
    # See http://www.mathwords.com/a/area_convex_polygon.htm for formula
    area = np.abs(0.5 * (np.sum(points[..., :-1, 0] * points[..., 1:, 1], axis=-1) -
                         np.sum(points[..., :-1, 1] * points[..., 1:, 0], axis=-1)))

    return area


def _edge_points(grid, corners, eps):
    """Ordered array of crossing points where the grid square intersects the rotated
    square.

    Parameters:
        grid (array): The corners of unit squares aligned with the (x,y) coordinate grid.
            Shape is (..., 4, 2).
        corners (array): The corners of unit squares that have arbitrary rotation. Shape
            is (..., 4, 2), with the same leading dimensions as `grid`.
        eps (float): Positional tolerance in pixel units.

    Returns:
        Intersection (x,y) coordinates in an array of shape (..., 8, 2). The first
        intersection point is repeated to reach the dimensioned limit of eight. It is also
        guaranteed that this first point is a crossing from outside the rotated square to
        inside the rotated square.
    """

    i = np.arange(4)
    j = np.arange(1, 5) % 4

    # From https://en.wikipedia.org/wiki/Line-line_intersection
    c1 = grid[..., i, np.newaxis, :]
    c2 = grid[..., j, np.newaxis, :]
    c3 = corners[..., np.newaxis, i, :]
    c4 = corners[..., np.newaxis, j, :]

    d12 = c1 - c2
    d13 = c1 - c3
    d34 = c3 - c4

    denom = d12[..., 0] * d34[..., 1] - d12[..., 1] * d34[..., 0]
    if not np.all(denom):
        raise ZeroDivisionError()

    t = (d13[..., 0] * d34[..., 1] - d13[..., 1] * d34[..., 0]) / denom
    u = (d13[..., 0] * d12[..., 1] - d13[..., 1] * d12[..., 0]) / denom

    mask1 = (t < -eps) | (t > 1+eps) | (u < -eps) | (u > 1+eps)
        # Shape is (..., 4, 4), where second-to-last axis is the grid index and the last
        # indices are the corners index. Value is True where intersections DO NOT occur.
        # Cases where the crossing point is formally outside the square, but really close
        # to a corner, are counted as intersections.
    # points = c1 - t[..., np.newaxis] * d12

    # We also need to check for corners of the grid square inside the rotated square
    mask2 = _grid_outside_corner_mask(grid, corners, eps)
        # mask2 is True where grid points are OUTSIDE the rotated square.

    # Create a merged array of distances, with shape (4, 5); (:, 4) is the corner point
    distance = np.empty(grid.shape[:-2] + (4, 5))
    distance[..., :4] = t + i[:, np.newaxis]
        # i is the index of the side, and t is the distance from one corner to the next
    distance[..., 4] = np.arange(4.)
        # distances to the corners are (0, 1, 2, 3)

    # It's possible to end up with distance == 4
    distance[distance >= 4 - eps] = 0

    # Sort the crossings, moving non-intersections (identified by value 5) to the end
    distance[..., :4][mask1] = 5
    distance[..., 4:][mask2] = 5
    distance = distance.reshape(distance.shape[:-2] + (20,))
    distance = np.sort(distance, axis=-1)

    # It is possible for distance values to be duplicated; reduce to one and re-sort
    mask = np.abs(distance[..., 1:] - distance[..., :-1]) <= 2*eps
    distance[..., 1:][mask] = 5
    distance = np.sort(distance, axis=-1)

    # Count the number of intersections, then truncate the list to the first eight
    invalid_distance_mask = (distance == 5)
    count = 20 - np.sum(invalid_distance_mask, axis=-1)
    if not np.all(count <= 8):
        raise RuntimeError('impossible number of unique intersections')
    distance = distance[..., :8]
    invalid_distance_mask = invalid_distance_mask[..., :8]

    # Locate a grid corner that is outside the rotated square, and is followed by an edge
    # with no intersections. This is where any interior points will be inserted. If
    # there are no such corners and edges, fine because that means there are no interior
    # points.
    origin_corner = np.zeros(grid.shape[:-2], dtype=np.int_)
    origin_mask = mask2 & np.all(mask1, axis=-1)
    origin_corner[origin_mask[..., 2]] = 2
    origin_corner[origin_mask[..., 1]] = 1
    origin_corner[origin_mask[..., 0]] = 0

    # "Rotate" the distances until this corner is first
    distance = (distance - origin_corner[..., np.newaxis]) % 4
    distance[invalid_distance_mask] = 5
    distance = np.sort(distance, axis=-1)

    # Replace the invalid values with the first distance
    first_distance = np.broadcast_to(distance[..., :1], distance.shape)
    distance[invalid_distance_mask] = first_distance[invalid_distance_mask]
    distance = np.sort(distance, axis=-1)

    # Rotate distances back
    distance = (distance + origin_corner[..., np.newaxis]) % 4

    # Convert distances to points
    edge = np.floor(distance).astype(np.int_)
    frac = distance - edge

    igrid = np.arange(grid.shape[0])[:, np.newaxis, np.newaxis]
    jgrid = np.arange(grid.shape[1])[:, np.newaxis]
    grid0 = grid[(igrid, jgrid, edge)]
    grid1 = grid[(igrid, jgrid, (edge + 1)%4)]
    points = grid0 + frac[..., np.newaxis] * (grid1 - grid0)

    return points


def _grid_outside_corner_mask(grid, corners, eps):
    """Mask of grid corners that are outside the rotated square defined by `corners`.

    Parameters:
        grid (array): The corners of unit squares aligned with the coordinate axes. Shape
            is (..., 4, 2).
        corners (array): The corners of arbitrarily oriented unit squares. Shape is
            (..., 4, 2).
        eps (float): Positional tolerance in pixel units.

    Returns:
        Boolean array of shape (..., 4) that is True where a grid corner lies outside the
        rotated square.
    """

    # Define the grid point and two edges relative to corners[1]
    grid1 = grid - corners[..., 1:2, :]
    side01 = corners[..., 0:1, :] - corners[..., 1:2, :]
    side21 = corners[..., 2:3, :] - corners[..., 1:2, :]

    # A grid point is inside the square if its dot product with both sides is between
    # zero and one.
    u = np.sum(grid1 * side01, axis=-1)
    v = np.sum(grid1 * side21, axis=-1)

    return (u < -eps) | (u > 1+eps) | (v < -eps) | (v > 1+eps)


def _inside_points(grid, corners, eps):
    """Corners of the rotated square that lie inside the axis-aligned grid square.

    Parameters:
        grid (array): The corners of unit squares aligned with the coordinate axes. Shape
            is (..., 4, 2).
        corners (array): The corners of the rotated unit squares. Shape is (..., 4, 2).
        eps (float): Positional tolerance in pixel units.

    Returns:
        A tuple (`points`, `count`):

        * `points`: Array of shape (..., 2, 2) containing up to two inside points.
          If there is only one inside point, it is duplicated as the second.
        * `count`: Integer array of shape (...) containing the number of inside points
          (0, 1, or 2).
    """

    # Define mask of points inside. There could be up to two.
    xmin = grid[..., 0].min(axis=-1)[..., np.newaxis] - eps
    ymin = grid[..., 1].min(axis=-1)[..., np.newaxis] - eps
    xmax = xmin + (1 + 2*eps)
    ymax = ymin + (1 + 2*eps)

    x = corners[..., 0]
    y = corners[..., 1]
    inside = (x >= xmin) & (x <= xmax) & (y >= ymin) & (y <= ymax)

    # Count the inside points
    count = np.sum(inside, axis=-1)

    # Move all the inside coordinates to the front
    points = corners.copy()
    flag = np.abs(corners).max() + np.abs(grid).max() + 1
    points[np.logical_not(inside)] = flag
    index = np.argsort(points[..., 0], axis=-1)

    igrid = np.arange(grid.shape[0])[:, np.newaxis, np.newaxis]
    jgrid = np.arange(grid.shape[1])[:, np.newaxis]
    points[..., 0] = points[(igrid, jgrid, index)][..., 0]
    points[..., 1] = points[(igrid, jgrid, index)][..., 1]

    # We only need to keep the first two points
    points = points[..., :2, :]

    # If there is one point, duplicate it as the second
    mask = (count == 1)
    points[mask, 1, :] = points[mask, 0, :]

    return (points, count)

##########################################################################################
