##########################################################################################
# tests/test_shift.py
##########################################################################################

import numpy as np
import re
from psiops.ishift import ishift
from psiops.shift  import shift
from psiops._filter import _use_shortcuts

PRINT_ANSWERS = False   # change to True to print out this value of `ANSWERS`

ANSWERS = {
    (0,0,0,0): ([5.75, 2.5, 2.5, 6.75, 4.25, 3.75, 8.75, 8, 7.75], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (0,0,0,1): ([2.5, 2.5, 2.5, 4.5, 4.25, 3.75, 8, 7, 6.5], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (0,0,1,0): ([2.5, 2.5, 2.5, 4.5, 4.25, 3.75, 8, 7, 6.5], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (0,0,2,0): ([2.5, 2.5, 2.5, 4, 4.25, 3.75, 6, 5.25, 5.25], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (0,0,3,0): ([2.5, 2.5, 2.5, 4.25, 4.25, 3.75, 4.25, 4.25, 3.75], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (0,0,4,0): ([2.5, 2.5, 2.5, 4.5, 4.25, 3.75, 8, 7, 6.5], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (0,1,0,0): ([9, 9, 9, 9, 9, 9, 6.5, 6, 7], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (0,1,0,1): ([0, 0, 0, 0, 0, 0, 4, 3, 5], [1, 1, 1, 1, 1, 1, 0, 0, 0]),
    (0,1,1,0): ([4, 3, 5, 4, 3, 5, 4, 3, 5], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (0,1,2,0): ([2.5, 2.5, 2.5, 4.5, 4, 3.5, 6, 4.5, 6], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (0,1,3,0): ([4.5, 4, 3.5, 4.5, 4, 3.5, 2.5, 2.5, 2.5], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (0,1,4,0): ([4.5, 4, 3.5, 2.5, 2.5, 2.5, 4, 3, 5], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (0,2,0,0): ([9, 9, 9, 4, 7, 9, 1, 4.5, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (0,2,0,1): ([0, 0, 0, 4, 5, 0, 1, 0, 0], [1, 1, 1, 0, 0, 1, 0, 0, 1]),
    (0,2,1,0): ([4, 5, 5, 4, 5, 5, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (0,2,2,0): ([6.5, 7.5, 7, 4, 4.5, 3.5, 1, 0.5, 1.5], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (0,2,3,0): ([1, 1, 1.5, 4, 4, 3.5, 1, 1, 1.5], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (0,2,4,0): ([4, 5, 4, 4, 5, 4, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (1,0,0,0): ([9, 2.5, 2.5, 9, 4.25, 3.75, 9, 9, 9], [0, 1, 1, 0, 1, 1, 0, 0, 0]),
    (1,0,0,1): ([1.25, 2.5, 2.5, 2.25, 4.25, 3.75, 2, 3.5, 3.25], [1, 1, 1, 1, 1, 1, 1, 1, 1]),
    (1,0,1,0): ([2.5, 2.5, 2.5, 4.5, 4.25, 3.75, 8, 7, 6.5], [1, 1, 1, 1, 1, 1, 1, 1, 1]),
    (1,0,2,0): ([2.5, 2.5, 2.5, 4, 4.25, 3.75, 6, 5.25, 5.25], [1, 1, 1, 1, 1, 1, 1, 1, 1]),
    (1,0,3,0): ([2.5, 2.5, 2.5, 4.25, 4.25, 3.75, 4.25, 4.25, 3.75], [1, 1, 1, 1, 1, 1, 1, 1, 1]),
    (1,0,4,0): ([2.5, 2.5, 2.5, 4.5, 4.25, 3.75, 8, 7, 6.5], [1, 1, 1, 1, 1, 1, 1, 1, 1]),
    (1,1,0,0): ([9, 9, 9, 9, 9, 9, 9, 9, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (1,1,0,1): ([0, 0, 0, 0, 0, 0, 2, 1.5, 2.5], [1, 1, 1, 1, 1, 1, 1, 1, 1]),
    (1,1,1,0): ([4, 3, 5, 4, 3, 5, 4, 3, 5], [1, 1, 1, 1, 1, 1, 1, 1, 1]),
    (1,1,2,0): ([2.5, 2.5, 2.5, 4.5, 4, 3.5, 6, 4.5, 6], [1, 1, 1, 1, 1, 1, 1, 1, 1]),
    (1,1,3,0): ([4.5, 4, 3.5, 4.5, 4, 3.5, 2.5, 2.5, 2.5], [1, 1, 1, 1, 1, 1, 1, 1, 1]),
    (1,1,4,0): ([4.5, 4, 3.5, 2.5, 2.5, 2.5, 4, 3, 5], [1, 1, 1, 1, 1, 1, 1, 1, 1]),
    (1,2,0,0): ([9, 9, 9, 4, 9, 9, 1, 9, 9], [0, 0, 0, 1, 0, 0, 1, 0, 0]),
    (1,2,0,1): ([0, 0, 0, 4, 2.5, 0, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1]),
    (1,2,1,0): ([4, 5, 5, 4, 5, 5, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1, 1]),
    (1,2,2,0): ([6.5, 7.5, 7, 4, 4.5, 3.5, 1, 0.5, 1.5], [1, 1, 1, 1, 1, 1, 1, 1, 1]),
    (1,2,3,0): ([1, 1, 1.5, 4, 4, 3.5, 1, 1, 1.5], [1, 1, 1, 1, 1, 1, 1, 1, 1]),
    (1,2,4,0): ([4, 5, 4, 4, 5, 4, 1, 0, 1], [1, 1, 1, 1, 1, 1, 1, 1, 1]),
    (2,0,0,0): ([7+1/3, 3, 2.5, 8+2/3, 5, 3, 8.75, 8+2/3, 8+1/3], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (2,0,0,1): ([4, 3, 2.5, 8, 5, 3, 8, 8, 7], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (2,0,1,0): ([4, 3, 2.5, 8, 5, 3, 8, 8, 7], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (2,0,2,0): ([3, 3, 2.5, 5, 5, 3, 6, 5, 5], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (2,0,3,0): ([3, 3, 2.5, 5, 5, 3, 5, 5, 3], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (2,0,4,0): ([4, 3, 2.5, 8, 5, 3, 8, 8, 7], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (2,1,0,0): ([9, 9, 9, 9, 9, 9, 6.5, 6, 7], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (2,1,0,1): ([0, 0, 0, 0, 0, 0, 4, 3, 5], [1, 1, 1, 1, 1, 1, 0, 0, 0]),
    (2,1,1,0): ([4, 3, 5, 4, 3, 5, 4, 3, 5], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (2,1,2,0): ([4, 2.5, 2.5, 8, 2, 3.5, 6, 3, 6], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (2,1,3,0): ([8, 2, 3.5, 8, 2, 3.5, 4, 2.5, 2.5], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (2,1,4,0): ([8, 2, 3.5, 4, 2.5, 2.5, 4, 3, 5], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (2,2,0,0): ([9, 9, 9, 4, 7, 9, 1, 4.5, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (2,2,0,1): ([0, 0, 0, 4, 5, 0, 1, 0, 0], [1, 1, 1, 0, 0, 1, 0, 0, 1]),
    (2,2,1,0): ([4, 5, 5, 4, 5, 5, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (2,2,2,0): ([7, 7.5, 8, 4, 4.5, 3.5, 1, 0, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (2,2,3,0): ([1, 1, 2, 4, 4, 3.5, 1, 1, 2], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (2,2,4,0): ([4, 5, 4, 4, 5, 4, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (3,0,0,0): ([9, 3, 3, 8+2/3, 8, 3.75, 8.75, 8+2/3, 9], [0, 0, 0, 0, 0, 1, 0, 0, 0]),
    (3,0,0,1): ([1.25, 3, 3, 8, 8, 3.75, 8, 8, 3.25], [1, 0, 0, 0, 0, 1, 0, 0, 1]),
    (3,0,1,0): ([2.5, 3, 3, 8, 8, 3.75, 8, 8, 6.5], [1, 0, 0, 0, 0, 1, 0, 0, 1]),
    (3,0,2,0): ([2.5, 3, 3, 8, 8, 3.75, 8, 5.5, 3], [1, 0, 0, 0, 0, 1, 0, 0, 0]),
    (3,0,3,0): ([3, 3, 3, 8, 8, 3.75, 8, 8, 3.75], [0, 0, 0, 0, 0, 1, 0, 0, 1]),
    (3,0,4,0): ([2.5, 3, 3, 8, 8, 3.75, 8, 8, 6.5], [1, 0, 0, 0, 0, 1, 0, 0, 1]),
    (3,1,0,0): ([9, 9, 9, 9, 9, 9, 9, 6, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (3,1,0,1): ([0, 0, 0, 0, 0, 0, 2, 3, 2.5], [1, 1, 1, 1, 1, 1, 1, 0, 1]),
    (3,1,1,0): ([4, 3, 5, 4, 3, 5, 4, 3, 5], [1, 0, 1, 1, 0, 1, 1, 0, 1]),
    (3,1,2,0): ([2.5, 3, 2.5, 8, 4, 3.5, 8, 3, 6], [1, 0, 1, 0, 1, 1, 0, 0, 1]),
    (3,1,3,0): ([8, 4, 3.5, 8, 4, 3.5, 2.5, 3, 2.5], [0, 1, 1, 0, 1, 1, 1, 0, 1]),
    (3,1,4,0): ([8, 4, 3.5, 2.5, 3, 2.5, 4, 3, 5], [0, 1, 1, 1, 0, 1, 1, 0, 1]),
    (3,2,0,0): ([9, 9, 9, 3, 9, 9, 1, 9, 9], [0, 0, 0, 0, 0, 0, 1, 0, 0]),
    (3,2,0,1): ([0, 0, 0, 3, 2.5, 0, 1, 0, 0], [1, 1, 1, 0, 1, 1, 1, 1, 1]),
    (3,2,1,0): ([3, 5, 5, 3, 5, 5, 1, 0, 0], [0, 1, 1, 0, 1, 1, 1, 1, 1]),
    (3,2,2,0): ([6.5, 8, 8, 3, 4.5, 3, 1, 0.5, 1.5], [1, 0, 0, 0, 1, 0, 1, 1, 1]),
    (3,2,3,0): ([1, 1, 1.5, 3, 3, 3, 1, 1, 1.5], [1, 1, 1, 0, 0, 0, 1, 1, 1]),
    (3,2,4,0): ([3, 5, 3, 3, 5, 3, 1, 0, 1], [0, 1, 0, 0, 1, 0, 1, 1, 1]),
    (4,0,0,0): ([5.75, 2+1/3, 2+1/3, 6.75, 4.25, 3.75, 8.75, 8, 7.75], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (4,0,0,1): ([2.5, 2+1/3, 2+1/3, 4.5, 4.25, 3.75, 8, 7, 6.5], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (4,0,1,0): ([2.5, 2+1/3, 2+1/3, 4.5, 4.25, 3.75, 8, 7, 6.5], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (4,0,2,0): ([2.5, 2+1/3, 2+1/3, 4, 4.25, 3.75, 6, 6, 6], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (4,0,3,0): ([2+1/3, 2+1/3, 2+1/3, 4.25, 4.25, 3.75, 4.25, 4.25, 3.75], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (4,0,4,0): ([2.5, 2+1/3, 2+1/3, 4.5, 4.25, 3.75, 8, 7, 6.5], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (4,1,0,0): ([9, 9, 9, 9, 9, 9, 6.5, 9, 7], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (4,1,0,1): ([3, 3, 3, 3, 3, 3, 4, 3, 5], [1, 1, 1, 1, 1, 1, 0, 1, 0]),
    (4,1,1,0): ([4, 3, 5, 4, 3, 5, 4, 3, 5], [0, 1, 0, 0, 1, 0, 0, 1, 0]),
    (4,1,2,0): ([2.5, 2, 2.5, 4.5, 4, 3.5, 6, 6, 6], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (4,1,3,0): ([4.5, 4, 3.5, 4.5, 4, 3.5, 2.5, 2, 2.5], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (4,1,4,0): ([4.5, 4, 3.5, 2.5, 2, 2.5, 4, 3, 5], [0, 0, 0, 0, 0, 0, 0, 1, 0]),
    (4,2,0,0): ([9, 9, 9, 5, 7, 9, 1, 4.5, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (4,2,0,1): ([3, 3, 3, 5, 5, 3, 1, 0, 3], [1, 1, 1, 0, 0, 1, 0, 0, 1]),
    (4,2,1,0): ([5, 5, 5, 5, 5, 5, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (4,2,2,0): ([6.5, 7.5, 7, 5, 4.5, 4, 1, 0.5, 1.5], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (4,2,3,0): ([1, 1, 1.5, 5, 5, 4, 1, 1, 1.5], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
    (4,2,4,0): ([5, 5, 5, 5, 5, 5, 1, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0]),
}

def brief_fmt(val):
    """Compact, exact format of a value that is a multiple of 1/3 or 1/4."""

    valstr = str(val)
    parts = valstr.partition('.')

    # Trim ".0"
    if valstr[2] == '0':
        return parts[0]

    # Convert .333 to 1/3
    sign = '-' if valstr[0] == '-' else '+'
    if parts[2].startswith('333'):
        return parts[0] + sign + '1/3'

    # Convert .666 to 2/3
    if parts[2].startswith('666'):
        return parts[0] + sign + '2/3'

    # No change
    return valstr


def test_shift() -> None:

    rng = np.random.default_rng(4280)

    printed = False
    for status in (False, True):
        _use_shortcuts(status)

        array = np.arange(10)
        image = array + array[:,np.newaxis] + array[:,np.newaxis,np.newaxis]
        mask = np.zeros(image.shape, dtype='bool')
        for k in range(1,9):
            mask[...,k,k] = True

        # small shifts, all modes, masked and unmasked
        for mode in ('constant', 'nearest', 'wrap', 'reflect', 'mirror'):

            shifted = shift(image, 0, mode=mode)
            assert np.all(shifted == image)

            shifted = shift(image, (0,0.5), mode=mode)
            assert np.all(shifted[...,1:] == image[...,1:] - 0.5)

            shifted = shift(image, (0,-0.25), mode=mode)
            assert np.all(shifted[...,:-1] == image[...,:-1] + 0.25)

            shifted = shift(image, -0.75, mode=mode)
            assert np.all(shifted[...,:-1,:-1] == image[...,:-1,:-1] + 1.5)

            # with diagonal mask
            shifted, smask = shift(image, 0, mask, mode=mode, cval=0)
            assert np.all(shifted[~mask] == image[~mask])
            assert np.all(smask == mask)

        mode = 'constant'
        shifted, smask = shift(image, (0,0.5), mode=mode, cval=None)
        assert np.all(shifted[...,1:] == image[...,1:] - 0.5)
        assert not np.any(smask)

        shifted, smask = shift(image, (0.5,0), mode=mode, cval=None)
        assert np.all(shifted[...,1:,:] == image[...,1:,:] - 0.5)
        assert not np.any(smask)

        # with diagonal mask
        shifted, smask = shift(image, 0, mask, mode=mode, cval=None)
        assert np.all(shifted[~mask] == image[~mask])
        assert np.all(smask == mask)

        shifted, smask = shift(image, (0,0.5), mask, mode=mode, cval=None)
        assert np.all(shifted[mask]
                      == image[...,:,:-1][mask[...,:,1:]])
        assert np.all(shifted[...,:,1:][mask[...,:,1:]]
                      == image[...,:,:-1][mask[...,:,1:]])
        assert not np.any(smask)

        shifted, smask = shift(image, (0,0.96875), mask, mode=mode, cval=None)
        assert np.all(shifted[mask]
                      == image[...,:,:-1][mask[...,:,1:]])
        assert np.all(shifted[...,:,1:][mask[...,:,1:]]
                      == image[...,:,:-1][mask[...,:,1:]])
        assert not np.any(smask)

        shifted, smask = shift(image, (0,-0.5), mask, mode=mode, cval=None)
        assert np.all(shifted[mask]
                      == image[...,1:][mask[...,:-1]])
        assert np.all(shifted[...,:-1][mask[...,:-1]]
                      == image[...,1:][mask[...,:-1]])
        assert not np.any(smask)

        shifted, smask = shift(image, (0,-0.96875), mask, mode=mode, cval=None)
        assert np.all(shifted[mask]
                      == image[...,1:][mask[...,:-1]])
        assert np.all(shifted[...,:-1][mask[...,:-1]]
                      == image[...,1:][mask[...,:-1]])
        assert not np.any(smask)

        shifted, smask = shift(image, (0.5,0), mask, mode=mode, cval=None)
        assert np.all(shifted[mask]
                      == image[:,:-1][mask[:,1:]])
        assert np.all(shifted[:,1:][mask[:,1:]]
                      == image[:,:-1][mask[:,1:]])
        assert not np.any(smask)

        shifted, smask = shift(image, (0.96875,0), mask, mode=mode, cval=None)
        assert np.all(shifted[mask]
                      == image[:,:-1][mask[:,1:]])
        assert np.all(shifted[:,1:][mask[:,1:]]
                      == image[:,:-1][mask[:,1:]])
        assert not np.any(smask)

        shifted, smask = shift(image, (-0.5,0), mask, mode=mode, cval=None)
        assert np.all(shifted[mask]
                      == image[:,1:][mask[:,:-1]])
        assert np.all(shifted[:,:-1][mask[:,:-1]]
                      == image[:,1:][mask[:,:-1]])
        assert not np.any(smask)

        shifted, smask = shift(image, (-0.96875,0), mask, mode=mode, cval=None)
        assert np.all(shifted[mask]
                      == image[:,1:][mask[:,:-1]])
        assert np.all(shifted[:,:-1][mask[:,:-1]]
                      == image[:,1:][mask[:,:-1]])
        assert not np.any(smask)

        # modest shifts, all modes, masked and unmasked
        for mask in (rng.random(image.shape) < 0.1,
                     rng.random(image.shape) < 0.8):
            for mode in ('constant', 'nearest', 'wrap', 'reflect', 'mirror'):

                shifted1 = shift(image, (0,0.5), mode=mode)
                shifted2 = shift(image, (0,2.5), mode=mode)
                assert np.all(shifted1[...,:-2] == shifted2[...,2:])

                shifted1 = shift(image, (0,-1.5), mode=mode)
                shifted2 = shift(image, (0, 0.5), mode=mode)
                assert np.all(shifted1[...,:-2] == shifted2[...,2:])

                shifted1 = shift(image, (-1.25,0), mode=mode)
                shifted2 = shift(image, ( 0.75,0), mode=mode)
                assert np.all(shifted1[...,:-2,:] == shifted2[...,2:,:])

                shifted = shift(image, (0,1.5), mode=mode)
                assert np.all(shifted[...,2:] == image[...,2:] - 1.5)

            for mode in ('constant',):
                shifted1, smask1 = shift(image, (0,0.5), mask, mode=mode, cval=None)
                shifted2, smask2 = shift(image, (0,2.5), mask, mode=mode, cval=None)
                assert np.all(shifted1[...,:-2] == shifted2[...,2:])
                assert np.all(smask1[...,:-2] == smask2[...,2:])
                assert np.all(smask2[...,:2])

                shifted1, smask1 = shift(image, (0,-1.5), mask, mode=mode, cval=None)
                shifted2, smask2 = shift(image, (0, 0.5), mask, mode=mode, cval=None)
                assert np.all(shifted1[...,:-2] == shifted2[...,2:])
                assert np.all(smask1[...,:-2] == smask2[...,2:])
                assert np.all(smask1[...,-1])

                shifted1, smask1 = shift(image, (-1.25,0), mask, mode=mode, cval=None)
                shifted2, smask2 = shift(image, ( 0.75,0), mask, mode=mode, cval=None)
                assert np.all(shifted1[...,:-2,:] == shifted2[...,2:,:])
                assert np.all(smask1[...,:-2,:] == smask2[...,2:,:])
                assert np.all(smask1[...,-1,:])

                shifted, smask = shift(image, (0,1.5), mode=mode, cval=None)
                assert np.all(shifted[...,2:] == image[...,2:] - 1.5)
                assert not np.any(smask[...,1:])
                assert np.all(smask[...,:1])

        # quasi-random inputs to check against prior results
        if PRINT_ANSWERS and not printed:
            print('\nANSWERS = {')

        image = np.array([4, 3, 5, 1, 2, 0, 8, 6, 7]).reshape(3,3)
        mask0 = False
        mask1 = True
        mask2 = np.array([0, 0, 0, 1, 0, 0, 0, 1, 0]).reshape(3,3).astype('bool')
        mask3 = np.array([1, 0, 1, 1, 1, 1, 0, 1, 1]).reshape(3,3).astype('bool')
        mask4 = 3

        for m,mask in enumerate([mask0, mask1, mask2, mask3, mask4]):
          for o,offset in enumerate([(-0.5,0.5), (2.5,0), (1.,-1.5)]):
            for k,mode in enumerate(['constant', 'nearest', 'wrap', 'mirror', 'reflect']):
              for c,cval in enumerate([9, None]):
                if mode != 'constant' and c > 0:
                    continue
                shifted, smask = shift(image, offset=offset, mask=mask, mode=mode, cval=cval)
                if PRINT_ANSWERS and not printed:
                    vals = '[' + ', '.join([brief_fmt(x) for x in shifted.flatten()]) + ']'
                    mask_ = str(list(smask.flatten().astype('int')))
                    print(f'    ({m},{o},{k},{c}): ({vals}, {mask_}),')
                elif not printed:
                    answer = ANSWERS[m,o,k,c]
                    assert np.all(shifted == np.array(answer[0]).reshape(3,3))
                    assert np.all(smask == np.array(answer[1]).reshape(3,3))

        if PRINT_ANSWERS and not printed:
            print('}')
            printed = True

        # Make sure dtype float32 is preserved
        array = np.arange(10)
        image = (array + array[:,np.newaxis]).astype('float32')
        shifted = shift(image, (1,1))
        assert image.dtype == shifted.dtype

##########################################################################################
