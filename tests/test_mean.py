##########################################################################################
# tests/test_mean.py
##########################################################################################

import numpy as np
import pytest

from psiops.mean import mean, mean_filter

##########################################################################################
# mean
##########################################################################################

def test_mean_basic() -> None:
    array = np.arange(10)
    image0 = array + array[:,np.newaxis]
    image = image0 * np.array([-3,1,2])[:,np.newaxis,np.newaxis]

    a = mean(image)
    assert np.all(a == 0.)


def test_mean_mask_2d(shortcuts: bool) -> None:
    # Masked reduction hits the `_use_shortcuts()`-gated path; check both settings.
    rng = np.random.default_rng(5965)
    array = np.arange(10)
    image0 = array + array[:,np.newaxis]
    image = image0 * np.array([-3,1,2])[:,np.newaxis,np.newaxis]

    a = mean(image)
    mask = rng.random((10,10)) < 0.3
    b, bmask = mean(image, mask=mask)
    assert np.all(mask == bmask)
    assert np.shape(bmask) == (10,10)
    assert np.all(a[~mask] == b[~mask])


def test_mean_mask_3d_one_layer(shortcuts: bool) -> None:
    array = np.arange(10)
    image0 = array + array[:,np.newaxis]
    image = image0 * np.array([-3,1,2])[:,np.newaxis,np.newaxis]

    mask = np.zeros(image.shape, dtype='bool')
    mask[0] = True
    b, bmask = mean(image, mask=mask)
    assert np.all(b == 1.5 * image0)
    assert not np.any(bmask)


def test_mean_mask_3d_partial(shortcuts: bool) -> None:
    rng = np.random.default_rng(5965)
    array = np.arange(10)
    image0 = array + array[:,np.newaxis]
    image = image0 * np.array([-3,1,2])[:,np.newaxis,np.newaxis]

    mask = rng.random((3,10,10)) < 0.3
    mask[0] = True
    mask[2] = mask[1]
    b, bmask = mean(image, mask=mask)
    assert np.all(bmask == mask[1])
    assert np.all((b == 1.5 * image0)[~bmask])


def test_mean_axis_variants() -> None:
    rng = np.random.default_rng(5965)
    image = rng.random((3,4,5,10,10))

    a = mean(image, axis=1)
    assert a.shape == (3,5,10,10)
    assert np.abs(a - np.mean(image, axis=1)).max() < 1.e-15

    a = mean(image, axis=(1,-1))
    assert a.shape == (3,10,10)
    assert np.abs(a - np.mean(image, axis=(1,2))).max() < 1.e-15

    a = mean(image, axis=None)
    assert a.shape == (10,10)
    assert np.abs(a - np.mean(image, axis=(0,1,2))).max() < 1.e-15


def test_mean_mask_newmask_all_masked(shortcuts: bool) -> None:
    rng = np.random.default_rng(5965)
    image = rng.random((5,4,3,10,10))
    mask = rng.random((5,4,3,10,10)) < 0.9        # mostly masked
    a, amask = mean(image, axis=2, mask=mask)
    assert a.shape == (5,4,10,10)
    assert amask.shape == (5,4,10,10)
    assert np.all(amask == np.all(mask, axis=2))


def test_mean_mask_values_match_numpy(shortcuts: bool) -> None:
    # Value-match vs numpy on a masked reduction; run on both gated paths.
    rng = np.random.default_rng(5965)
    image = rng.random((5,4,100,200))
    mask = rng.random((5,4,100,200)) < 0.6        # mostly masked
    a, amask = mean(image, axis=0, mask=mask)

    sorted_ = image.copy()
    sorted_[mask] = 10
    sorted_ = np.sort(sorted_, axis=0)

    k = np.sum(np.logical_not(mask), axis=0)
    assert np.all((k == 0) == amask)
    assert np.all((a == sorted_[0])[k == 1])
    assert np.abs(a - np.mean(sorted_[:2], axis=0))[k == 2].max() < 1.e-15
    assert np.abs(a - np.mean(sorted_[:3], axis=0))[k == 3].max() < 1.e-15
    assert np.abs(a - np.mean(sorted_[:4], axis=0))[k == 4].max() < 1.e-15
    assert np.abs(a - np.mean(sorted_[:5], axis=0))[k == 5].max() < 1.e-15


##########################################################################################
# mean with factors (linear, so duplication equals integer factors exactly)
##########################################################################################

def test_mean_factors_duplication_equivalence() -> None:
    rng = np.random.default_rng(5965)
    image = rng.random((6,1,1,4,4))
    image[5] = image[4]
    factors = np.array([1,1,1,1,2]).reshape(5,1,1)
    a = mean(image[:5], axis=0, factors=factors)
    b = mean(image, axis=0)
    assert np.abs(a - b).max() < 1.e-15


def test_mean_factors_2d_multi_axis() -> None:
    rng = np.random.default_rng(5965)
    image = rng.random((5,2,6,10,10))
    image[:,0,4] = image[:,0,3]
    image[:,0,5] = image[:,0,3]
    image[:,1,4] = image[:,1,0]
    image[:,1,5] = image[:,1,1]
    factors = np.array([[1,1,1,3],[2,2,1,1]])
    a = mean(image[:,:,:4], axis=(1,2), factors=factors)
    b = mean(image, axis=(1,2))
    assert np.abs(a - b).max() < 1.e-15

    a = mean(image[:,:,:4], axis=2, factors=factors)
    b = mean(image, axis=2)
    assert np.abs(a - b).max() < 1.e-15


def test_mean_factors_3d_axis_0_2() -> None:
    rng = np.random.default_rng(5965)
    image = rng.random((2,5,6,10,10))
    image[0,:,4] = image[0,:,3]
    image[0,:,5] = image[0,:,3]
    image[1,:,4] = image[1,:,0]
    image[1,:,5] = image[1,:,1]
    factors = np.array([[1,1,1,3],[2,2,1,1]]).reshape(2,1,4)
    a = mean(image[:,:,:4], axis=(0,2), factors=factors)
    b = mean(image, axis=(0,2))
    assert np.abs(a - b).max() < 1.e-15

    a = mean(image[:,:,:4], axis=2, factors=factors.reshape(2,1,4))
    b = mean(image, axis=2)
    assert np.abs(a - b).max() < 1.e-15


def test_mean_factors_3d_axis_0_1() -> None:
    rng = np.random.default_rng(5965)
    image = rng.random((2,6,5,10,10))
    image[0,4] = image[0,3]
    image[0,5] = image[0,3]
    image[1,4] = image[1,0]
    image[1,5] = image[1,1]
    factors = np.array([[1,1,1,3],[2,2,1,1]]).reshape(2,4,1)
    a = mean(image[:,:4], axis=(0,1), factors=factors)
    b = mean(image, axis=(0,1))
    assert np.abs(a - b).max() < 1.e-15

    a = mean(image[:,:4], axis=1, factors=factors)
    b = mean(image, axis=1)
    assert np.abs(a - b).max() < 1.e-15


##########################################################################################
# mean with both factors and a mask
##########################################################################################

def test_mean_factors_with_2d_mask() -> None:
    rng = np.random.default_rng(5965)
    image = rng.random((6,5,4,10,10))
    image[5] = image[4]
    factors = np.array([1,1,1,1,2])
    mask = rng.random((10,10)) < 0.3
    a, amask = mean(image[:5], axis=0, factors=factors.reshape(5,1,1), mask=mask)
    b, bmask = mean(image, axis=0, mask=mask)
    assert amask.shape == (5,4,10,10)
    assert np.all(amask == mask)
    assert np.all(amask == bmask)
    assert np.abs(a - b)[~amask].max() < 1.e-15


def test_mean_factors_with_4d_mask() -> None:
    rng = np.random.default_rng(5965)
    image = rng.random((6,5,4,10,10))
    image[5] = image[4]
    factors = np.array([1,1,1,1,2])
    mask = rng.random((5,4,10,10)) < 0.3
    a, amask = mean(image[:5], axis=0, factors=factors.reshape(5,1,1), mask=mask)
    b, bmask = mean(image, axis=0, mask=mask)
    assert amask.shape == (5,4,10,10)
    assert np.all(amask == bmask)
    assert np.abs(a - b)[~amask].max() < 1.e-15


##########################################################################################
# mean with weights, maskval, nans, MaskedArray (covers both shortcut paths)
##########################################################################################

def test_mean_weights(shortcuts: bool) -> None:
    rng = np.random.default_rng(5965)
    image = rng.random((6,8,8))
    weights = rng.random((6,8,8)) + 0.5
    a, aw = mean(image, axis=0, weights=weights)
    expect = np.sum(weights * image, axis=0) / np.sum(weights, axis=0)
    assert np.abs(a - expect).max() < 1.e-14
    assert np.abs(aw - np.sum(weights, axis=0)).max() < 1.e-14


def test_mean_unweighted_no_shortcuts(shortcuts: bool) -> None:
    rng = np.random.default_rng(5965)
    image = rng.random((6,8,8))
    a = mean(image)
    assert np.abs(a - np.mean(image, axis=0)).max() < 1.e-14


def test_mean_mask_no_shortcuts(shortcuts: bool) -> None:
    rng = np.random.default_rng(5965)
    image = rng.random((6,8,8))
    mask = rng.random((6,8,8)) < 0.4
    a, amask = mean(image, mask=mask)
    count = np.sum(np.logical_not(mask), axis=0)
    masked_img = np.where(mask, 0., image)
    with np.errstate(invalid='ignore', divide='ignore'):
        expect = np.sum(masked_img, axis=0) / count
    good = count > 0
    assert np.abs((a - expect)[good]).max() < 1.e-14
    assert np.all(amask == (count == 0))


def test_mean_maskval(shortcuts: bool) -> None:
    rng = np.random.default_rng(5965)
    image = rng.random((4,8,8))
    image[0, 0, 0] = 7.
    a = mean(image, maskval=7.)
    b = mean(image[1:, 0:1, 0:1])
    assert abs(a[0, 0] - b[0, 0]) < 1.e-14


def test_mean_nans(shortcuts: bool) -> None:
    # Regression: both the shortcut and general paths must drop NaN-masked pixels
    # (`nans=True`) rather than letting `0 * NaN` propagate into the weighted sum.
    rng = np.random.default_rng(5965)
    image = rng.random((4,8,8))
    image[1, 2, 2] = np.nan
    a = mean(image, nans=True)
    assert not np.isnan(a).any()
    b = mean(image[np.array([0, 2, 3])][:, 2:3, 2:3])
    assert abs(a[2, 2] - b[0, 0]) < 1.e-14


def test_mean_maskedarray_input(shortcuts: bool) -> None:
    rng = np.random.default_rng(5965)
    image = rng.random((4,8,8))
    m = rng.random((4,8,8)) < 0.3
    ma = np.ma.MaskedArray(image, mask=m)
    a = mean(ma)
    assert isinstance(a, np.ma.MaskedArray)


##########################################################################################
# mean dtypes, axis combinations, list inputs, error path
##########################################################################################

@pytest.mark.parametrize('dtype', ['bool', 'uint8', 'int8', 'uint16', 'int16',
                                   'uint32', 'int32', 'int64', 'float32', 'float64'])
def test_mean_dtypes(dtype: str) -> None:
    image = np.arange(10) + np.arange(10)[:,None] + np.arange(10)[:,None,None]
    test = mean(image.astype(dtype))
    if dtype == 'float32':
        assert test.dtype == np.float32
    else:
        assert test.dtype == np.float64


def test_mean_axis_combinations_match_numpy() -> None:
    rng = np.random.default_rng(5965)
    image = rng.random((2,4,8,10,10))
    assert np.all(mean(image) == np.mean(image, axis=(0,1,2)))
    assert np.all(mean(image, axis=1) == np.mean(image, axis=1))
    assert np.all(mean(image, axis=(0,-1)) == np.mean(image, axis=(0,2)))


def test_mean_keepdims(shortcuts: bool) -> None:
    # keepdims restores reduced axes as length-1 dimensions (exercised via the masked
    # path, which returns both the image and mask).
    rng = np.random.default_rng(5965)
    image = rng.random((4,3,8,8))
    mask = rng.random((4,3,8,8)) < 0.2
    a, amask = mean(image, axis=1, keepdims=True, mask=mask)
    assert a.shape == (4,1,8,8)
    assert amask.shape == (4,1,8,8)


def test_mean_list_input() -> None:
    rng = np.random.default_rng(5965)
    image = list(rng.random((2,4,8,10,10)))     # works for non-arrays?
    assert np.all(mean(image) == np.mean(image, axis=(0,1,2)))


def test_mean_too_few_dimensions() -> None:
    image = np.arange(12).reshape(4,3)
    with pytest.raises(ValueError) as exc_info:
        _ = mean(image)
    assert str(exc_info.value) == 'invalid image shape (4, 3); must be at least 3-D'


##########################################################################################
# mean_filter
##########################################################################################

def test_mean_filter_no_mask(shortcuts: bool) -> None:
    image = np.arange(10) + np.arange(10)[:,None] + np.arange(4)[:,None,None]
    a = mean_filter(image, (2,2))
    b = np.empty((4,10,10))
    b[:,0,0] = image[:,0,0]
    b[:,0,1:] = (image[:,0,:-1] + image[:,0,1:]) / 2.
    b[:,1:,0] = (image[:,:-1,0] + image[:,1:,0]) / 2.
    for i in range(1,10):
        for j in range(1,10):
            b[:,i,j] = np.mean(image[:,i-1:i+1,j-1:j+1], axis=(-2,-1))
    assert np.abs(a - b).max() < 1.e-15


def test_mean_filter_int_footprint() -> None:
    rng = np.random.default_rng(8063)
    image = rng.random((3,20,20))
    a = mean_filter(image, 3)
    b = mean_filter(image, (3, 3))
    assert np.abs(a - b).max() < 1.e-15


def test_mean_filter_mask_irregular_footprint(shortcuts: bool) -> None:
    rng = np.random.default_rng(8063)
    image = rng.random((100,100))
    mask = rng.random((100,100)) < 0.6
    footprint = np.ones((3,3), dtype='bool')
    footprint[0,0] = False
    footprint[0,2] = False
    footprint[2,0] = False
    a, amask = mean_filter(image, footprint=footprint, mask=mask)

    for i in range(100):
        islice1 = slice(max(i-1, 0), min(i+2, 100))
        islice2 = slice(1 if i==0 else 0, 2 if i==99 else 3)
        for j in range(100):
            jslice1 = slice(max(j-1, 0), min(j+2, 100))
            jslice2 = slice(1 if j==0 else 0, 2 if j==99 else 3)
            values = image[islice1,jslice1]
            vmask = mask[islice1,jslice1]

            fp = footprint[islice2,jslice2]
            values = values[~vmask & fp]
            if len(values) == 0:
                assert amask[i,j]
                continue

            assert not amask[i,j]
            assert np.abs(a[i,j] - np.mean(values)) < 1.e-14


def test_mean_filter_weights(shortcuts: bool) -> None:
    # A weighted mean filter must apply the weight values (regression: the shortcut
    # path used to ignore them and return the unweighted mean). Checked on both code
    # paths via the `shortcuts` fixture.
    rng = np.random.default_rng(909)
    image = rng.random((20, 20))
    weights = rng.random((20, 20)) + 0.5

    a, aw = mean_filter(image, 3, weights=weights)
    assert a.shape == (20, 20)

    # Manual weighted mean over the 3x3 window at an interior pixel
    i = j = 10
    win = image[i-1:i+2, j-1:j+2]
    ww = weights[i-1:i+2, j-1:j+2]
    assert np.isclose(a[i, j], np.sum(win * ww) / np.sum(ww))
    assert np.isclose(aw[i, j], ww.sum())

    # The weighted result genuinely differs from the unweighted filter
    plain = mean_filter(image, 3)
    assert not np.isclose(a[i, j], plain[i, j])


def test_mean_zero_size_raises(shortcuts: bool) -> None:
    # A reduction over a zero-size array is undefined and must raise, not NaN.
    empty = np.ones((0, 4, 4))
    with pytest.raises(ValueError, match='size cannot be zero'):
        mean(empty)
    with pytest.raises(ValueError, match='size cannot be zero'):
        mean(empty, mask=np.zeros((0, 4, 4), dtype=bool))

##########################################################################################
