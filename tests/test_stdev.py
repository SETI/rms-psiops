##########################################################################################
# tests/test_stdev.py
##########################################################################################

import numpy as np
import pytest
from psiops.stdev import stdev, stdev_filter, variance_filter


##########################################################################################
# stdev
##########################################################################################

def test_stdev_basic() -> None:
    array = np.arange(10)
    image0 = array + array[:,np.newaxis]
    image = image0 * np.array([1,2,3])[:,np.newaxis,np.newaxis]

    a = stdev(image)
    assert np.abs(a - image0).max() < 1.e-15


def test_stdev_mask_2d() -> None:
    rng = np.random.default_rng(3534)
    array = np.arange(10)
    image0 = array + array[:,np.newaxis]
    image = image0 * np.array([1,2,3])[:,np.newaxis,np.newaxis]

    a = stdev(image)
    mask = rng.random((10,10)) < 0.3
    b, bmask = stdev(image, mask=mask)
    assert np.all(mask == bmask)
    assert np.shape(bmask) == (10,10)
    assert np.all(a[~mask] == b[~mask])


def test_stdev_mask_3d_one_layer() -> None:
    array = np.arange(10)
    image0 = array + array[:,np.newaxis]
    image = image0 * np.array([1,2,3])[:,np.newaxis,np.newaxis]

    mask = np.zeros(image.shape, dtype='bool')
    mask[0] = True
    b, bmask = stdev(image, mask=mask)
    assert np.abs(b - np.sqrt(0.5) * image0).max() < 1.e-14
    assert not np.any(bmask)


def test_stdev_mask_3d_partial() -> None:
    rng = np.random.default_rng(3534)
    array = np.arange(10)
    image0 = array + array[:,np.newaxis]
    image = image0 * np.array([1,2,3])[:,np.newaxis,np.newaxis]

    mask = rng.random((3,10,10)) < 0.3
    mask[0] = True
    mask[2] = mask[1]
    b, bmask = stdev(image, mask=mask)
    assert np.all(bmask == mask[1])
    assert np.abs(b - np.sqrt(0.5) * image0)[~bmask].max() < 1.e-14


def test_stdev_mask_3d_first_layer_only() -> None:
    rng = np.random.default_rng(3534)
    array = np.arange(10)
    image0 = array + array[:,np.newaxis]
    image = image0 * np.array([1,2,3])[:,np.newaxis,np.newaxis]

    mask = rng.random((3,10,10)) < 0.3
    mask[1] = False
    mask[2] = False
    b, bmask = stdev(image, mask=mask)
    assert not np.any(bmask)
    assert np.abs(b - image0)[~mask[0]].max() < 1.e-14
    assert np.abs(b - np.sqrt(0.5) * image0)[mask[0]].max() < 1.e-14


def test_stdev_axis_variants() -> None:
    rng = np.random.default_rng(3534)
    image = rng.random((3,4,5,10,10))

    a = stdev(image, axis=1)
    assert a.shape == (3,5,10,10)
    assert np.abs(a - np.std(image, axis=1, ddof=1)).max() < 1.e-15

    a = stdev(image, axis=(1,-1))
    assert a.shape == (3,10,10)
    assert np.abs(a - np.std(image, axis=(1,2), ddof=1)).max() < 1.e-15

    a = stdev(image, axis=None)
    assert a.shape == (10,10)
    assert np.abs(a - np.std(image, axis=(0,1,2), ddof=1)).max() < 1.e-15


def test_stdev_mask_newmask_reliability() -> None:
    # Default stdtype is 'reliability'; an unweighted mask makes the reliability denom
    # equal to (count - 1), so a pixel is masked exactly where fewer than two values
    # contribute.
    rng = np.random.default_rng(3534)
    image = rng.random((5,4,3,10,10))
    mask = rng.random((5,4,3,10,10)) < 0.7
    count = np.sum(np.logical_not(mask), axis=2)

    a, amask = stdev(image, axis=2, mask=mask)
    assert a.shape == (5,4,10,10)
    assert amask.shape == (5,4,10,10)
    assert np.all(amask == (count < 2))


def test_stdev_mask_newmask_unbiased() -> None:
    rng = np.random.default_rng(3534)
    image = rng.random((5,4,3,10,10))
    mask = rng.random((5,4,3,10,10)) < 0.7
    count = np.sum(np.logical_not(mask), axis=2)

    a, amask = stdev(image, axis=2, mask=mask, stdtype='unbiased')
    assert amask.shape == (5,4,10,10)
    assert np.all(amask == (count < 2))


def test_stdev_mask_newmask_frequency() -> None:
    rng = np.random.default_rng(3534)
    image = rng.random((5,4,3,10,10))
    mask = rng.random((5,4,3,10,10)) < 0.7
    count = np.sum(np.logical_not(mask), axis=2)

    a, amask = stdev(image, axis=2, mask=mask, stdtype='frequency')
    assert np.all(amask == (count < 2))


def test_stdev_mask_newmask_biased() -> None:
    # The biased estimator only requires one contributing value, so a pixel is masked
    # only where every value is masked.
    rng = np.random.default_rng(3534)
    image = rng.random((5,4,3,10,10))
    mask = rng.random((5,4,3,10,10)) < 0.7
    count = np.sum(np.logical_not(mask), axis=2)

    a, amask = stdev(image, axis=2, mask=mask, stdtype='biased')
    assert np.all(amask == (count == 0))


def test_stdev_mask_values_match_numpy() -> None:
    rng = np.random.default_rng(3534)
    image = rng.random((5,4,100,200))
    mask = rng.random((5,4,100,200)) < 0.6        # mostly masked
    a, amask = stdev(image, axis=0, mask=mask)

    sorted_ = image.copy()
    sorted_[mask] = 10
    sorted_ = np.sort(sorted_, axis=0)

    k = np.sum(np.logical_not(mask), axis=0)
    assert np.all(((k == 0) | (k == 1)) == amask)
    assert np.abs(a - np.std(sorted_[:2], axis=0, ddof=1))[k == 2].max() < 1.e-14
    assert np.abs(a - np.std(sorted_[:3], axis=0, ddof=1))[k == 3].max() < 1.e-14
    assert np.abs(a - np.std(sorted_[:4], axis=0, ddof=1))[k == 4].max() < 1.e-14
    assert np.abs(a - np.std(sorted_[:5], axis=0, ddof=1))[k == 5].max() < 1.e-14


##########################################################################################
# stdev with factors (frequency / unbiased estimator)
##########################################################################################

def test_stdev_factors_frequency_first_principles() -> None:
    # With 'frequency', each integer factor counts as that many measurements: the divisor
    # is sum(factors) - 1.
    rng = np.random.default_rng(99)
    values = rng.random(5)
    factors = np.array([1, 1, 0, 4, 2])
    image = np.empty((5, 3, 3))
    image[...] = values[:, np.newaxis, np.newaxis]

    a = stdev(image, factors=factors, stdtype='frequency')
    sw = np.sum(factors)
    mean = np.sum(factors * values) / sw
    expect = np.sqrt(np.sum(factors * (values - mean)**2) / (sw - 1))
    assert abs(a[0, 0] - expect) < 1.e-14


def test_stdev_factors_duplication_equivalence() -> None:
    # Under 'frequency' semantics, an integer factor counts as that many measurements,
    # so factor=2 on a layer is exactly equivalent to duplicating that layer.
    rng = np.random.default_rng(3534)
    image = rng.random((6,5,4,10,10))
    image[5] = image[4]
    factors = np.array([1,1,1,1,2]).reshape(5,1,1)
    a = stdev(image[:5], axis=0, factors=factors, stdtype='frequency')
    b = stdev(image, axis=0, stdtype='frequency')
    assert np.abs(a - b).max() < 1.e-14


def test_stdev_factors_zero_weights() -> None:
    rng = np.random.default_rng(3534)
    image = rng.random((6,5,4,10,10))
    image[5] = image[4]
    factors = np.array([0,0,1,1,2]).reshape(5,1,1)      # check zero weights
    a = stdev(image[:5], axis=0, factors=factors, stdtype='frequency')
    b = stdev(image[2:], axis=0, stdtype='frequency')
    assert np.abs(a - b).max() < 1.e-14


def test_stdev_factors_2d_multi_axis() -> None:
    rng = np.random.default_rng(3534)
    image = rng.random((5,2,6,10,10))
    image[:,0,4] = image[:,0,3]
    image[:,0,5] = image[:,0,3]
    image[:,1,4] = image[:,1,0]
    image[:,1,5] = image[:,1,1]
    factors = np.array([[1,1,1,3],[2,2,1,1]])
    a = stdev(image[:,:,:4], axis=(1,2), factors=factors, stdtype='frequency')
    b = stdev(image, axis=(1,2), stdtype='frequency')
    assert np.abs(a - b).max() < 1.e-14


def test_stdev_factors_3d_axis_0_2() -> None:
    rng = np.random.default_rng(3534)
    image = rng.random((2,5,6,10,10))
    image[0,:,4] = image[0,:,3]
    image[0,:,5] = image[0,:,3]
    image[1,:,4] = image[1,:,0]
    image[1,:,5] = image[1,:,1]
    factors = np.array([[1,1,1,3],[2,2,1,1]]).reshape(2,1,4)
    a = stdev(image[:,:,:4], axis=(0,2), factors=factors, stdtype='frequency')
    b = stdev(image, axis=(0,2), stdtype='frequency')
    assert np.abs(a - b).max() < 1.e-14


def test_stdev_factors_3d_axis_0_1() -> None:
    rng = np.random.default_rng(3534)
    image = rng.random((2,6,5,10,10))
    image[0,4] = image[0,3]
    image[0,5] = image[0,3]
    image[1,4] = image[1,0]
    image[1,5] = image[1,1]
    factors = np.array([[1,1,1,3],[2,2,1,1]]).reshape(2,4,1)
    a = stdev(image[:,:4], axis=(0,1), factors=factors, stdtype='frequency')
    b = stdev(image, axis=(0,1), stdtype='frequency')
    assert np.abs(a - b).max() < 1.e-14


##########################################################################################
# stdev with factors, stdtype='reliability' (cross-checked against numpy.cov aweights)
##########################################################################################

def test_stdev_factors_reliability_vs_cov_a() -> None:
    rng = np.random.default_rng(3534)
    image = rng.random((8,4,4))
    factors = np.array([1,0,1,1,2,3,0,2])
    a = stdev(image, axis=0, factors=factors, stdtype='reliability')
    b = np.empty((4,4))
    for i in range(4):
        for j in range(4):
            b[i,j] = np.sqrt(np.cov(image[:,i,j], aweights=factors))
    assert np.abs(a - b).max() < 1.e-15


def test_stdev_factors_reliability_vs_cov_b() -> None:
    rng = np.random.default_rng(3534)
    image = rng.random((8,4,4))
    factors = np.array([1,1,2,1,1,0,1,20])
    a = stdev(image, axis=0, factors=factors, stdtype='reliability')
    b = np.empty((4,4))
    for i in range(4):
        for j in range(4):
            b[i,j] = np.sqrt(np.cov(image[:,i,j], aweights=factors))
    assert np.abs(a - b).max() < 1.e-15


##########################################################################################
# stdev with both weights/factors and a mask
##########################################################################################

def test_stdev_factors_with_mask() -> None:
    # 'frequency' factor=2 duplicates a layer, combined with a 2-D mask.
    rng = np.random.default_rng(3534)
    image = rng.random((6,5,4,10,10))
    image[5] = image[4]
    factors = np.array([1,1,1,1,2]).reshape(5,1,1)
    mask = rng.random((10,10)) < 0.3
    a, amask = stdev(image[:5], axis=0, factors=factors, mask=mask, stdtype='frequency')
    b, bmask = stdev(image, axis=0, mask=mask, stdtype='frequency')
    assert np.all(amask == mask)
    assert np.all(amask == bmask)
    assert np.abs(a - b)[~amask].max() < 1.e-14


def test_stdev_factors_with_3d_mask() -> None:
    rng = np.random.default_rng(3534)
    image = rng.random((8,10,10))
    factors = np.array([1,1,2,1,1,0,1,20])
    mask = rng.random((8,10,10)) < 0.3
    mask[:] = mask[0]
    mask[1:3] = True
    a, amask = stdev(image, factors=factors, mask=mask)
    b = stdev(image, factors=[1,0,0,1,1,0,1,20])
    assert np.abs(a - b)[~amask].max() < 1.e-15


##########################################################################################
# stdev dtypes, list inputs, error paths
##########################################################################################

@pytest.mark.parametrize('dtype', ['bool', 'uint8', 'int8', 'uint16', 'int16',
                                   'uint32', 'int32', 'int64', 'float32', 'float64'])
def test_stdev_dtypes(dtype: str) -> None:
    image = np.arange(10) + np.arange(10)[:,None] + np.arange(10)[:,None,None]
    test = stdev(image.astype(dtype))
    if dtype == 'float32':
        assert test.dtype == np.float32
    else:
        assert test.dtype == np.float64


def test_stdev_axis_combinations_match_numpy() -> None:
    rng = np.random.default_rng(3534)
    image = rng.random((2,3,4,10,10))
    assert np.abs(stdev(image)
                  - np.std(image, axis=(0,1,2), ddof=1)).max() < 1.e-14
    assert np.abs(stdev(image, axis=1)
                  - np.std(image, axis=1, ddof=1)).max() < 1.e-14
    assert np.abs(stdev(image, axis=(0,-1))
                  - np.std(image, axis=(0,2), ddof=1)).max() < 1.e-14


def test_stdev_list_input() -> None:
    rng = np.random.default_rng(3534)
    image = list(rng.random((2,3,4,10,10)))     # works for non-arrays?
    assert np.abs(stdev(image)
                  - np.std(image, axis=(0,1,2), ddof=1)).max() < 1.e-14


def test_stdev_keepdims() -> None:
    rng = np.random.default_rng(3534)
    image = rng.random((4,3,8,8))
    mask = rng.random((4,3,8,8)) < 0.2
    a, amask = stdev(image, axis=1, keepdims=True, mask=mask)
    assert a.shape == (4,1,8,8)
    assert amask.shape == (4,1,8,8)


def test_stdev_bad_stdtype() -> None:
    rng = np.random.default_rng(3534)
    image = rng.random((2,3,4,10,10))
    with pytest.raises(ValueError) as exc_info:
        _ = stdev(image, stdtype='huh?')
    assert str(exc_info.value) == "unrecognized value for stdtype: 'huh?'"


def test_stdev_too_few_dimensions() -> None:
    image = np.arange(12).reshape(4,3)
    with pytest.raises(ValueError) as exc_info:
        _ = stdev(image)
    assert str(exc_info.value) == 'invalid image shape (4, 3); must be at least 3-D'


##########################################################################################
# stdev_filter
##########################################################################################

def test_stdev_filter_no_mask() -> None:
    image = np.arange(10) + np.arange(10)[:,None] + np.arange(4)[:,None,None]
    a = stdev_filter(image, 2)
    b = np.empty((4,10,10))
    # The (0,0) corner has only one contributing pixel, so the sample standard deviation
    # is undefined and the result is NaN.
    b[:,0,0] = np.nan
    b[:,0,1:] = np.abs(image[:,0,:-1] - image[:,0,1:]) / np.sqrt(2)
    b[:,1:,0] = np.abs(image[:,:-1,0] - image[:,1:,0]) / np.sqrt(2)
    for i in range(1,10):
        for j in range(1,10):
            b[:,i,j] = np.std(image[:,i-1:i+1,j-1:j+1], axis=(-2,-1), ddof=1)
    assert np.all(np.isnan(a[:,0,0]))
    diff = np.abs(a - b)
    diff[:,0,0] = 0
    assert diff.max() < 1.e-14


def test_stdev_filter_int_footprint() -> None:
    rng = np.random.default_rng(8063)
    image = rng.random((3,20,20))
    a = stdev_filter(image, 3)
    b = stdev_filter(image, (3, 3))
    assert np.all(np.isnan(a) == np.isnan(b))
    assert np.abs((a - b)[~np.isnan(a)]).max() < 1.e-15


def test_stdev_filter_mask_irregular_footprint() -> None:
    rng = np.random.default_rng(8063)
    image = rng.random((100,100))
    mask = rng.random((100,100)) < 0.4
    footprint = np.ones((3,3), dtype='bool')
    footprint[0,0] = False
    footprint[0,2] = False
    footprint[2,0] = False
    a, amask = stdev_filter(image, footprint=footprint, mask=mask)
    b, bmask = variance_filter(image, footprint=footprint, mask=mask)

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
            if len(values) < 2:
                assert amask[i,j]
                assert bmask[i,j]
                continue

            assert not amask[i,j]
            assert not bmask[i,j]
            assert np.abs(a[i,j] - np.std(values, ddof=1)) < 1.e-14
            assert np.abs(b[i,j] - np.var(values, ddof=1)) < 1.e-14


def test_stdev_filter_bad_stdtype() -> None:
    image = np.arange(48.).reshape(3,4,4)
    with pytest.raises(ValueError) as exc_info:
        _ = stdev_filter(image, 3, stdtype='nope')
    assert str(exc_info.value) == "unrecognized value for stdtype: 'nope'"

##########################################################################################
