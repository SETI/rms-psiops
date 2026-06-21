##########################################################################################
# tests/test_variance.py
##########################################################################################

import numpy as np
import pytest

from psiops.variance import variance, variance_filter

##########################################################################################
# variance: unweighted, matches numpy.var
##########################################################################################

def test_variance_basic() -> None:
    array = np.arange(10)
    image0 = array + array[:,np.newaxis]
    image = image0 * np.array([1,2,3])[:,np.newaxis,np.newaxis]
    # Default 'reliability' on uniform unweighted data == ddof=1.
    a = variance(image)
    assert np.abs(a - image0**2).max() < 1.e-12


@pytest.mark.parametrize(('vartype', 'ddof'), [('biased', 0), ('unbiased', 1),
                                               ('frequency', 1), ('reliability', 1)])
def test_variance_vartypes_match_numpy(vartype: str, ddof: int) -> None:
    rng = np.random.default_rng(11)
    image = rng.random((6,10,10))
    a = variance(image, vartype=vartype)
    assert np.abs(a - np.var(image, axis=0, ddof=ddof)).max() < 1.e-14


def test_variance_axis_variants() -> None:
    rng = np.random.default_rng(11)
    image = rng.random((3,4,5,10,10))

    a = variance(image, axis=1)
    assert a.shape == (3,5,10,10)
    assert np.abs(a - np.var(image, axis=1, ddof=1)).max() < 1.e-14

    a = variance(image, axis=(1,-1))
    assert a.shape == (3,10,10)
    assert np.abs(a - np.var(image, axis=(1,2), ddof=1)).max() < 1.e-14

    a = variance(image, axis=None)
    assert a.shape == (10,10)
    assert np.abs(a - np.var(image, axis=(0,1,2), ddof=1)).max() < 1.e-14


def test_variance_negative_axis() -> None:
    rng = np.random.default_rng(11)
    image = rng.random((2,3,4,8,8))
    a = variance(image, axis=-1, vartype='biased')   # third-to-last axis
    assert np.abs(a - np.var(image, axis=2, ddof=0)).max() < 1.e-14


def test_variance_keepdims() -> None:
    rng = np.random.default_rng(11)
    image = rng.random((4,3,8,8))
    mask = rng.random((4,3,8,8)) < 0.2
    a, amask = variance(image, axis=1, keepdims=True, mask=mask)
    assert a.shape == (4,1,8,8)
    assert amask.shape == (4,1,8,8)


def test_variance_unweighted_no_shortcuts(shortcuts: bool) -> None:
    # Exercise both the optimized shortcut path and the general unweighted path.
    rng = np.random.default_rng(11)
    image = rng.random((6,10,10))
    a = variance(image, vartype='unbiased')
    assert np.abs(a - np.var(image, axis=0, ddof=1)).max() < 1.e-13
    b = variance(image, vartype='biased')
    assert np.abs(b - np.var(image, axis=0, ddof=0)).max() < 1.e-13


def test_variance_unweighted_multi_axis_no_shortcuts(shortcuts: bool) -> None:
    rng = np.random.default_rng(11)
    image = rng.random((3,4,10,10))
    a = variance(image, axis=(0, 1), vartype='reliability')
    assert np.abs(a - np.var(image, axis=(0, 1), ddof=1)).max() < 1.e-13


##########################################################################################
# variance: masked
##########################################################################################

def test_variance_mask_2d() -> None:
    rng = np.random.default_rng(11)
    array = np.arange(10)
    image0 = array + array[:,np.newaxis]
    image = image0 * np.array([1,2,3])[:,np.newaxis,np.newaxis]

    a = variance(image)
    mask = rng.random((10,10)) < 0.3
    b, bmask = variance(image, mask=mask)
    assert np.all(mask == bmask)
    assert np.all(a[~mask] == b[~mask])


def test_variance_mask_newmask_reliability() -> None:
    rng = np.random.default_rng(11)
    image = rng.random((5,4,3,10,10))
    mask = rng.random((5,4,3,10,10)) < 0.7
    count = np.sum(np.logical_not(mask), axis=2)
    _, amask = variance(image, axis=2, mask=mask)
    assert np.all(amask == (count < 2))


def test_variance_mask_newmask_biased() -> None:
    rng = np.random.default_rng(11)
    image = rng.random((5,4,3,10,10))
    mask = rng.random((5,4,3,10,10)) < 0.7
    count = np.sum(np.logical_not(mask), axis=2)
    _, amask = variance(image, axis=2, mask=mask, vartype='biased')
    assert np.all(amask == (count == 0))


def test_variance_mask_values_match_numpy() -> None:
    rng = np.random.default_rng(11)
    image = rng.random((5,4,100,200))
    mask = rng.random((5,4,100,200)) < 0.6
    a, amask = variance(image, axis=0, mask=mask)

    sorted_ = image.copy()
    sorted_[mask] = 10
    sorted_ = np.sort(sorted_, axis=0)

    k = np.sum(np.logical_not(mask), axis=0)
    assert np.all(((k == 0) | (k == 1)) == amask)
    assert np.abs(a - np.var(sorted_[:2], axis=0, ddof=1))[k == 2].max() < 1.e-13
    assert np.abs(a - np.var(sorted_[:3], axis=0, ddof=1))[k == 3].max() < 1.e-13
    assert np.abs(a - np.var(sorted_[:5], axis=0, ddof=1))[k == 5].max() < 1.e-13


##########################################################################################
# variance: weights / factors
##########################################################################################

def test_variance_weights_vs_cov() -> None:
    rng = np.random.default_rng(11)
    image = rng.random((8,4,4))
    weights = rng.random((8,4,4)) + 0.5
    a, aw = variance(image, axis=0, weights=weights, vartype='reliability')
    b = np.empty((4,4))
    for i in range(4):
        for j in range(4):
            b[i,j] = np.cov(image[:,i,j], aweights=weights[:,i,j])
    assert np.abs(a - b).max() < 1.e-14
    assert np.abs(aw - np.sum(weights, axis=0)).max() < 1.e-14


def test_variance_weights_biased() -> None:
    rng = np.random.default_rng(11)
    image = rng.random((8,5,5))
    weights = rng.random((8,5,5)) + 0.5
    a = variance(image, axis=0, weights=weights, vartype='biased', returns='i')
    for i in range(5):
        for j in range(5):
            w = weights[:,i,j]
            v = image[:,i,j]
            mean = np.sum(w*v)/np.sum(w)
            expect = np.sum(w*(v-mean)**2)/np.sum(w)
            assert abs(a[i,j] - expect) < 1.e-14


def test_variance_factors_frequency_duplication() -> None:
    rng = np.random.default_rng(11)
    image = rng.random((6,5,4,10,10))
    image[5] = image[4]
    factors = np.array([1,1,1,1,2]).reshape(5,1,1)
    a = variance(image[:5], axis=0, factors=factors, vartype='frequency')
    b = variance(image, axis=0, vartype='frequency')
    assert np.abs(a - b).max() < 1.e-13


def test_variance_factors_multidim_broadcast() -> None:
    # A 2-D factor array broadcasts over two reduced axes.
    rng = np.random.default_rng(11)
    image = rng.random((5,2,6,10,10))
    image[:,0,4] = image[:,0,3]
    image[:,0,5] = image[:,0,3]
    image[:,1,4] = image[:,1,0]
    image[:,1,5] = image[:,1,1]
    factors = np.array([[1,1,1,3],[2,2,1,1]])
    a = variance(image[:,:,:4], axis=(1,2), factors=factors, vartype='frequency')
    b = variance(image, axis=(1,2), vartype='frequency')
    assert np.abs(a - b).max() < 1.e-13


##########################################################################################
# variance: maskval, nans, MaskedArray inputs
##########################################################################################

def test_variance_maskval() -> None:
    rng = np.random.default_rng(11)
    image = rng.random((4,8,8))
    image[0, 0, 0] = 99.
    a, amask = variance(image, maskval=99., returns='im')
    assert not amask[0, 0]
    b = variance(image[1:, 0:1, 0:1])
    assert abs(a[0, 0] - b[0, 0]) < 1.e-13


def test_variance_nans() -> None:
    rng = np.random.default_rng(11)
    image = rng.random((4,8,8))
    image[1, 2, 2] = np.nan
    a = variance(image, nans=True)
    assert not np.isnan(a).any()
    b = variance(image[np.array([0, 2, 3])][:, 2:3, 2:3])
    assert abs(a[2, 2] - b[0, 0]) < 1.e-13


def test_variance_maskedarray_input() -> None:
    rng = np.random.default_rng(11)
    image = rng.random((4,8,8))
    m = rng.random((4,8,8)) < 0.3
    ma = np.ma.MaskedArray(image, mask=m)
    a = variance(ma)
    assert isinstance(a, np.ma.MaskedArray)
    b, bmask = variance(image, mask=m)
    assert np.abs(np.asarray(a)[~bmask] - b[~bmask]).max() < 1.e-13


##########################################################################################
# variance: dtypes, list input, error paths
##########################################################################################

@pytest.mark.parametrize('dtype', ['bool', 'uint8', 'int8', 'uint16', 'int16',
                                   'uint32', 'int32', 'int64', 'float32', 'float64'])
def test_variance_dtypes(dtype: str) -> None:
    image = np.arange(10) + np.arange(10)[:,None] + np.arange(10)[:,None,None]
    test = variance(image.astype(dtype))
    if dtype == 'float32':
        assert test.dtype == np.float32
    else:
        assert test.dtype == np.float64


def test_variance_list_input() -> None:
    rng = np.random.default_rng(11)
    image = list(rng.random((2,3,4,10,10)))
    assert np.abs(variance(image)
                  - np.var(image, axis=(0,1,2), ddof=1)).max() < 1.e-13


def test_variance_bad_vartype() -> None:
    rng = np.random.default_rng(11)
    image = rng.random((2,3,4,10,10))
    with pytest.raises(ValueError) as exc_info:
        _ = variance(image, vartype='huh?')
    assert str(exc_info.value) == "unrecognized value for vartype: 'huh?'"


def test_variance_too_few_dimensions() -> None:
    image = np.arange(12).reshape(4,3)
    with pytest.raises(ValueError) as exc_info:
        _ = variance(image)
    assert str(exc_info.value) == 'invalid image shape (4, 3); must be at least 3-D'


##########################################################################################
# variance_filter
##########################################################################################

def test_variance_filter_no_mask() -> None:
    image = np.arange(10) + np.arange(10)[:,None] + np.arange(4)[:,None,None]
    a = variance_filter(image, 2)
    b = np.empty((4,10,10))
    # The (0,0) corner has only one contributing pixel: undefined sample variance -> NaN.
    b[:,0,0] = np.nan
    b[:,0,1:] = np.var(np.stack([image[:,0,:-1], image[:,0,1:]]), axis=0, ddof=1)
    b[:,1:,0] = np.var(np.stack([image[:,:-1,0], image[:,1:,0]]), axis=0, ddof=1)
    for i in range(1,10):
        for j in range(1,10):
            b[:,i,j] = np.var(image[:,i-1:i+1,j-1:j+1], axis=(-2,-1), ddof=1)
    assert np.all(np.isnan(a[:,0,0]))
    diff = np.abs(a - b)
    diff[:,0,0] = 0
    assert diff.max() < 1.e-13


def test_variance_filter_footprint_int_tuple_bool_equivalent() -> None:
    rng = np.random.default_rng(5)
    image = rng.random((3,15,15))
    a = variance_filter(image, 3)
    b = variance_filter(image, (3, 3))
    c = variance_filter(image, np.ones((3, 3), dtype=bool))
    nan = np.isnan(a)
    assert np.all(nan == np.isnan(b))
    assert np.all(nan == np.isnan(c))
    assert np.abs((a - b)[~nan]).max() < 1.e-15
    assert np.abs((a - c)[~nan]).max() < 1.e-15


def test_variance_filter_biased() -> None:
    rng = np.random.default_rng(5)
    image = rng.random((2,15,15))
    a = variance_filter(image, 3, vartype='biased')
    # With a full 3x3 footprint and constant-zero padding outside, interior pixels match
    # the biased variance over the 3x3 window.
    for i in range(1, 14):
        for j in range(1, 14):
            window = image[:, i-1:i+2, j-1:j+2]
            expect = np.var(window, axis=(-2, -1), ddof=0)
            assert np.abs(a[:, i, j] - expect).max() < 1.e-13


def test_variance_filter_weights(shortcuts) -> None:
    # A weighted filter must apply the weight values (regression: the shortcut path
    # used to ignore them and return the unweighted variance). Checked on both the
    # shortcut and general code paths via the `shortcuts` fixture.
    rng = np.random.default_rng(5)
    image = rng.random((20, 20))
    weights = rng.random((20, 20)) + 0.5

    a, aw = variance_filter(image, 3, weights=weights)
    assert a.shape == (20, 20)
    assert aw.shape == (20, 20)

    # Manual reliability-weighted variance over the 3x3 window at an interior pixel
    i = j = 10
    win = image[i-1:i+2, j-1:j+2].ravel()
    ww = weights[i-1:i+2, j-1:j+2].ravel()
    sw = ww.sum()
    mean = np.sum(ww * win) / sw
    denom = sw - np.sum(ww**2) / sw
    expect = np.sum(ww * (win - mean)**2) / denom
    assert np.isclose(a[i, j], expect)
    assert np.isclose(aw[i, j], sw)

    # The weighted result genuinely differs from the unweighted filter
    plain = variance_filter(image, 3, returns='i')
    assert not np.isclose(a[i, j], plain[i, j])


def test_variance_filter_mask_irregular_footprint() -> None:
    rng = np.random.default_rng(8063)
    image = rng.random((100,100))
    mask = rng.random((100,100)) < 0.4
    footprint = np.ones((3,3), dtype='bool')
    footprint[0,0] = False
    footprint[0,2] = False
    footprint[2,0] = False
    a, amask = variance_filter(image, footprint=footprint, mask=mask)

    for i in range(0, 100, 7):
        islice1 = slice(max(i-1, 0), min(i+2, 100))
        islice2 = slice(1 if i==0 else 0, 2 if i==99 else 3)
        for j in range(0, 100, 7):
            jslice1 = slice(max(j-1, 0), min(j+2, 100))
            jslice2 = slice(1 if j==0 else 0, 2 if j==99 else 3)
            values = image[islice1,jslice1]
            vmask = mask[islice1,jslice1]
            fp = footprint[islice2,jslice2]
            values = values[~vmask & fp]
            if len(values) < 2:
                assert amask[i,j]
                continue
            assert not amask[i,j]
            assert np.abs(a[i,j] - np.var(values, ddof=1)) < 1.e-13


def test_variance_filter_bad_vartype() -> None:
    image = np.arange(48.).reshape(3,4,4)
    with pytest.raises(ValueError) as exc_info:
        _ = variance_filter(image, 3, vartype='nope')
    assert str(exc_info.value) == "unrecognized value for vartype: 'nope'"

##########################################################################################
