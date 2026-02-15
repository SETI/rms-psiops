##########################################################################################
# psiops/fft.py
##########################################################################################

import numpy as np
from scipy import fftpack

from gaussian_filter import gaussian_filter


def fft(image, *, retile=False, real=False):
    """The FFT of an image. This is a wrapper of scipy.fftpack.fft2().

    Parameters:
        image (array): Image array, in which the last two axes are the spatial dimensions.
        retile (bool, optional): True to re-tile the returned image. This places the
            constant (0,0) position at the center of the image rather than at the corner.
        real (bool, optional): True to return only the real component.

    Returns:
        (array): The FFT of `image`.
    """

    if image.ndim == 2:
        return _retile(fftpack.fft2(image), retile=retile, real=real, dest=None)

    dest = np.empty(image.shape, dtype=(np.float64 if real else np.complex128))
    for k in np.nditer(image.shape[:-2]):
        _ = _retile(fftpack.fft2(image[k]), retile=retile, real=real, dest=dest[k])

    return dest


def ifft(image, *, retile=False, real=False):
    """The inverse FFT of an image. This is a wrapper of scipy.fftpack.ifft2().

    Parameters:
        image (array): Image array, in which the last two axes are the spatial dimensions.
        retile (bool, optional): True to re-tile the returned image. This places the
            constant (0,0) position at the center of the image rather than at the corner.
        real (bool, optional): True to return only the real component.

    Returns:
        (array): The inverse FFT of `image`.
    """

    if image.ndim == 2:
        return _retile(fftpack.ifft2(image), retile=retile, dest=None)

    dest = np.empty(image.shape, dtype=(np.float64 if real else np.complex128))
    for k in np.nditer(image.shape[:-2]):
        _ = _retile(fftpack.ifft2(image[k]), retile=retile, dest=dest[k])

    return dest


def fft_power(image, retile=False):
    """FFT power of an image.

    Parameters:
        image (array): Image array, in which the last two axes are the spatial dimensions.
        retile (bool, optional): True to re-tile the returned image. This places the
            constant (0,0) position at the center of the image rather than at the corner.

    Returns:
        (array): The FFT power in `image`.
    """

    image = np.asarray(image)
    image_fft = fft(image)
    return _retile(image_fft * np.conj(image_fft), retile=retile, real=True)


def correlate(image, reference, *, normalize=False, retile=False):
    """2-D correlation function between an image and a reference image.

    Parameters:
        image (array): Image array, in which the last two axes are the spatial dimensions.
        reference (array): The reference image.
        normalize (bool, optional): True to normalize the correlation values to the range
            -1 to 1.
        retile (bool, optional): True to re-tile the returned image. This places the
            constant (0,0) position at the center of the image rather than at the corner.

    Returns:
        (array): The real correlation array. The returned shape is the result of
            broadcasting the shapes of the input image and reference.
    """

    image = np.asarray(image)
    image_fft = fftpack.fft2(image)
    reference_fft = fftpack.fft2(reference)
    corr = _retile(fftpack.ifft2(image_fft * np.conj(reference_fft)), retile=retile,
                   real=True)

    if normalize:
        amp = np.sqrt(np.sum(image**2, axis=(-1,-2)) * np.sum(reference**2, axis=(-1,-2)))
        corr /= amp[..., np.newaxis, np.newaxis]

    return corr


def autocorrelate(image, retile=False):
    """2-D autocorrelation function for an image.

    Parameters:
        image (array): Image array, in which the last two axes are the spatial dimensions.
        retile (bool, optional): True to re-tile the returned image. This places the
            constant (0,0) position at the center of the image rather than at the corner.

    Returns:
        (array): The real autocorrelation array.
    """

    image_fft = fftpack.fft2(image)
    corr = fftpack.ifft2(image_fft * np.conj(image_fft)).real
    corr /= corr[0,0]

    return _retile(corr, retile=retile)


def ialign(image, reference, sigma):
    """Integer offset required to align two images, based on the location of maximum
    correlation.

    Parameters:
        image (array): Image array, in which the last two axes are the spatial dimensions.
        reference (array): The reference image.
        sigma (scalar): The sigma to use for unsharp masking each image prior to the
            alignment.

    Returns:
        tuple: Two integers such that `ishift(image, offset, mode='wrap')` provides the
            best match to the reference image.
    """

    if sigma:
        image1 = image - gaussian_filter(image, sigma)
        image2 = reference - gaussian_filter(reference, sigma)
    else:
        image1 = image
        image2 = reference

    shape = image1.shape

    edge = (np.sum(image1[0]) + np.sum(image1[-1]) + np.sum(image1[1:-1,0])
            + np.sum(image1[1:-1,-1])) / (2 * (shape[0] + shape[1]) - 4)
    enlarged1 = np.zeros((2*shape[0], 2*shape[1]))
    enlarged1[:shape[0],:shape[1]] = image1 - edge

    edge = (np.sum(image2[0]) + np.sum(image2[-1]) + np.sum(image2[1:-1,0])
            + np.sum(image2[1:-1,-1])) / (2 * (shape[0] + shape[1]) - 4)
    enlarged2 = np.zeros((2*shape[0], 2*shape[1]))
    enlarged2[:shape[0],:shape[1]] = image2 - edge

    corr = correlate(image2, image1)
    offset = np.where(corr == corr.max())
    di = offset[0][0]
    dj = offset[1][0]

    if di > (shape[0] + 1) // 2:
        di -= shape[0]

    if dj > (shape[1] + 1) // 2:
        dj -= shape[1]

    return (di, dj)


def _retile(image, retile=False, real=False, dest=None):
    """Re-tile if necessary; convert to real if necessary; write to destination if
    provided.
    """

    if real and image.dtype.kind == 'c':
        image = image.real

    if retile:
        if dest is None:
            dest = np.empty(image.shape, dtype=image.dtype)

        x = (image.shape[-2] + 1) // 2
        y = (image.shape[-1] + 1) // 2
        dest[...,   :x,   :y] = image[..., -x: , -y: ]
        dest[...,   :x, -y: ] = image[..., -x: ,   :y]
        dest[..., -x: ,   :y] = image[...,   :x, -y: ]
        dest[..., -x: , -y: ] = image[...,   :x,   :y]
        return dest

    elif dest is None:
        return image

    dest[...] = image[...]
    return dest

##########################################################################################
