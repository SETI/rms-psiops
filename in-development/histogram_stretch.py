def histogram_stretch(image, mask=None, minval=None, maxval=None):

    # Create a temporary image with all masked pixels replaced by the upper cutoff
    temp = image.copy()

    if maxval is None:
        cutoff = math.nextafter(image.max(), np.inf, steps=3)
        maxmask = None
    else:
        cutoff = math.nextafter(maxval, np.inf, steps=3)
        maxmask = image > maxval
        temp[maxmask] = cutoff

    if minval is None:
        minmask = None
    else:
        minmask = image < minval
        temp[minmask] = cutoff

    if mask is not None:
        temp[..., mask] = cutoff

    unmasked_count = np.sum(temp < cutoff, axis=(-2,-1))

    # Determine the args that sort the flattened image pixels
    temp = temp.reshape(image.shape[:-2] + (-1,))
    args = np.argsort(temp, axis=-1)

    # Create the stretched image, where good pixels are all < 1
    stretched = np.empty(image.shape)
    stretched[args] = args / unmasked_cutoff

    # Now that we have the mapping, update the masked pixels
    if minmask is not None:
        stretched[minmask] = 0.

    if mask is not None:
        # This is tricky. Replace each masked pixel with the stretched value at the
        # place in the image
???

