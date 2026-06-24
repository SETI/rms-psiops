# rms-psiops

Photometrically accurate Science Image Operations.

<!-- start-after-point -->

## Overview

`psiops` is a Python library of image-processing routines built on NumPy and
SciPy. Unlike general-purpose image tools, every operation is designed to
preserve the *photometric* content of an image: the total signal in any region is
conserved (or scaled in a precisely known way) as the image is shifted, rotated,
zoomed, or resampled. This makes the library suitable for quantitative scientific
work, where the number of photons in a feature matters and not just its
appearance.

## Features

- **Spatial transforms** that conserve flux: `shift`, `ishift`, `zoom`, `unzoom`,
  `resample`, `reshape`, and `rotate`.
- **Spatial filters**: `gaussian_filter`, plus `mean`, `median`, `minimum`,
  `maximum`, `variance`, and `standard deviation` filters over an arbitrary
  footprint.
- **Stack operations** that combine an array of images: `mean`, `median`,
  `minimum`, `maximum`, `variance`, and `standard deviation`.
- First-class support for **masked data** via boolean masks, mask values,
  per-pixel weights, NaNs, or NumPy `MaskedArray` inputs.
- Operates on **arbitrarily-shaped stacks** of images at once; the last two axes
  are the spatial axes.

## Installation

```bash
pip install rms-psiops
```

## Quick example

```python
import numpy as np
import psiops

# A stack of ten 128x128 exposures with some bad pixels
stack = np.random.default_rng(0).random((10, 128, 128))
mask = np.random.default_rng(1).random((10, 128, 128)) < 0.05

# Mask-aware median combination; pixels bad in every frame stay masked
combined, combined_mask = psiops.median(stack, mask=mask)
```

## Documentation

See the User's Guide and API reference in the `docs/` directory (build them with
`scripts/read-docs.sh`). The User's Guide introduces the core concepts — image
stacks, the pixel-coordinate convention, masking, and return values — and walks
through every family of operations with examples.
