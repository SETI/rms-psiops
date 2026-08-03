# rms-psiops

<!-- pyml disable MD025 -->

[![GitHub release; latest by date](https://img.shields.io/github/v/release/SETI/rms-psiops)](https://github.com/SETI/rms-psiops/releases)
[![GitHub Release Date](https://img.shields.io/github/release-date/SETI/rms-psiops)](https://github.com/SETI/rms-psiops/releases)
[![Test Status](https://img.shields.io/github/actions/workflow/status/SETI/rms-psiops/run-tests.yml?branch=main)](https://github.com/SETI/rms-psiops/actions)
[![Documentation Status](https://readthedocs.org/projects/rms-psiops/badge/?version=latest)](https://rms-psiops.readthedocs.io/en/latest/?badge=latest)
[![Code coverage](https://img.shields.io/codecov/c/github/SETI/rms-psiops/main?logo=codecov)](https://codecov.io/gh/SETI/rms-psiops)
<br />
[![PyPI - Version](https://img.shields.io/pypi/v/rms-psiops)](https://pypi.org/project/rms-psiops)
[![PyPI - Format](https://img.shields.io/pypi/format/rms-psiops)](https://pypi.org/project/rms-psiops)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/rms-psiops)](https://pypi.org/project/rms-psiops)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/rms-psiops)](https://pypi.org/project/rms-psiops)
<br />
[![GitHub commits since latest release](https://img.shields.io/github/commits-since/SETI/rms-psiops/latest)](https://github.com/SETI/rms-psiops/commits/main/)
[![GitHub commit activity](https://img.shields.io/github/commit-activity/m/SETI/rms-psiops)](https://github.com/SETI/rms-psiops/commits/main/)
[![GitHub last commit](https://img.shields.io/github/last-commit/SETI/rms-psiops)](https://github.com/SETI/rms-psiops/commits/main/)
<br />
[![Number of GitHub open issues](https://img.shields.io/github/issues-raw/SETI/rms-psiops)](https://github.com/SETI/rms-psiops/issues)
[![Number of GitHub closed issues](https://img.shields.io/github/issues-closed-raw/SETI/rms-psiops)](https://github.com/SETI/rms-psiops/issues)
[![Number of GitHub open pull requests](https://img.shields.io/github/issues-pr-raw/SETI/rms-psiops)](https://github.com/SETI/rms-psiops/pulls)
[![Number of GitHub closed pull requests](https://img.shields.io/github/issues-pr-closed-raw/SETI/rms-psiops)](https://github.com/SETI/rms-psiops/pulls)
<br />
![GitHub License](https://img.shields.io/github/license/SETI/rms-psiops)
[![Number of GitHub stars](https://img.shields.io/github/stars/SETI/rms-psiops)](https://github.com/SETI/rms-psiops/stargazers)
![GitHub forks](https://img.shields.io/github/forks/SETI/rms-psiops)
[![DOI](https://zenodo.org/badge/rms-psiops.svg)](https://zenodo.org/badge/latestdoi/rms-psiops)
<!-- start-after-point -->

Photometrically accurate Science Image Operations.

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

Full documentation is published at
[rms-psiops.readthedocs.io](https://rms-psiops.readthedocs.io). The
[User's Guide](https://rms-psiops.readthedocs.io/en/latest/userguide.html)
introduces the core concepts — image stacks, the pixel-coordinate convention,
masking, and return values — and walks through every family of operations with
examples; the
[API reference](https://rms-psiops.readthedocs.io/en/latest/module.html)
documents every public function.

To build the documentation locally, run `scripts/read-docs.sh`.

## Contributing

Information on contributing to this package can be found in the
[Contributing Guide](https://github.com/SETI/rms-psiops/blob/main/CONTRIBUTING.md).

## Links

- [Documentation](https://rms-psiops.readthedocs.io)
- [Repository](https://github.com/SETI/rms-psiops)
- [Issue tracker](https://github.com/SETI/rms-psiops/issues)
- [PyPi](https://pypi.org/project/rms-psiops)

## Licensing

This code is licensed under the [Apache License v2.0](https://github.com/SETI/rms-psiops/blob/main/LICENSE).
