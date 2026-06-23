##########################################################################################
# psiops/stretch.py
##########################################################################################
"""Stretch is a class that defines a modification of the pixel values in an image.

The primary use of this class is to stretch one image or ImageModel to match the content
of another image. It can also be used to scale a model point spread function to match a
point source in an image.

Let (i,j) be the indices of an image. A stretch uses a sequence of functions to modify the
image as follows::

    stretched_image[i,j] = function[0](i,j)
                         + function[1](i,j) * image[i,j]
                         + function[2](i,j) * image[i,j]**2
                         + ...

Each function is itself defined by a 2-D polynomial in (i,j)::

    function(i,j) = c0 + c1*i + c2*j + c3*i**2 + c4*i*j + c5*j**2 + ...

There can be any number of functions (although rarely more than two) and each function can
be a polynomial of arbitrary order.

Because all coefficients are linear, the `fit` function can use linear least-squares
fitting (which is very fast and efficient) to solve for the coefficients that allow a
stretched image to optimally match the values in a target image.
"""

import numpy as np
import scipy.linalg

from ._utils import _merge_weights
from ._validation import _check_image


class Stretch:
    """Stretch is a class that defines modifications to the pixel values in an image.

    Attributes:
        orders (tuple): The sequence of orders of the polynomial functions. -1 indicates
            that the function is skipped.
        ranks (tuple): The sequence of ranks of the polynomial functions. The rank is the
            number of coefficients in the polynomial: -1 -> 0; 0 -> 1; 1 -> 3; etc.
        coeffs (array): The 1-D array of coefficients for this Stretch.
        ncoeffs (int): The number of coefficients used by this Stretch.
        image (array): The 2-D image to which this Stretch applies.
        image_mask (array): The boolean mask for `image`.
        shape (tuple): The shape of the `image`.
        target (array): The 2-D image to which this Stretch has been fitted.
        target_weights (array): The array of weights that apply to `target`; None if
            `target` is uniformly weighted.
        mask (array): The boolean array equal to True where the Stretch values are masked,
            because either the image or target is masked. None if the Stretch is unmasked.
        dof (int): The number of degrees of freedom in the fit.
        weight_sum (scalar): The total weight of the fit.
        chi_sq (float): The weighted chi-squared value from the latest fit.
        rms (float): The root-mean-squared residual from the latest fit.
        covar (array): The covariance matrix for the fitted coefficients.

    Properties:
        model (array): The stretched 2-D image, obtained by applying the Stretch to the
            `image`.
        residuals (array): The 2-D array of residuals, `target` minus `model`.
        residuals_1d (array): A 1-D array containing only the unmasked pixels of
            `residuals`.
        background (array or scalar): The background 2-D array or value for this Stretch.
        scaling (array or scalar): The 2-D array or value that multiplies `image` in
            this Stretch.
        m_sigma (array or scalar): The uncertainty in `model`.
        b_sigma (array or scalar): The uncertainty in `background`.
        s_sigma (array or scalar): The uncertainty in `scaling`.
    """

    def __init__(self, orders, coeffs=None, *, image=None, mask=None, maskval=None,
                 weights=None, nans=False):
        """Constructor for a Stretch.

        Parameters:
            orders (sequence of ints): The sequence of integer orders of the polynomials
                for each 2-D function that will be applied to an image. For example, (2,0)
                describes a Stretch involving a second-order background polynomial plus a
                constant multiplying the image. Note that an order of -1 means that a
                function is zero, so (-1,0) represents a simple scale factor on the image,
                with no offset.
            coeffs (array-like, optional): The sequence of coefficients that are used for
                the polynomial functions comprising this Stretch. Note that if a
                polynomial is order `n`, then ((n+1) * (n+2))//2 coefficients are
                required. For example, if `orders` is [2,0], then seven coefficients are
                needed: six for the second-order function followed by one for the
                zero-order function. If these are not specified with the constructor, they
                can be filled in later.
            image (array, optional): The 2-D image to which this Stretch is to be applied.
                This can be specified with the constructor or else by calling `apply`.
        """

        self.orders = tuple(orders)
        self._max_order = max(self.orders)
        self._max_image_expo = len(self.orders) - 1

        self.ranks = tuple(Stretch._rank_from_order(o) for o in self.orders)
        self.ncoeffs = int(sum(self.ranks))

        self.coeffs = None
        if coeffs is not None:
            self.set_coeffs(coeffs)

        # Info about the image to which this Stretch is applied
        self.image = None
        self.image_mask = None          # could be None
        self.shape = None

        self._image_powers = []         # [image, image**2, ...]
        self._ij_powers = []            # [i, j, i**2, i*j, j**2, ...]
        self._matrix3d = None           # model = sum(self.coeffs * self._matrix3d)

        # Info about the target to which this Stretch is fitted
        self.target = None
        self.target_mask = None
        self.target_weights = None
        self._target2d = None

        if image is not None:
            self.set_image(image, mask=mask, maskval=maskval, nans=nans)

        self._reset_fit()

    def _reset_fit(self):
        """Initialize info about the latest fit."""

        self.mask = None
        self._antimask = None
        self.dof = None
        self.weight_sum = None
        self.chi_sq = None
        self.rms = None
        self.covar = None

    def set_coeffs(self, coeffs):
        """Set the coefficients of the Stretch.

        Parameters:
            coeffs (array-like): The sequence of coefficients for this Stretch. Must have
                exactly `self.ncoeffs` elements.

        Raises:
            ValueError: If the number of coefficients does not match `self.ncoeffs`.
        """

        coeffs = np.asarray(coeffs, dtype=np.float64)
        if len(coeffs) != self.ncoeffs:
            raise ValueError(f'incorrect number of parameters; {self.ncoeffs} required')

        self.coeffs = coeffs
        self._reset_fit()

    def set_image(self, image, mask=None, maskval=None, nans=False):
        """Assign the given image to this Stretch.

        The image can be masked but cannot have variable weights.

        Parameters:
            image (array): The 2-D image, to which this Stretch object should apply.
            mask (array, optional): Boolean mask array, equal to True where the values in
                `image` are to be ignored.
            maskval (float, optional): A value that should be masked wherever it appears
                in `target`. This can be used instead of or in addition to the `image`.
            nans (bool, optional): True to check `image` for NaNs and interpret them as
                masked values.
        """

        image, mask, _, _ = _check_image(image, mask, maskval=maskval, nans=nans,
                                         two=True)
        shape = image.shape

        # Fill in the index arrays if the shape has changed
        if shape != self.shape:
            half_i = 0.5 * (shape[0] - 1)
            half_j = 0.5 * (shape[1] - 1)
            i = (np.arange(shape[0]) - half_i)[:, np.newaxis] / half_i
            j = (np.arange(shape[1]) - half_j)[np.newaxis, :] / half_j
            self._ij_powers = [i, j]
            for expo in range(2, self._max_order + 1):
                for _terms in range(expo):
                    self._ij_powers.append(i * self._ij_powers[-expo])
                self._ij_powers.append(j * self._ij_powers[-expo-1])

        # Fill in the powers of the image
        self._image_powers = [image]
        for _expo in range(2, self._max_image_expo + 1):
            self._image_powers.append(image * self._image_powers[-1])

        # Combine...
        self._matrix3d = np.empty(self.ncoeffs, dtype='object')

        # Zero-order values
        rank = self.ranks[0]
        self._matrix3d[0] = 1.
        self._matrix3d[1:rank] = self._ij_powers[:rank-1]

        indx = rank
        for image_expo in range(1, self._max_image_expo + 1):
            rank = self.ranks[image_expo]
            self._matrix3d[indx] = self._image_powers[image_expo-1]
            for offset in range(1, rank):
                self._matrix3d[indx+offset] = (self._ij_powers[offset-1]
                                               * self._image_powers[image_expo-1])
            indx += rank

        # Save info about the image
        self.image = image
        self.image_mask = mask
        self.shape = shape

        self._reset_fit()

    def set_target(self, target, *, mask=None, maskval=None, weights=None, nans=False):
        """Set the target image of the fitting.

        Parameters:
            target (array): The 2-D target image, which this Stretch object should match.
            mask (array, optional): Boolean mask array, equal to True where the values in
                `target` are to be ignored.
            maskval (float, optional): A value that should be masked wherever it appears
                in `target`. This can be used instead of or in addition to the `mask`.
            weights (array, optional): Weight array specifying the possibly unequal
                weights associated with the pixels in `target`. A weight of zero is
                equivalent to a `mask` value of True. This can be provided in addition to
                or instead of the `mask` or `maskval`. Values should never be negative.
            nans (bool, optional): True to check `target` for NaNs and interpret them as
                masked values.

        Raises:
            ValueError: If `target` shape does not match the image shape.
        """

        # Interpret the image inputs
        target, mask, weights, _ = _check_image(target, mask, maskval, weights, nans=nans,
                                                comps=False, two=True)
        if self.image is None:
            raise ValueError('no image has been assigned to this Stretch')
        if target.shape != self.image.shape:
            raise ValueError(f'shape mismatch, must be {self.image.shape}')

        # Fill in the mask and weights
        target_weights = _merge_weights(mask, weights)
        self.target = target
        self.target_mask = mask
        self.target_weights = target_weights
        if target_weights is not None:
            self.target_weights = self.target_weights / np.max(target_weights)

        if self.target_weights is None:
            self._target2d = self.target
        else:
            self._target2d = self.target * self.target_weights

        self._reset_fit()

    def fit(self):
        """Fit the stretched image to the target.

        This function updates these attributes of this Stretch:

        * `coeffs` (array): The 1-D array of coefficients for this Stretch.
        * `mask` (array): The boolean array equal to True where the Stretch values are
          masked, because either the image or target is masked. None if the Stretch is
          unmasked.
        * `dof` (int): The number of degrees of freedom in the fit.
        * `weight_sum` (scalar): The total weight of the fit.
        * `chi_sq` (float): The chi-squared value from the latest fit.
        * `rms` (float): The root-mean-squared residual from the latest fit.
        * `covar` (array): The covariance matrix for the fitted coefficients.
        """

        # We seek coefficients C to minimize the RMS residuals of C @ M - T, where
        # T is the target image and M is the matrix we have already obtained, based on the
        # assigned image.
        #
        # Without weights, the answer is C = (M_transpose @ M)**-1 @ M_transpose @ T.
        #
        # The formula with weights is the same, provided we replace M by W@M and T by W@T.

        # Build a dense (ncoeffs, ni, nj) matrix from the object array of terms, each of
        # which is a scalar or a 2-D array broadcastable to the image shape.
        dense = np.empty((self.ncoeffs, *self.shape), dtype=np.float64)
        for c in range(self.ncoeffs):
            dense[c] = np.broadcast_to(self._matrix3d[c], self.shape)

        # Weight the image matrix
        if self.target_weights is not None:
            dense = dense * self.target_weights

        # Determine the new mask and select the unmasked matrix elements
        if self.image_mask is None and self.target_mask is None:
            self.mask = None
            self._antimask = np.ones(self.shape, dtype=np.bool_)
        else:
            self.mask = (self.image_mask if self.target_mask is None
                         else self.target_mask if self.image_mask is None
                         else self.image_mask | self.target_mask)
            self._antimask = np.logical_not(self.mask)

        matrix2d = dense[:, self._antimask]
        target1d = self._target2d[self._antimask]

        # Solve
        self.coeffs = scipy.linalg.lstsq(matrix2d.T, target1d)[0]

        # Evaluate the unbiased RMS residual
        if self.target_weights is None:
            self.weight_sum = np.sum(self._antimask)
            self.dof = self.weight_sum - self.ncoeffs
            self.chi_sq = float(np.sum(self.residuals_1d**2))
            self.rms = np.sqrt(self.chi_sq / (self.dof - 1))
        else:
            self.dof = np.sum(self._antimask) - self.ncoeffs
            w = self.target_weights[self._antimask]
            wsum = np.sum(w)
            w2sum = np.sum(w**2)
            self.chi_sq = float(np.sum(w * self.residuals_1d**2))
            self.weight_sum = wsum
            self.rms = np.sqrt(self.chi_sq / (wsum - w2sum/wsum))

        # Determine the covariance matrix
        self.covar = np.linalg.inv(matrix2d @ matrix2d.T) * (self.rms**2)

    ######################################################################################
    # Array evaluation
    ######################################################################################

    def _eval(self, imin, rank, powers_only=False):
        """Evaluate part of the Stretch after a fit.

        Parameters:
            imin (int): Starting index in the coefficients array.
            rank (int): Number of coefficients to use.
            powers_only (bool, optional): If True, use only the polynomial powers of (i,j)
                rather than the full matrix.

        Returns:
            The evaluated 2-D array or scalar result.

        Raises:
            ValueError: If no coefficients have been defined.
            ValueError: If no image has been assigned.
        """

        if self.coeffs is None:
            raise ValueError('no coefficients have been defined')
        if self._matrix3d is None:
            raise ValueError('this Stretch has not yet been applied to an image')

        if rank == 0:
            return 0.

        imax = imin + rank
        if powers_only:
            matrix3d = [1., *self._ij_powers[:rank-1]]
        else:
            matrix3d = list(self._matrix3d[imin:imax])

        result = 0.
        for c in range(rank):
            result = result + self.coeffs[imin+c] * matrix3d[c]
        return result

    def _sigma(self, imin, rank, powers_only=False):
        """Evaluate the uncertainty in part of the Stretch after a fit.

        Parameters:
            imin (int): Starting index in the coefficients array.
            rank (int): Number of coefficients to use.
            powers_only (bool, optional): If True, use only the polynomial powers of (i,j)
                rather than the full matrix.

        Returns:
            The 2-D array of uncertainties.

        Raises:
            ValueError: If no coefficients have been defined.
            ValueError: If no image has been assigned.
        """

        if self.coeffs is None:
            raise ValueError('no coefficients have been defined')
        if self._matrix3d is None:
            raise ValueError('this Stretch has not yet been applied to an image')
        if self.covar is None:
            raise ValueError('this Stretch has not yet been fitted')

        if rank == 0:
            return np.zeros(self.shape)

        imax = imin + rank
        if powers_only:
            matrix3d = [1., *self._ij_powers[:rank-1]]
        else:
            matrix3d = list(self._matrix3d[imin:imax])

        # Broadcast each basis term to the full image shape
        terms = [np.broadcast_to(term, self.shape) for term in matrix3d]

        var = np.zeros(self.shape)
        for i in range(rank):
            for j in range(rank):
                var = var + self.covar[imin+i, imin+j] * terms[i] * terms[j]

        return np.sqrt(var)

    @property
    def model(self):
        """The 2-D model obtained by applying this Stretch to the applied image."""
        return self._eval(0, self.ncoeffs)

    @property
    def background(self):
        """The model background array that best fits the target image."""
        return self._eval(0, self.ranks[0])

    @property
    def scaling(self):
        """The model scale factor array that multiplies the image to best fits the target.

        Note that this returned result neglects the scaling of any second- or higher-order
        exponent of the image.
        """
        return self._eval(self.ranks[0], self.ranks[1], powers_only=True)

    @property
    def residuals(self):
        """The 2-D array of residuals image minus model."""
        diff = self.target - self.model
        if self.mask is not None:
            diff = np.where(self.mask, 0., diff)
        return diff

    @property
    def residuals_1d(self):
        """The 1-D array of unmasked residuals."""
        diff = self.target - self.model
        return diff[self._antimask]

    @property
    def m_sigma(self):
        """Statistical uncertainty in the 2-D model."""
        return self._sigma(0, self.ncoeffs)

    @property
    def b_sigma(self):
        """Statistical uncertainty in the 2-D background."""
        return self._sigma(0, self.ranks[0])

    @property
    def s_sigma(self):
        """Statistical uncertainty in the 2-D scaling."""
        return self._sigma(self.ranks[0], self.ranks[1], powers_only=True)

    ######################################################################################
    # Utilities
    ######################################################################################

    @staticmethod
    def _rank_from_order(order):
        """Convert the polynomial order to number of coefficients::

                -1 -> 0; 1 -> 1; 2 -> 3; 3 -> 6; 4 -> 10; etc.

        Parameters:
            order (int): The order of the 2-D polynomial; -1 or None for no polynomial.

        Returns:
            The number of polynomial coefficients; always a triangular number.
        """

        return ((order + 1) * (order + 2)) // 2

##########################################################################################
