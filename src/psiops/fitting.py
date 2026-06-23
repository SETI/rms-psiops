##########################################################################################
# psiops/fitting.py
##########################################################################################
"""Class defining a spatial transformation and optional stretch of an ImageModel to best
match an image.

The spatial transformation can include a pixel offset, symmetric zoom factor, and
rotation. The image can be re-scaled based on a Stretch object.
"""

import itertools
import math

import numpy as np
from scipy.optimize import least_squares

from ._utils import _check_tuple, _merge_weights
from ._validation import _check_image


class Fitting:
    """Class defining a spatial transformation and optional stretch of an ImageModel to
    best match an image.

    The spatial transformation can include a pixel offset, symmetric zoom factor, and
    rotation. The image can be re-scaled based on a Stretch object.

    Attributes:
        imagemodel (ImageModel): The ImageModel to fit to the target image.
        stretch (Stretch): The Stretch object that provides the best fit of the model to
            the target.
        target (array): The 2-D image that this Fitting should match after the
            transformation and Stretch are applied.
        mask (array or None): The boolean mask of `target` pixels to ignore, if any.
        weights (array or None): The weight array if the `target` pixels are weighted
            nonuniformly.
        params (array): Best-fit values for the spatial transformation parameters (`x`,
            `y`, `zoom`, `rotate`), where `x` and `y` are the pixel offsets, `zoom` is a
            zoom factor, and `rotate` is a rotation angle in radians.
        x (float): Alternative name for params[0].
        y (float): Alternative name for params[1].
        zoom (float): Alternative name for params[2].
        rotate (float): Alternative name for params[3].
        transformed (array): An array representing the `imagemodel` after the
            transformation has been applied but without the `stretch`.
        guesses (array): Initial guesses for the spatial transformation parameters (`x`,
            `y`, `zoom`, `rotate`).
        flags (array-like): Four boolean flags equal to True if the corresponding
            parameter is being fitted.
        limits (array-like): The maximum allowed change in the corresponding parameter
            from its initial guess. A value of zero indicates that there is no limit on
            that parameter.
        nparams (int): The number of spatial transformation parameters.
        lsq_dict (dict): Any additional parameters that are used as inputs to
            `scipy.optimize.least_squares`.
        dof (int): The number of degrees of freedom in the fit.
        weight_sum (scalar): The total weight of the unmasked, fitted pixels of `target`.
        chi_sq (float): The chi-squared value from the fit.
        rms (float): The root-mean-squared residual from the fit.
        covar (array): The 4x4 covariance matrix for the transformation coefficients.
        dx (float): One-sigma uncertainty in the best-fit value for `x`.
        dy (float): One-sigma uncertainty in the best-fit value for `y`.
        corr (float): The correlation coefficient between `dx` and `dy`.

    Properties:
        model (array): The stretched 2-D image, obtained by applying the Stretch to the
            transformed `imagemodel`.
        residuals (array): The 2-D array of residuals, `target` minus `model`.
        median_abs_residual (float): The median of |`target` - `model`| over the unmasked
            pixels (weights ignored).
        background (array or scalar): The background 2-D array or value for this Stretch.
        scaling (array or scalar): The 2-D array or value that multiplies `image` in
            this Stretch.
        m_sigma (array or scalar): The uncertainty in `model`.
        b_sigma (array or scalar): The uncertainty in `background`.
        s_sigma (array or scalar): The uncertainty in `scaling`.
    """

    def __init__(self, model, stretch):
        """Constructor for a Fitting.

        Parameters:
            model (ImageModel): The ImageModel to fit to the target image.
            stretch (Stretch): The Stretch properties to fit to the target image.
        """

        self.imagemodel = model
        self.stretch = stretch

    def set_target(self, target, *, mask=None, maskval=None, weights=None, nans=False,
                   corner=(0, 0), shape=None):
        """Set the target image for this Fitting.

        Parameters:
            target (array): The 2-D target image, which this Stretch object should
                match.
            mask (array, optional): Boolean mask array, equal to True where the values
                in `target` are to be ignored.
            maskval (float, optional): A value that should be masked wherever it appears
                in `target`. This can be used instead of or in addition to the `mask`.
            weights (array, optional): Weight array specifying the possibly unequal
                weights associated with the pixels in `target`. A weight of zero is
                equivalent to a `mask` value of True. This can be provided in addition
                to or instead of the `mask` or `maskval`. Values should never be
                negative.
            nans (bool, optional): True to check `target` for NaNs and interpret them
                as masked values.
            corner (tuple of two ints, optional): The lower coordinates of a slice of
                `target`, to be provided if only part of `target` is to be fitted.
            shape (tuple of two ints, optional): The shape of a slice of `target`, to be
                provided if only part of `target` is to be fitted.
        """

        # Interpret the image inputs
        target, mask, weights, _ = _check_image(target, mask, maskval, weights, nans=nans,
                                                comps=False, two=True)

        target_weights = _merge_weights(mask, weights)
        self._full_target = target
        self._full_mask = mask
        self._full_weights = target_weights
        self._full_shape = target.shape

        # Interpret the slice
        corner = _check_tuple(corner, 'corner coordinates', floats=False, negs=True,
                              default=(0,0))
        max_shape = (target.shape[0] - corner[0], target.shape[1] - corner[1])
        if shape is None:
            shape = max_shape
        else:
            shape = _check_tuple(shape, 'shape', floats=False, negs=False,
                                 default=max_shape)
        self.corner = corner
        self.shape = shape

        islice = slice(corner[0], corner[0] + shape[0])
        jslice = slice(corner[1], corner[1] + shape[1])
        self.target = target[islice, jslice]

        if mask is None:
            self.mask = None
        else:
            self.mask = mask[islice, jslice]

        if weights is None:
            self.weights = None
        else:
            self.weights = weights[islice, jslice]
            self.weights = self.weights / np.max(self.weights)  # pass div-by-zero forward

        # Seed the Stretch with a correctly-shaped placeholder image so set_target() can
        # validate the target shape. The real image (the transformed model) is supplied
        # per iteration by `_func` during the fit, overwriting this placeholder.
        self.stretch.set_image(np.zeros(self.shape))
        self.stretch.set_target(self.target, mask=self.mask, maskval=maskval,
                                weights=self.weights, nans=nans)

    def remask(self, mask):
        """Overlay a new mask atop the mask originally defined via `set_target`.

        Parameters:
            mask (array): New boolean mask to overlay. Must have shape `self.shape`.

        Raises:
            ValueError: If `mask` shape does not match `self.shape`.
        """

        if not hasattr(self, 'target'):
            raise ValueError('target must be defined before remask()')

        if mask.shape != self.shape:
            raise ValueError(f'new mask shape must be {self.shape}')

        if self.mask is None:
            self.mask = mask
        else:
            self.mask = self.mask | mask

        self.stretch.set_target(self.target, mask=self.mask, weights=self.weights)

    @staticmethod
    def _func(x, fitting):
        """Function called by SciPy.optimize.least_squares(), returning the vector of
        residuals.

        Parameters:
            x (array): Current parameter values as provided by the optimizer.
            fitting (Fitting): The Fitting instance whose model and stretch are
                evaluated.

        Returns:
            The 1-D array of residuals from the current stretch fit.
        """

        # Re-scale the parameters
        params = fitting.guesses.copy()
        i = 0
        for k in range(4):
            if fitting.flags[k]:
                params[k] += fitting._funcs[k](x[i])
                i += 1

        # Apply the geometric transform
        transformed = fitting.imagemodel.transform(fitting.shape,
                                                   center=(params[0], params[1]),
                                                   expand=params[2], rotate=params[3])
        fitting.stretch.set_image(transformed)

        # Save intermediate results
        fitting._x = x
        fitting._params = params
        fitting._transformed = transformed

        # Fit the Stretch
        fitting.stretch.fit()
        return fitting.stretch.residuals_1d

    @staticmethod
    def _guess_values(guess):
        """Normalize one entry of `guesses` to a list of one or more candidate values.

        Parameters:
            guess (float or sequence of floats): A single starting value or a sequence of
                starting values for one transformation parameter.

        Returns:
            A list of the candidate value(s).
        """

        if np.isscalar(guess):
            return [guess]
        return list(guess)

    def fit(self, guesses, flags=(True, True, False, False), limits=(10., 10., 0.4, 1.),
            lsq_dict=None):
        """Fit the transformation and stretch, searching over a grid of initial guesses.

        A separate nonlinear least-squares fit is performed from every combination of the
        per-parameter initial guesses, and the result with the smallest residual is kept.
        Upon return, the attributes and properties of this Fitting hold the best-fit
        values from that combination.

        Parameters:
            guesses (sequence of four items): The four initial values of `params`
                (`x`, `y`, `zoom`, `rotate`), where `x` and `y` are the pixel offsets,
                `zoom` is a zoom factor, and `rotate` is a rotation angle in radians. Each
                item is either a single value or a sequence of values to try. Every
                combination of the per-parameter values is fitted -- including for
                parameters whose `flags` entry is False (held fixed at each value) -- and
                the combination yielding the lowest residual is retained.
            flags (sequence of four bools, optional): Four flags that are True if the
                corresponding parameter is to be fitted, False if it is to be held fixed.
            limits (sequence of four floats, optional): The amount by which each fitted
                parameter is allowed to change from its initial value. Use zero to
                indicate that the corresponding parameter can change without limit. Where
                `flags` is False, the `limits` value is ignored.
            lsq_dict (dict, optional): Any parameters provided as additional inputs to
                `scipy.optimize.least_squares`.

        Raises:
            ValueError: If a `zoom` guess is not positive.
        """

        candidates = [Fitting._guess_values(guess) for guess in guesses]
        combos = list(itertools.product(*candidates))

        best_index = 0
        best_chi_sq = np.inf
        for index, combo in enumerate(combos):
            self._fit1(combo, flags, limits, lsq_dict)
            if self.chi_sq < best_chi_sq:
                best_chi_sq = self.chi_sq
                best_index = index

        # Leave this Fitting in the state of the best (lowest-residual) combination. The
        # final combo is already current, so only re-fit when a different one won.
        if best_index != len(combos) - 1:
            self._fit1(combos[best_index], flags, limits, lsq_dict)

    def _fit1(self, guesses, flags=(True, True, False, False),
              limits=(10., 10., 0.4, 1.), lsq_dict=None):
        """Perform a single nonlinear least-squares fit from one set of initial guesses.

        Upon return, the attributes and properties of this Fitting have been updated to
        contain the best-fit values for this fit.

        Parameters:
            guesses (sequence of four floats): The four initial values of `params`, prior
                to the fit: (`x`, `y`, `zoom`, `rotate`), where `x` and `y` are the pixel
                offsets, `zoom` is a zoom factor, and `rotate` is a rotation angle in
                radians.
            flags (sequence of four bools, optional): Four flags that are True if the
                corresponding value of `guesses` is to be fitted, False if it is to be
                held fixed.
            limits (sequence of four floats, optional): The amount by which each of the
                corresponding `guesses` is allowed to change from its initial value. Use
                zero to indicate that the corresponding parameter can change without
                limit. Where `flags` is False, the `limits` value is ignored.
            lsq_dict (dict, optional): Any parameters provided as additional inputs to
                `scipy.optimize.least_squares`.

        Raises:
            ValueError: If the `zoom` guess is not positive.
        """

        # Force float dtype: an integer `guesses` array would truncate the fractional
        # parameter updates applied during the fit, freezing the search.
        self.guesses = np.array(guesses, dtype=float)
        if self.guesses[2] <= 0.:
            raise ValueError(f'zoom must be positive; got {self.guesses[2]}')
        self.flags = np.array(flags)
        self.nparams = sum(self.flags)
        self.limits = np.array([limit or 0. for limit in limits])
        self.lsq_dict = lsq_dict.copy() if isinstance(lsq_dict, dict) else {}

        self._funcs = []
        self._dfunc_dx = []
        for limit in self.limits:
            if limit:
                self._funcs.append(lambda x, _l=limit: _l * math.sin(x))
                self._dfunc_dx.append(lambda x, _l=limit: _l * math.cos(x))
            else:
                self._funcs.append(lambda x: x)
                self._dfunc_dx.append(lambda x: 1.)

        result = least_squares(Fitting._func, np.zeros(self.nparams), args=(self,),
                               **self.lsq_dict)

        self._fill_stats(result)

    def _fill_stats(self, result):
        """Fill in quality-of-fit attributes from the OptimizeResult object returned by
        least_squares().

        Parameters:
            result (object): The OptimizeResult object returned by
                `scipy.optimize.least_squares`.

        Raises:
            LinAlgError: If matrix inversion fails due to a near-singular Jacobian or too
                many free parameters.
        """

        self._result = result

        # Update the cached values from `_func` if necessary
        if not np.all(result.x == self._x):
            _ = self._func(result.x, self)

        self.params = self._params
        (self.x, self.y, self.zoom, self.rotate) = self.params
        self.transformed = self._transformed

        # Count the unmasked pixels
        if self.mask is None:
            unmasked = self.target.size
        else:
            unmasked = self.target.size - np.sum(self.mask)

        # Evaluate the unbiased RMS residual
        if self.weights is None:
            self.weight_sum = unmasked
            self.dof = unmasked - self.nparams - self.stretch.ncoeffs
            self.chi_sq = 2 * result.cost   # cost is half of chi_sq in least_squares
            self.rms = np.sqrt(self.chi_sq / (self.dof - 1))
        else:
            antimask = np.logical_not(self.mask)
            self.dof = np.sum(antimask) - self.nparams - self.stretch.ncoeffs
            w = self.weights[antimask]
            wsum = np.sum(w)
            w2sum = np.sum(w**2)
            self.chi_sq = np.sum(w * self.stretch.residuals_1d ** 2)
            self.weight_sum = wsum
            self.rms = np.sqrt(self.chi_sq / (wsum - w2sum/wsum))

        # Covariance matrix...
        # Suppose a = f(x) and b = g(y):
        #   cov(a,b) = df/dx(x0) * dg/dy(y0) * cov(x,y)

        derivs = np.zeros(4)
        fitted = np.zeros(4, dtype=bool)
        i = 0
        for k in range(4):
            if self.flags[k]:
                derivs[k] = self._dfunc_dx[k](result.x[i]) * self.rms
                # ^note the RMS scaling here
                fitted[k] = True
                i += 1

        # The covariance of the fitted parameters is the inverse of the curvature matrix
        # J^T J. Scatter this nparams x nparams result into a full 4x4 matrix at the
        # positions of the fitted parameters, leaving the rows and columns of the unfitted
        # parameters equal to zero.
        cov_fitted = np.linalg.inv(result.jac.T @ result.jac)   # forward LinAlgError
        cov_full = np.zeros((4, 4))
        index = np.where(fitted)[0]
        cov_full[np.ix_(index, index)] = cov_fitted
        self.covar = cov_full * derivs[:,np.newaxis] * derivs
        self.dx = np.sqrt(self.covar[0, 0])
        self.dy = np.sqrt(self.covar[1, 1])
        denom = self.dx * self.dy
        self.corr = self.covar[0, 1] / denom if denom else 0.

    ######################################################################################
    # Array evaluation
    ######################################################################################

    @property
    def model(self):
        """The 2-D best-fit model."""
        return self.stretch.model

    @property
    def background(self):
        """The model background array that best fits the target image."""
        return self.stretch.background

    @property
    def scaling(self):
        """The model scale factor array that multiplies the image to best fit the target.

        Note that this returned result neglects the scaling of any second- or
        higher-order exponent of the image in the Stretch.
        """
        return self.stretch.scaling

    @property
    def residuals(self):
        """The 2-D array of residuals image minus model."""
        return self.stretch.residuals

    @property
    def median_abs_residual(self):
        """The median of the absolute residuals |target - model|.

        Masked pixels are skipped. Weights are ignored, so this is the median of the raw
        absolute residuals over the unmasked pixels -- a robust, outlier-insensitive
        measure of the fit quality.
        """

        diff = np.abs(self.target - self.model)
        if self.mask is not None:
            diff = diff[np.logical_not(self.mask)]
        return float(np.median(diff))

    @property
    def m_sigma(self):
        """Statistical uncertainty in the 2-D model."""
        return self.stretch.m_sigma

    @property
    def b_sigma(self):
        """Statistical uncertainty in the 2-D background."""
        return self.stretch.b_sigma

    @property
    def s_sigma(self):
        """Statistical uncertainty in the 2-D array of scale factors."""
        return self.stretch.s_sigma

##########################################################################################
