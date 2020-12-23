# Taken from SciPy
# scipy/stats/kde.py at 79ed161bf603dc3af3986efe7064df79212c4dd4

# Weighted KDE computation based on:
# https://stackoverflow.com/a/27623920/2178047
# https://gist.github.com/tillahoffmann/f844bce2ec264c1c8cb5

# Weighted KDE computation using FFT based on:
# https://github.com/scipy/scipy/issues/6176
# https://github.com/michaelhb/superplot/blob/master/superplot/statslib/kde.py

from __future__ import division, print_function, absolute_import

# Standard library imports.
import warnings

# Scipy imports.
from scipy import linalg, special
from scipy.special import logsumexp
from scipy.stats import norm, multivariate_normal
from scipy.signal import fftconvolve
from scipy.interpolate import interp1d, RegularGridInterpolator

from numpy import atleast_2d, reshape, zeros, newaxis, dot, exp, pi, sqrt, \
     ravel, power, atleast_1d, squeeze, sum, transpose
import numpy as np


__all__ = ['gaussian_kde']


class gaussian_kde(object):
    def __init__(self, dataset, bw_method=None, weights=None, fft=True, extend=True, interp_method='nearest'):
        self.dataset = atleast_2d(dataset)
        if not self.dataset.size > 1:
            raise ValueError("`dataset` input should have multiple elements.")

        self.d, self.n = self.dataset.shape

        if weights is not None:
            self.weights = weights / np.sum(weights)
        else:
            self.weights = np.ones(self.n) / self.n

        # Compute the effective sample size
        # http://surveyanalysis.org/wiki/Design_Effects_and_Effective_Sample_Size#Kish.27s_approximate_formula_for_computing_effective_sample_size
        self.neff = 1.0 / np.sum(self.weights ** 2)
        self.fft = fft
        self.extend = extend
        self.interp_method = interp_method

        self.set_bandwidth(bw_method=bw_method)

    def evaluate(self, points):
        """Evaluate the estimated pdf on a set of points.

        Parameters
        ----------
        points : (# of dimensions, # of points)-array
            Alternatively, a (# of dimensions,) vector can be passed in and
            treated as a single point.

        Returns
        -------
        values : (# of points,)-array
            The values at each point.

        Raises
        ------
        ValueError : if the dimensionality of the input points is different than
                     the dimensionality of the KDE.

        """

        if self.fft:
            return self.evaluate_fft(points)
        else:
            return self.evaluate_not_fft(points)

    def evaluate_not_fft(self, points):
        points = atleast_2d(points)

        d, m = points.shape
        if d != self.d:
            if d == 1 and m == self.d:
                # points was passed in as a row vector
                points = reshape(points, (self.d, 1))
                m = 1
            else:
                msg = "points have dimension %s, dataset has dimension %s" % (d,
                    self.d)
                raise ValueError(msg)

        result = zeros((m,), dtype=float)

        whitening = linalg.cholesky(self.inv_cov)
        scaled_dataset = dot(whitening, self.dataset)
        scaled_points = dot(whitening, points)

        if m >= self.n:
            # there are more points than data, so loop over data
            for i in range(self.n):
                diff = scaled_dataset[:, i, newaxis] - scaled_points
                energy = sum(diff * diff, axis=0) / 2.0
                result += exp(-energy) * self.weights[i]
        else:
            # loop over points
            for i in range(m):
                diff = scaled_dataset - scaled_points[:, i, newaxis]
                energy = sum(diff * diff, axis=0) / 2.0
                result[i] = sum(exp(-energy) * self.weights, axis=0)

        result = result / self._norm_factor

        return result

    def evaluate_fft(self, points):

        if self.d == 1:

            binned_pdf, bin_centers = self._bin_dataset(points)
            mean_bin = np.mean(bin_centers)

            def gauss_kernel(x):
                """ 1D Gaussian kernel. """
                return norm.pdf(x, loc=mean_bin, scale=self.det_cov**0.5)

            gauss_bin_centers = gauss_kernel(bin_centers)

            pdf = fftconvolve(binned_pdf, gauss_bin_centers, mode='same')
            pdf = np.real(pdf)

            bin_width = bin_centers[1] - bin_centers[0]
            pdf /= pdf.sum() * bin_width

            kde = interp1d(bin_centers,
                           pdf,
                           bounds_error=False,
                           fill_value=None)

            return np.array([kde(x) for x in points])  # max?

        elif self.d == 2:

            binned_pdf, (bin_centers_x, bin_centers_y) = self._bin_dataset(points)
            mean_bin = [np.mean(bin_centers_x), np.mean(bin_centers_y)]

            def gauss_kernel(x):
                """ 2D Gaussian kernel. """
                return multivariate_normal.pdf(x, mean=mean_bin, cov=self.covariance)

            grid_x, grid_y = np.meshgrid(bin_centers_x, bin_centers_y)
            grid = np.column_stack([grid_x.flatten(), grid_y.flatten()])

            gauss_bin_centers = gauss_kernel(grid)
            gauss_bin_centers = np.reshape(gauss_bin_centers, binned_pdf.shape, order='F')

            pdf = fftconvolve(binned_pdf, gauss_bin_centers, mode='same')
            pdf = np.real(pdf)

            bin_width_x = bin_centers_x[1] - bin_centers_x[0]
            bin_width_y = bin_centers_y[1] - bin_centers_y[0]
            bin_vol = bin_width_x * bin_width_y
            pdf /= pdf.sum() * bin_vol

            kde = RegularGridInterpolator((bin_centers_x, bin_centers_y),
                                          pdf,
                                          method=self.interp_method,
                                          bounds_error=False,
                                          fill_value=None)

            return kde(points.T)  # max?

        else:
            raise ValueError("FFT only implemented in 1 or 2 dimesions")

    __call__ = evaluate

    def integrate_gaussian(self, mean, cov):
        raise NotImplementedError

    def integrate_box_1d(self, low, high):
        raise NotImplementedError

    def integrate_box(self, low_bounds, high_bounds, maxpts=None):
        raise NotImplementedError

    def integrate_kde(self, other):
        raise NotImplementedError

    def resample(self, size=None):
        raise NotImplementedError

    def scotts_factor(self):
        return power(self.neff, -1./(self.d+4))

    def silverman_factor(self):
        return power(self.neff*(self.d+2.0)/4.0, -1./(self.d+4))

    #  Default method to calculate bandwidth, can be overwritten by subclass
    covariance_factor = scotts_factor
    covariance_factor.__doc__ = """Computes the coefficient (`kde.factor`) that
        multiplies the data covariance matrix to obtain the kernel covariance
        matrix. The default is `scotts_factor`.  A subclass can overwrite this
        method to provide a different method, or set it through a call to
        `kde.set_bandwidth`."""

    def set_bandwidth(self, bw_method=None):
        """Compute the estimator bandwidth with given method.

        The new bandwidth calculated after a call to `set_bandwidth` is used
        for subsequent evaluations of the estimated density.

        Parameters
        ----------
        bw_method : str, scalar or callable, optional
            The method used to calculate the estimator bandwidth.  This can be
            'scott', 'silverman', a scalar constant or a callable.  If a
            scalar, this will be used directly as `kde.factor`.  If a callable,
            it should take a `gaussian_kde` instance as only parameter and
            return a scalar.  If None (default), nothing happens; the current
            `kde.covariance_factor` method is kept.

        Notes
        -----
        .. versionadded:: 0.11

        Examples
        --------
        >>> import scipy.stats as stats
        >>> x1 = np.array([-7, -5, 1, 4, 5.])
        >>> kde = stats.gaussian_kde(x1)
        >>> xs = np.linspace(-10, 10, num=50)
        >>> y1 = kde(xs)
        >>> kde.set_bandwidth(bw_method='silverman')
        >>> y2 = kde(xs)
        >>> kde.set_bandwidth(bw_method=kde.factor / 3.)
        >>> y3 = kde(xs)

        >>> import matplotlib.pyplot as plt
        >>> fig, ax = plt.subplots()
        >>> ax.plot(x1, np.ones(x1.shape) / (4. * x1.size), 'bo',
        ...         label='Data points (rescaled)')
        >>> ax.plot(xs, y1, label='Scott (default)')
        >>> ax.plot(xs, y2, label='Silverman')
        >>> ax.plot(xs, y3, label='Const (1/3 * Silverman)')
        >>> ax.legend()
        >>> plt.show()

        """
        if bw_method is None:
            pass
        elif bw_method == 'scott':
            self.covariance_factor = self.scotts_factor
        elif bw_method == 'silverman':
            self.covariance_factor = self.silverman_factor
        elif np.isscalar(bw_method) and not isinstance(bw_method, str):
            self._bw_method = 'use constant'
            self.covariance_factor = lambda: bw_method
        elif callable(bw_method):
            self._bw_method = bw_method
            self.covariance_factor = lambda: self._bw_method(self)
        else:
            msg = "`bw_method` should be 'scott', 'silverman', a scalar " \
                  "or a callable."
            raise ValueError(msg)

        self._compute_covariance()

    def _compute_covariance(self):
        """Computes the covariance matrix for each Gaussian kernel using
        covariance_factor().
        """
        self.factor = self.covariance_factor()
        # Cache covariance and inverse covariance of the data
        if not hasattr(self, '_data_inv_cov'):
            # Compute the mean and residuals
            _mean = sum(self.weights * self.dataset, axis=1)
            _residual = (self.dataset - _mean[:, None])
            # Compute the biased covariance
            self._data_covariance = np.atleast_2d(np.dot(_residual * self.weights, _residual.T))
            # Correct for bias (http://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_covariance)
            self._data_covariance /= (1 - sum(self.weights ** 2))
            self._data_inv_cov = linalg.inv(self._data_covariance)

        # Scale by bandwidth
        self.covariance = self._data_covariance * self.factor**2
        self.inv_cov = self._data_inv_cov / self.factor**2
        self._norm_factor = sqrt(linalg.det(2 * pi * self.covariance))

        # Determinant of covariance matrix
        self.det_cov = np.linalg.det(self.covariance)

    def _bin_dataset(self, points):
        """
        Histogram dataset so that it is uniformly spaced. Once it is uniformly
        spaced, one can apply a discrete fast-Fourier transform.
        """

        if self.d == 1:

            nbins = self.n
            xmin, xmax = np.min(points[0]), np.max(points[0])
            binned_pdf, bin_edges = np.histogram(self.dataset[0],
                                                 bins=nbins,
                                                 range=(xmin, xmax) if self.extend else None,
                                                 normed=True,
                                                 weights=self.weights)
            bin_centers = np.array((bin_edges[:-1] + bin_edges[1:]) * 0.5)

        elif self.d == 2:

            nbins = int(self.n**0.5)
            xmin, xmax = np.min(points[0]), np.max(points[0])
            ymin, ymax = np.min(points[1]), np.max(points[1])
            binned_pdf, bin_edges_x, bin_edges_y = np.histogram2d(self.dataset[0],
                                                                  self.dataset[1],
                                                                  bins=nbins,
                                                                  range=((xmin, xmax), (ymin, ymax)) if self.extend else None,
                                                                  normed=True,
                                                                  weights=self.weights)
            bin_centers_x = 0.5 * (bin_edges_x[:-1] + bin_edges_x[1:])
            bin_centers_y = 0.5 * (bin_edges_y[:-1] + bin_edges_y[1:])
            bin_centers = [np.array(bin_centers_x), np.array(bin_centers_y)]

        else:
            raise ValueError("Bining only implemented in 1 or 2 dimesions")

        return binned_pdf, bin_centers

    def pdf(self, x):
        """
        Evaluate the estimated pdf on a provided set of points.

        Notes
        -----
        This is an alias for `gaussian_kde.evaluate`.  See the ``evaluate``
        docstring for more details.

        """
        return self.evaluate(x)

    def logpdf(self, x):
        raise NotImplementedError
