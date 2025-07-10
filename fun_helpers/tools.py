import iisignature as isig
import numpy as np
from skfda.representation.grid import FDataGrid
from skfda.representation.basis import BSplineBasis,FourierBasis
from skfda.preprocessing.smoothing import BasisSmoother
from skfda.preprocessing.dim_reduction import FPCA


def add_time(X):
    """
    Append a normalized time channel to each sample path.
    
    This function augments each sample in the 3D path array `X` with a time dimension,
    where time is linearly spaced between 0 and 1. The time channel is concatenated 
    along the feature axis.

    (Adapted from https://github.com/afermanian/signature-regression)

    Parameters
    ----------
    X : ndarray of shape (n, npoints, d)
        A batch of n piecewise linear paths in R^d, each sampled at `npoints` time steps.

    Returns
    -------
    Xtime : ndarray of shape (n, npoints, d + 1)
        The same paths as in `X`, but with an additional (d+1)-th dimension representing time.
    """
    times = np.tile(np.linspace(0, 1, X.shape[1]), (X.shape[0], 1))
    Xtime = np.concatenate([X, times.reshape((times.shape[0], times.shape[1], 1))], axis=2)
    return Xtime



def add_timeMis(X, prob_save=0.9, random_state=42):
    """
    Randomly remove internal time steps from each path while preserving start and end points.
    
    This function simulates missing data scenarios in functional time series. It first adds a time 
    dimension to the input using `add_time`, and then, for each sample path, randomly drops some 
    of the intermediate time steps with probability `1 - p_save`.

    Parameters
    ----------
    X : ndarray of shape (n, npoints, d)
        A batch of n piecewise linear paths in R^d, each with `npoints` time steps.

    prob_save : float, default=0.9
        Probability of retaining each internal time point (excluding the endpoints).

    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    X_mis : list of length n
        List of arrays, each with shape (n_i, d+1), where `n_i <= npoints`.
        Each entry corresponds to a single path with dropped internal time steps,
        and includes the appended time dimension.
    """
    X = add_time(X)
    np.random.seed(random_state)
    X_num = X.shape[0]
    X_mis = []

    for i in range(X_num):
        arr = np.arange(0, X[i, :, -1].shape[0])
        mask = np.ones_like(arr, dtype=bool)

        if len(arr) > 2:
            middle_mask = np.random.binomial(1, prob_save, size=len(arr) - 2).astype(bool)
            mask[1:-1] = middle_mask  # Retain start and end, drop randomly in between

        X_temp = X[i, arr[mask], :]
        X_mis.append(X_temp)

    return X_mis



def get_sigX(X, p):
    """
    Compute the truncated signature of order `p` for a batch of paths.

    The signature transform is a feature map from a path to a collection of iterated 
    integrals, widely used in rough path theory and time series analysis. This function
    computes the truncated signature up to order `p` for each sample path.

    (Adapted from https://github.com/afermanian/signature-regression)

    Parameters
    ----------
    X : ndarray of shape (n, npoints, d)
        A batch of n piecewise linear paths in R^d, each sampled at `npoints` time steps.

    p : int
        Truncation order of the signature. If `p=0`, returns a constant vector of ones.

    Returns
    -------
    sigX : ndarray of shape (n, sd_p)
        A matrix where each row is the truncated signature of a sample path.
        The length `sd_p` is determined by `isig.siglength(d, p) + 1`, accounting
        for the leading constant term (typically the 0-th level signature).
    """
    if p == 0:
        return np.full((np.shape(X)[0], 1), 1)
    else:
        d = X.shape[2]
        sigX = np.zeros((X.shape[0], isig.siglength(d, p) + 1))
        sigX[:, 0] = 1  # 0-th level (constant term) of the signature
        for i in range(X.shape[0]):
            sigX[i, 1:] = isig.sig(X[i, :, :], p)
        return sigX
    


def get_sigX_timeMis(X_mis, p):
    """
    Compute truncated signatures for a list of irregularly sampled paths.
    
    Given a list of paths with potentially missing time steps, this function
    applies the signature transform (truncated at order `p`) to each one individually
    and stacks the resulting vectors into a single matrix.

    Parameters
    ----------
    X_mis : list of arrays
        List of n paths with shape (n_i, d+1), where each path may have a different number
        of time steps due to missing values. The last dimension must include the time component.

    p : int
        Truncation order of the signature.

    Returns
    -------
    sigX : ndarray of shape (n, sig_dim)
        Matrix of truncated signatures for each path in the list. Each row contains the 
        signature features of one path.
    """
    X_num = len(X_mis)

    for i in range(X_num):
        X_temp = X_mis[i]
        x_len = X_temp.shape[0]
        x_dim = X_temp.shape[1]
        X_new = np.ndarray((1, x_len, x_dim))
        X_new[0, :, :] = X_temp
        X_sig_new = get_sigX(X_new, p)

        if i == 0:
            sigX = X_sig_new
        else:
            sigX = np.concatenate((sigX, X_sig_new), axis=0)

    return sigX


def perform_bspline(X, n_knots=7, degree=3, T_non_uniform=False):
    """
    Perform B-spline basis expansion on each functional sample.

    This function projects each feature dimension of the sample paths in `X`
    onto a B-spline basis. It returns the basis coefficients as features.
    If `inconsist=True`, the last column in `X` is interpreted as an uneven time grid
    specific to each sample.

    Parameters
    ----------
    X : ndarray of shape (n, npoints, d) or (n, npoints, d+1)
        Input functional data. The last channel is used as time grid if `T_non_uniform=True`.
        
    n_knots : int, default=7
        Number of knots used to construct the B-spline basis.
        
    degree : int, default=3
        Degree of the B-spline basis (e.g., 3 = cubic splines).
        
    T_non_uniform : bool, default=False
        Whether the input samples have non-uniform (per-sample) time grids.

    Returns
    -------
    c : ndarray of shape (n, nb), where nb = dim * n_basis
        Array of flattened B-spline coefficients for each sample.
    """
    knots = np.linspace(0, 1, n_knots)
    n = X.shape[0]
    dim = X.shape[2] - 1 if T_non_uniform else X.shape[2]
    n_basis = n_knots + degree - 2  # Adjusted for B-spline order

    Bsb = BSplineBasis(n_basis=n_basis, knots=knots, order=degree)
    c = []

    for i in range(n):
        c_temp = []
        for j in range(dim):
            if T_non_uniform:
                fd = FDataGrid(data_matrix=X[i, :, j], grid_points=X[i, :, -1])
            else:
                fd = FDataGrid(data_matrix=X[i, :, j])
            smoother = BasisSmoother(Bsb, return_basis=True)
            fd_smooth = smoother.fit_transform(fd)
            c_temp.append(fd_smooth.coefficients)
        c_temp = np.array(c_temp).reshape(-1)
        c.append(c_temp)

    return np.array(c)



def perform_fourier(X, n_basis=8, period=1, T_non_uniform=False):
    """
    Perform Fourier basis expansion on each functional sample.

    This function projects each feature dimension of the sample paths in `X`
    onto a Fourier basis and returns the expansion coefficients as features.
    If `inconsist=True`, the last column in `X` is used as the time grid for each sample.

    Parameters
    ----------
    X : ndarray of shape (n, npoints, d) or (n, npoints, d+1)
        Input functional data. The last channel is used as time grid if `T_non_uniform=True`.
        
    n_basis : int, default=8
        Number of Fourier basis functions to use in the expansion.
        
    period : float, default=1
        Period of the underlying signal (assumed time domain is [0, period]).
        
    T_non_uniform : bool, default=False
        Whether the input samples have non-uniform (per-sample) time grids.

    Returns
    -------
    c : ndarray of shape (n, nf), where nf = dim * n_basis
        Matrix of flattened Fourier coefficients per sample.
    """
    n = X.shape[0]
    dim = X.shape[2] - 1 if T_non_uniform else X.shape[2]

    Fb = FourierBasis(domain_range=(0, 1), n_basis=n_basis, period=period)
    c = []

    for i in range(n):
        c_temp = []
        for j in range(dim):
            if T_non_uniform:
                fd = FDataGrid(data_matrix=X[i, :, j], grid_points=X[i, :, -1])
            else:
                fd = FDataGrid(data_matrix=X[i, :, j])
            smoother = BasisSmoother(Fb, return_basis=True)
            fd_smooth = smoother.fit_transform(fd)
            c_temp.append(fd_smooth.coefficients)
        c_temp = np.array(c_temp).reshape(-1)
        c.append(c_temp)

    return np.array(c)



def perform_FPCA(X, time=None, n_FPC=5):
    """
    Perform Functional Principal Component Analysis (FPCA) on each dimension of functional data.

    This function applies FPCA independently to each feature dimension in the input 3D array `X`,
    using the same (or provided) time grid. The resulting principal component scores across all
    dimensions are concatenated into a single feature vector per sample.

    Parameters
    ----------
    X : ndarray of shape (n_samples, npoints, d)
        Functional data array with `n_samples` observations, each represented by `d` functional
        features sampled at `npoints` time steps.
        
    time : array-like of shape (npoints,), optional
        Time grid corresponding to the observations. If None, a uniform grid on [0,1] is used.
        
    n_FPC : int, default=5
        Number of FPCA components (functional principal components) to retain per dimension.

    Returns
    -------
    scores : ndarray of shape (n_samples, d * n_FPC)
        Concatenated FPCA scores across all feature dimensions.
    """
    if time is None:
        time = np.linspace(0, 1, X.shape[1])

    for i in range(X.shape[2]):
        fd = FDataGrid(data_matrix=X[:, :, i], grid_points=time)
        fpca = FPCA(n_components=n_FPC)
        fpca.fit(fd)
        scores_temp = fpca.transform(fd)

        if i == 0:
            scores = scores_temp
        else:
            scores = np.concatenate([scores, scores_temp], axis=1)

    return scores



def perform_uniform_embedding(X, time_num=100, Tincon=False):
    """
    Resample each functional path onto a uniform time grid.

    For functional data with either consistent or inconsistent (per-sample) time grids,
    this function interpolates all sample paths onto a common uniform grid of `time_num` points.
    It is useful for standardizing input before feature extraction (e.g., FPCA, Bspline, etc.).

    Parameters
    ----------
    X : ndarray or list
        - If `Tincon=False`: ndarray of shape (n_samples, npoints, d+1), where the last channel is time.
        - If `Tincon=True`: list of `n_samples` arrays with shape (npoints_i, d+1), where d is number of features
          and the last column is the time grid for each sample.
          
    time_num : int, default=100
        Number of points in the uniform time grid for interpolation.
        
    Tincon : bool, default=False
        Whether the input samples have inconsistent (per-sample) time grids.

    Returns
    -------
    X_new : ndarray of shape (n_samples, time_num, d)
        Resampled functional data on a uniform grid for each feature, excluding the time channel.
    """
    t_uniform = np.linspace(0, 1, time_num)

    if Tincon:
        n = len(X)
        d = X[0].shape[1]
        X_new = np.zeros([n, time_num, d - 1])
        for i in range(n):
            Xtemp = X[i]
            for j in range(d - 1):
                fd_irr = FDataGrid(data_matrix=Xtemp[:, j], grid_points=Xtemp[:, d - 1])
                X_new[i, :, j] = fd_irr(t_uniform).flatten()
    else:
        n, T, d = X.shape
        X_new = np.zeros([n, time_num, d - 1])
        for i in range(n):
            for j in range(d - 1):
                fd_irr = FDataGrid(data_matrix=X[i, :, j], grid_points=X[i, :, d - 1])
                X_new[i, :, j] = fd_irr(t_uniform).flatten()

    return X_new



def data_normal(X, gap=1e-6):
    """
    Standardize data by removing mean and scaling to unit variance per feature.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.

    gap : float, default=1e-6
        Small value added to standard deviation to avoid division by zero.

    Returns
    -------
    X_norm : ndarray of same shape as X
        Normalized data.
    """
    c_means = np.mean(X, axis=0)
    c_stds = np.std(X, axis=0) + gap
    c_stds[c_stds == 0] = 1
    return (X - c_means) / c_stds