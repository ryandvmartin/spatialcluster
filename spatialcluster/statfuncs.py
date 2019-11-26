"""
Utility functions, jitted where appropriate, to facilitate spatial clustering methodologies
developed in this package

(c) Ryan Martin 2018 under GPLv3 license
"""

import numba
import numpy as np

TWOPI = 2.0 * np.pi


# --------------------------------------------------------------------------------------------------
# Distance functions
# --------------------------------------------------------------------------------------------------
@numba.njit(cache=True)
def pointwise_distance(pt, pts):
    " get an array of distances between pt and pts "
    dists = np.zeros(pts.shape[0], dtype=np.float64)
    for ipt in range(pts.shape[0]):
        dists[ipt] = distance(pt, pts[ipt, :])
    return dists


@numba.njit(cache=True)
def pointwise_distance_sq(pt, pts):
    " get an array of sqaured distances between pt and pts "
    dists = np.zeros(pts.shape[0], dtype=np.float64)
    for ipt in range(pts.shape[0]):
        dists[ipt] = distance_sq(pt, pts[ipt, :])
    return dists


@numba.njit(cache=True)
def mdistance_sq(x, cinv):
    " Mahalanobis distance, assumes that the mean is removed from x .. "
    dis = 0.0
    for i in range(cinv.shape[0]):
        for j in range(cinv.shape[0]):
            dis += x[i] * cinv[i, j] * x[i]
    return dis


@numba.njit(cache=True)
def distance(pt1, pt2):
    " base nd euclidean distance calculation "
    return np.sqrt(distance_sq(pt1, pt2))


@numba.njit(cache=True)
def distance_sq_itemwise(ix, idxs, coords, dists):
    for i, jx in enumerate(idxs):
        dists[i] = distance_sq(coords[ix], coords[jx])


@numba.njit(cache=True)
def distance_sq(pt1, pt2):
    " base nd sqr-euclidean distance calcualtion "
    sqdist = 0.0
    for i in range(pt1.shape[0]):
        sqdist += (pt1[i] - pt2[i]) ** 2
    return sqdist


@numba.njit(cache=True)
def andistance(x1, y1, z1, x2, y2, z2, rm):
    """ the anisotropic distance """
    return np.sqrt(andistance_sq(x1, y1, z1, x2, y2, z2, rm))


@numba.njit(cache=True)
def andistance_sq(x1, y1, z1, x2, y2, z2, rm):
    """ the anisotropic squared distance """
    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    sqdist = (rm[0, 0] * dx + rm[0, 1] * dy + rm[0, 2] * dz)**2 + \
        (rm[1, 0] * dx + rm[1, 1] * dy + rm[1, 2] * dz)**2 + \
        (rm[2, 0] * dx + rm[2, 1] * dy + rm[2, 2] * dz)**2
    return sqdist


@numba.njit(cache=True)
def setrot(a1, a2, a3, r1, r2, r3):
    " basically from JM rotationmatrix.f90 "
    rotmat = np.zeros((3, 3), dtype=np.float64)
    a1 = 90 - a1 if 0 <= a1 <= 270 else 450 - a1
    a1, a2, a3 = np.radians(a1), -np.radians(a2), np.radians(a3)
    sina, sinb, sint = np.sin(a1), np.sin(a2), np.sin(a3)
    cosa, cosb, cost = np.cos(a1), np.cos(a2), np.cos(a3)
    afact1 = r1 / r2
    afact2 = r1 / r3
    rotmat[0, 0] = cosb * cosa
    rotmat[0, 1] = cosb * sina
    rotmat[0, 2] = -sinb
    rotmat[1, 0] = afact1 * (-cost * sina + sint * sinb * cosa)
    rotmat[1, 1] = afact1 * (cost * cosa + sint * sinb * sina)
    rotmat[1, 2] = afact1 * (sint * cosb)
    rotmat[2, 0] = afact2 * (sint * sina + cost * sinb * cosa)
    rotmat[2, 1] = afact2 * (-sint * cosa + cost * sinb * sina)
    rotmat[2, 2] = afact2 * (cost * cosb)
    return rotmat


# --------------------------------------------------------------------------------------------------
# Population distances
# --------------------------------------------------------------------------------------------------
@numba.njit(cache=True)
def wards_distance(x, y):
    """
    Parameters
    ----------
    x, y : ndarray
        (ndata x dx) and (ndata x dy) arrays of multivariate data to compare

    Returns
    -------
    dist : float
        Wards distance between x and y

    .. codeauthor:: Ryan Martin - 04-07-2018
    """
    ndx, dx = x.shape
    ndy, dy = y.shape
    assert dx == dy, "ERROR: can only compare vectors with similar nvar"
    xm = np.zeros(dx)
    ym = np.zeros(dy)
    xym = np.zeros(dx)
    # get the means and the mean between the two sets of clusters
    for i in range(dx):
        xm[i] = x[:, i].mean()
        ym[i] = y[:, i].mean()
        xym[i] = 0.5 * (xm[i] + ym[i])
    # init the SSE
    a = 0.0
    b = 0.0
    c = 0.0
    # calculate the SSE
    for i in range(ndx):
        a += distance_sq(x[i, :], xym)
        b += distance_sq(x[i, :], xm)
    for i in range(ndy):
        a += distance_sq(y[i, :], xym)
        c += distance_sq(y[i, :], ym)
    return (a - (b + c)) / max(a, 1e-6)


@numba.njit(cache=True)
def wards_distance_mdist(x, y, regconst=0.001):
    """
    Parameters
    ----------
    x, y : ndarray
        (ndata x dx) and (ndata x dy) arrays of multivariate data to compare
    regconst : float
        applied to the diagonal of the covariance matrix to improve stability

    Returns
    -------
    dist : float
        Wards mahalanobis distance between x and y

    .. codeauthor:: Ryan Martin - 04-07-2018
    """
    ndx, dx = x.shape
    ndy, dy = y.shape
    assert dx == dy, "ERROR: can only compare vectors with similar nvar"
    xm = np.zeros(dx)
    ym = np.zeros(dy)
    xym = np.zeros(dx)
    # get the means and the mean between the two sets of clusters
    for i in range(dx):
        xm[i] = x[:, i].mean()
        ym[i] = y[:, i].mean()
        xym[i] = 0.5 * (xm[i] + ym[i])
    # invert the covariance with the regconst on the diagonal
    invx = np.linalg.inv(cov(x, regconst))
    invy = np.linalg.inv(cov(y, regconst))
    # init the SSE
    a = 0.0
    b = 0.0
    c = 0.0
    for i in range(ndx):
        a += mdistance_sq(x[i, :] - xym, invx)
        b += mdistance_sq(x[i, :] - xm, invx)
    for i in range(ndy):
        a += mdistance_sq(y[i, :] - xym, invy)
        c += mdistance_sq(y[i, :] - ym, invy)
    return (a - (b + c)) / a


# --------------------------------------------------------------------------------------------------
# Statistics functions
# --------------------------------------------------------------------------------------------------
@numba.njit(cache=True)
def mean(dat):
    """ get the mean of each variable (as columns) """
    tdat = dat.copy() if dat.ndim > 1 else dat[:, np.newaxis]
    nvar, ndata = tdat.shape
    mvec = np.zeros(nvar)
    for i in range(nvar):
        mvec[i] = tdat[i, :].sum() / ndata
    return mvec


@numba.njit(cache=True)
def weighted_mean(dat, wts):
    """ get the weighted mean for columns in dat given the 1D vector of weights in wts """
    if dat.ndim == 1:
        dat = np.atleast_2d(dat)
    nd, nvar = dat.shape
    means = np.zeros(nvar, dtype=np.float64)
    for i in range(nd):
        means += wts[i] * dat[i, :]
    return means / wts.sum()


@numba.njit(cache=True)
def std(dat):
    """ get the weighted std for columns in dat given the 1D vector of weights in wts """
    if dat.ndim == 1:
        dat = np.atleast_2d(dat)
    return np.sqrt(var(dat))


@numba.njit(cache=True)
def weighted_std(dat, wts):
    """ get the weighted std for columns in dat given the 1D vector of weights in wts """
    if dat.ndim == 1:
        dat = np.atleast_2d(dat)
    return np.sqrt(weighted_var(dat, wts))


@numba.njit(cache=True)
def var(dat):
    """ get the mean of each variable (as columns) """
    if dat.ndim == 1:
        dat = np.atleast_2d(dat)
    mvec = mean(dat)
    tdat = dat.copy().T
    nvar, ndata = tdat.shape
    varvec = np.zeros(nvar)
    for i in range(nvar):
        for j in range(ndata):
            varvec[i] += (tdat[i, j] - mvec[i]) ** 2
    return varvec / ndata


@numba.njit(cache=True)
def weighted_var(dat, wts):
    """ get the weighted std for columns in dat given the 1D vector of weights in wts """
    if dat.ndim == 1:
        dat = np.atleast_2d(dat).T
    nd, nvar = dat.shape
    mean = weighted_mean(dat, wts)
    var = np.zeros(nvar, dtype=np.float64)
    for i in range(nd):
        for j in range(nvar):
            var[j] += wts[i] * (dat[i, j] - mean[j]) ** 2
    return var / wts.sum()


@numba.njit(cache=True)
def cov(dat, regconst=0.001):
    " return the nvar x nvar cov matrix "
    ndata, nvar = dat.shape
    tdat = dat.copy().T
    for i in range(nvar):
        tdat[i, :] -= tdat[i, :].mean()
    cmat = np.zeros((nvar, nvar), dtype=np.float64)
    for i in range(nvar):
        for j in range(nvar):
            csum = 0.0
            for k in range(ndata):
                csum += tdat[i, k] * tdat[j, k]
            cmat[i, j] = csum
    cmat = cmat / ndata
    for i in range(nvar):
        cmat[i, i] += regconst
    return cmat


@numba.njit(cache=True)
def weighted_cov(dat, wts, regconst=0.001):
    " return the nvar x nvar cov matrix "
    ndata, nvar = dat.shape
    wtsum = wts.sum()
    tdat = dat.copy().T
    for i in range(nvar):
        tdat[i, :] -= np.sum(wts * tdat[i, :]) / wtsum
    cmat = np.zeros((nvar, nvar), dtype=np.float64)
    for i in range(nvar):
        for j in range(nvar):
            csum = 0.0
            for k in range(ndata):
                csum += wts[k] * tdat[i, k] * tdat[j, k]
            cmat[i, j] = csum
    cmat = cmat / wtsum
    for i in range(nvar):
        cmat[i, i] += regconst
    return cmat


def columnwise_standardize(columndata):
    """ Column-wise independent forward standardization """
    if hasattr(columndata, 'values'):
        columndata = columndata.values
    columndata_t = columndata.T
    for ivar in range(columndata_t.shape[0]):
        columndata_t[ivar, :] = (columndata_t[ivar, :] - columndata_t[ivar, :].mean()) / \
            np.maximum(1e-10, columndata_t[ivar, :].std())
    return columndata_t.T


def columnwise_nscore(columndata):
    """ Column-wise independent forward Gaussian transformation """
    from scipy.stats import norm
    ndata, nvar = columndata.shape
    gaussdata = columndata.copy()
    for icol in range(nvar):
        sortidx = columndata[:, icol].argsort()
        cdfinc = 1 / ndata
        gaussrange = norm.ppf(np.linspace(0.5 * cdfinc, 1 - 0.5 * cdfinc, ndata))
        gaussdata[sortidx, icol] = gaussrange
    return gaussdata


@numba.njit(cache=True)
def gausskde(pts, band=-1, tinyval=1e-10):
    """ compute the Gaussian density at `pts`, an (ndata, nvar) dimensioned matrix"""
    ndata, nvar = pts.shape
    # optionally scale the points
    tpts = pts.copy()
    if band == -1:
        for j in range(nvar):
            sd = pts[:, j].std()
            m = pts[:, j].mean()
            tpts[:, j] = (pts[:, j] - m) / sd
        band = 0.2
    # calculate the covariance and the norm
    _cov = cov(tpts)
    invcov = np.linalg.inv(_cov)
    norm = np.sqrt((_cov ** 2).sum()) * TWOPI ** (0.5 * nvar)
    # precalculate the inverse cov and the norm:
    if band > 0:
        invcov = invcov / (2.0 * band)
    # cycle through the locations and compute the density
    dens = np.zeros(ndata, np.float64)
    for ipt in range(ndata):
        dens[ipt] = kde_at_point(tpts[ipt, :], tpts, invcov, norm, tinyval)
    return dens


@numba.njit(cache=True)
def kde_at_point(pt, pts, invcov, norm, tinyval=1e-10):
    """
    Compute the Gaussian density at `pt` given `pts` and the `cov` and `invcov` of pts
    """
    dens = 0.0
    ndata, nvar = pts.shape
    dist = np.zeros(nvar, dtype=np.float64)
    for i in range(ndata):
        for j in range(nvar):
            dist[j] = pts[i, j] - pt[j]
        dens += np.exp(-mdistance_sq(dist, invcov))
    return dens / np.maximum(tinyval, norm)
