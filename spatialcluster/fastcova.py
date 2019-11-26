"""
Numba-Accelerated Cova Functions

(c) Ryan Martin 2018 under GPLv3 license
"""

import numpy as np
from numpy.random import randint
from math import sqrt, exp
from numba import njit


def kriging_matrices(point, points, varmodel, Ko=None, ko=None):
    """
    Returns K, k from the system K*x = k. Convenience wrapper around `pairwise_covariance` and
    `pointwise_covariance`

    Parameters:
        points (ndarray): 2-3 sized single `point`
        points (ndarray): ndata x 2-3 array of points
        varmodel (VarModel): variogram model describing the spatial continuity

    """
    n = points.shape[0]
    if Ko is None:
        Ko = np.zeros((n, n))
    pairwise_covariance(points, varmodel, Ko)
    if ko is None:
        ko = np.zeros(n)
    pointwise_covariance(point, points, varmodel, ko)
    return Ko, ko


def ordkriging_matrices(point, points, varmodel, Ko=None, ko=None):
    """
    Returns K, k from the system K*x = k consdering the constraint from ordinary kriging

    Parameters:
        points (ndarray): 2-3 sized single `point`
        points (ndarray): ndata x 2-3 array of points
        varmodel (VarModel): variogram model describing the spatial continuity
    """
    n = points.shape[0]
    if Ko is None:
        Ko = np.ones((n + 1, n + 1))
    pairwise_covariance(points, varmodel, Ko[:-1, :-1])
    Ko[-1, -1] = 0.0
    if ko is None:
        ko = np.ones(n + 1)
    pointwise_covariance(point, points, varmodel, ko[:-1])
    return Ko, ko


def pairwise_covariance(points, varmodel, covmat=None):
    """
    Fast wrapper around jit functions that calculates the pairwise covariance between all `points`
    in the passed array according to the varmodel

    Parameters:
        points (ndarray): ndata x 2-3 array of points
        varmodel (VarModel): variogram model describing the spatial continuity
        covmat (ndarray): optionally pass the allocated memory to fill with covariance values

    .. codeauthor:: Ryan Martin - 27-11-2017
    """
    c0, its, ccs, aas, rotmats = get_varg_arrays(varmodel)
    npoints, dim = points.shape
    if dim == 2:
        points = np.c_[points, np.full(npoints, 0.5)]
    elif dim == 1:
        points = np.c_[points, np.full(npoints, 0.5), np.full(npoints, 0.5)]
    if covmat is not None:
        covmat[:] = 0.0
        return _pairwise_covariance(points, c0, its, ccs, aas, rotmats, covmat)
    covmat = np.zeros((npoints, npoints), dtype=np.float64)
    return _pairwise_covariance(points, c0, its, ccs, aas, rotmats, covmat)


def pointwise_covariance(point, points, varmodel, covmat=None):
    """
    Fast wrapper around jit functions that calculates the covariance between `point` and all
    `points` according to the varmodel

    Parameters:
        point (ndarray): 2-3 sized single `point`
        points (ndarray): ndata x 2-3 array of points
        varmodel (VarModel): variogram model describing the spatial continuity
        covmat (ndarray): optionally pass the allocated memory to fill with covariance values

    .. codeauthor:: Ryan Martin - 27-11-2017
    """
    c0, its, ccs, aas, rotmats = get_varg_arrays(varmodel)
    npoints, dim = points.shape
    if dim == 2:
        points = np.c_[points, np.full(npoints, 0.5)]
        point = np.append(point, [0.5], axis=0)
    elif dim == 1:
        points = np.c_[points, np.full(npoints, 0.5), np.full(npoints, 0.5)]
        point = np.append(point, [0.5, 0.5], axis=0)
    if covmat is not None:
        covmat[:] = 0.0
        return _pointwise_covariance(point, points, c0, its, ccs, aas, rotmats, covmat)
    covmat = np.zeros(npoints, dtype=np.float64)
    return _pointwise_covariance(point, points, c0, its, ccs, aas, rotmats, covmat)


def centered_covariance(mp, griddef, varmodel):
    """
    Fast wrapper around jit functions that calculates the covariance from the xyzpoint `mp` to
    all points in the passed griddef, acccording to the varmodel

    Parameters:

    .. codeauthor:: Ryan Martin - 27-11-2017
    """
    c0, its, ccs, aas, rotmats = get_varg_arrays(varmodel)
    gridcov = np.zeros(griddef.count(), dtype=np.float64)
    return _centered_covariance(mp, *griddef.gridarray(), c0, its, ccs, aas, rotmats, gridcov)


def get_varg_arrays(varmodel):
    """
    parse the varmodel into arrays

    Returns:
        c0, its[nst], ccs[nst], aas[nst], rotmats[nst, 3, 3]
    """
    if hasattr(varmodel, "it"):  # this is a pygeostat variogram model
        its = np.array(varmodel.it)
        ccs = np.array(varmodel.cc)
        aas = np.array(varmodel.ahmax)
        rotmats = np.array(varmodel.rotmat)
    else:  # this is a gglib variogram model
        its = varmodel.its
        ccs = varmodel.ccs
        aas = varmodel.ranges[:, 0]
        rotmats = varmodel.rmat
    return varmodel.c0, its, ccs, aas, rotmats


@njit(cache=True)
def _centered_covariance(mp, nx, xmn, xsiz, ny, ymn, ysiz, nz, zmn, zsiz,
                         c0, its, ccs, aas, rotmats, gridcov):
    """
    Calculate the covariance for all cells in the grid from the midpoint `mp`.
    Used for `SpectralSimulator`
    """
    icell = 0
    invahmax = 1 / aas[0]
    nst = len(its)
    cmax = c0 + ccs.sum()
    thispt = np.zeros(3)
    for i in range(nst):
        it = its[i]
        cc = ccs[i]
        rm = rotmats[i, :, :]
        for iz in range(nz):
            thispt[2] = zmn + iz * zsiz
            for iy in range(ny):
                thispt[1] = ymn + iy * ysiz
                for ix in range(nx):
                    thispt[0] = xmn + ix * xsiz
                    gridcov[icell] += cc * _cova(it, mp, thispt, rm, invahmax, cmax)
                    icell += 1
    return gridcov


@njit(cache=True)
def _pairwise_covariance(points, c0, its, ccs, aas, rotmats, C):
    """
    For a covariance matrix C of size [npoints, npoints] (allocated outside), fill the matrix with
    the pairwise covariance between all `points` according to the c0, its, ccs, aas, rotmats \
    input arrays
    .. codeauthor:: Ryan Martin - 02-03-2018
    """
    npoints, dim = points.shape
    invahmax = 1 / aas[0]
    nst = len(its)
    cmax = c0
    for c in ccs:
        cmax += c
    for i in range(npoints):
        C[i, i] = cmax
    for i in range(npoints):
        pt1 = points[i, :]
        for j in range(i + 1, npoints):
            pt2 = points[j, :]
            for ist in range(nst):
                cov = ccs[ist] * _cova(its[ist], pt1, pt2, rotmats[ist], invahmax, cmax)
                C[i, j] += cov
    for i in range(npoints):
        for j in range(i + 1, npoints):
            C[j, i] = C[i, j]
    return C


@njit(cache=True)
def _pointwise_covariance(point, points, c0, its, ccs, aas, rotmats, X):
    """
    Fill the array `X` with the covariance between `point` and `points` according to the
    variogram arrays
    .. codeauthor:: Ryan Martin - 02-03-2018
    """
    npoints, dim = points.shape
    nst = len(its)
    cmax = c0
    for c in ccs:
        cmax += c
    pt1 = point
    for ist in range(nst):
        it = its[ist]
        cc = ccs[ist]
        rm = rotmats[ist, :, :]
        invahmax = 1 / aas[ist]
        for i in range(npoints):
            pt2 = points[i, :]
            cov = cc * _cova(it, pt1, pt2, rm, invahmax, cmax)
            X[i] += cov
    return X


@njit(cache=True)
def _getcova(pt1, pt2, c0, its, ccs, aas, rotmats, cmax):
    """ the total covariance between 1 and 2 considering all structures """
    nst = its.shape[0]
    cov = 0.0
    for ist in range(nst):
        it = its[ist]
        cc = ccs[ist]
        rm = rotmats[ist, :, :]
        invahmax = 1 / aas[ist]
        cov += cc * _cova(it, pt1, pt2, rm, invahmax, cmax)
    return cov


@njit(cache=True)
def _cova(it, pt1, pt2, rm, invahmax, cmax):
    """ the covariance between 1 and 2 considering a single structure """
    if it == 1:
        cova = _sphcov(pt1, pt2, rm, invahmax, cmax)
    elif it == 2:
        cova = _expcov(pt1, pt2, rm, invahmax, cmax)
    elif it == 3:
        cova = _gauscov(pt1, pt2, rm, invahmax, cmax)
    return cova


@njit(cache=True)
def _rand_validstruct_intcode():
    """ For other functions to get a random selection of available covariance types """
    # valid covars == [1, 2, 3]
    return randint(1, 4)


@njit(cache=True)
def _sphcov(pt1, pt2, rotmat, invahmax, cmax):
    """ spherical covariance """
    hsq = _anisosqdist(pt1, pt2, rotmat)
    if hsq < 10e-10:
        c = cmax
    else:
        h = sqrt(hsq) * invahmax
        if h > 1.0:
            c = 0.0
        else:
            c = 1.0 - h * (1.5 - 0.5 * h * h)
    return c


@njit(cache=True)
def _gauscov(pt1, pt2, rotmat, invahmax, cmax):
    """ Gaussian covariance """
    hsq = _anisosqdist(pt1, pt2, rotmat)
    if hsq < 10e-10:
        c = cmax
    else:
        h = sqrt(hsq) * invahmax
        c = exp(-3.0 * h * h)
    return c


@njit(cache=True)
def _expcov(pt1, pt2, rotmat, invahmax, cmax):
    """ exponential covariance """
    hsq = _anisosqdist(pt1, pt2, rotmat)
    if hsq < 10e-10:
        c = cmax
    else:
        h = sqrt(hsq) * invahmax
        c = exp(-3.0 * h)
    return c


@njit(cache=True)
def _setrot(a1, a2, a3, r1, r2, r3):
    " basically from JM rotationmatrix.f90 "
    rmat = np.zeros((3, 3), dtype=np.float64)
    a1 = 90 - a1 if 0 <= a1 <= 270 else 450 - a1
    a1, a2, a3 = np.radians(a1), -np.radians(a2), np.radians(a3)
    sina, sinb, sint = np.sin(a1), np.sin(a2), np.sin(a3)
    cosa, cosb, cost = np.cos(a1), np.cos(a2), np.cos(a3)
    afact1 = r1 / r2
    afact2 = r1 / r3
    rmat[0, 0] = cosb * cosa
    rmat[0, 1] = cosb * sina
    rmat[0, 2] = -sinb
    rmat[1, 0] = afact1 * (-cost * sina + sint * sinb * cosa)
    rmat[1, 1] = afact1 * (cost * cosa + sint * sinb * sina)
    rmat[1, 2] = afact1 * (sint * cosb)
    rmat[2, 0] = afact2 * (sint * sina + cost * sinb * cosa)
    rmat[2, 1] = afact2 * (-sint * cosa + cost * sinb * sina)
    rmat[2, 2] = afact2 * (cost * cosb)
    return rmat


@njit(cache=True)
def _sqdistance(pt1, pt2):
    dsq = 0.0
    for i in range(pt1.shape[0]):
        dsq += (pt1[i] - pt2[i]) ** 2
    return dsq


@njit(cache=True)
def _anisosqdist(pt1, pt2, rm):
    dx = pt1[0] - pt2[0]
    dy = pt1[1] - pt2[1]
    dz = pt1[2] - pt2[2]
    sqdist = (rm[0, 0] * dx + rm[0, 1] * dy + rm[0, 2] * dz)**2 + \
        (rm[1, 0] * dx + rm[1, 1] * dy + rm[1, 2] * dz)**2 + \
        (rm[2, 0] * dx + rm[2, 1] * dy + rm[2, 2] * dz)**2
    return sqdist
