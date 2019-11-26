""" (c) Ryan Martin 2018 under MIT license """

import numba
import numpy as np

from .anisokdtree import build_anisotree
from .cluster_utils import cluster
from .statfuncs import (columnwise_nscore, columnwise_standardize,
                        pointwise_distance, weighted_var)

_acmetric_MAPPING = {'morans': 1, 'getis': 2}
_acmetric_MAPPING_R = {1: 'morans', 2: 'getis'}


class ACCluster:
    """
    Autocorrelation-based spatial clustering, from Scrucca 2005

    Parameters
    ----------
    mvdata: pd.DataFrame
        the multivariate dataframe
    locations: pd.DataFrame
        the Cartesian locations of the data
    nclus: int
        the number of clusters
    cluster_method: str
        can be one of 'kmeans', 'gmm', or 'hier'
    acmetric: str
        the autocorrelation acmetric to consider, options are 'morans', 'getis'
    nnears: int
        the number of nearest neighbors in a local search to consider
    searchparams: 6-tuple
        (ang1, ang2, ang3, r1, r2, r3) anisotropic properties following the GSLIB rotation
        conventions

    .. codeauthor:: Ryan Martin - 17-10-2017
    """

    def __init__(self, mvdata, locations, nclus=None, cluster_method='kmeans',
                 acmetric='getis', nnears=25, searchparams=(0, 0, 0, 1, 1, 1), n_init=100):
        # parse the input, make sure arrays are nvar x ndata dimensioned
        if hasattr(mvdata, 'values'):
            mvdata = mvdata.values
        if hasattr(locations, 'values'):
            locations = locations.values
        # set all the parmeters to this object
        self.mvdata = mvdata
        self.locations = locations
        self.nclus = nclus
        self.cluster_method = cluster_method
        self.searchparams = searchparams
        self.nnears = nnears
        self.acmetric = acmetric
        self.n_init = n_init

    def fit(self, nclus=None):
        """
        Call all the functions to get a cluster using the autocorrelation acmetrics and the data
        passed to the constructor of this class
        """
        if nclus is not None:
            self.nclus = nclus
        assert self.nclus is not None, " set a number of clusters ! "
        # get the autocorrelation data and z-score it
        self.acdata = local_autocorr(self.acmetric, self.mvdata, self.locations,
                                     self.nnears, self.searchparams)
        self.acdata = columnwise_nscore(self.acdata)
        self.labels, self.clusobj = cluster(self.nclus, self.acdata, n_init=self.n_init,
                                            method=self.cluster_method)

    def predict(self):
        """
        Return the cluster labels obtained from the `fit` method of this class
        """
        return self.labels


# --------------------------------------------------------------------------------------------------
# Autocorrelation Functions
# --------------------------------------------------------------------------------------------------
def global_autocorr(acmetric, mvdata, locs, nnears, searchparams):
    """
    Global autocorrelation calculation

    Parameters
    ----------
    acmetric: str
        one of `morans` or `getis`, respectively
    mvdata: DataFrame or ndarray
        The multivariate dataset with variables as columns and observations as rows
    locs: DataFrame or ndarray
        The locations of the multivariate samples
    nnears: int
        The number of nearest neighbors to consider
    searchparams: 6-tuple
        A 6-tuple of ang1, ang2, ang3, r1, r2, r3 following GSLIB conventions

    .. codeauthor:: Ryan Martin - 13-03-2018
    """
    if hasattr(mvdata, "values"):
        mvdata = mvdata.values
    if hasattr(locs, "values"):
        locs = locs.values
    rtree, rotlocs = build_anisotree(searchparams, locs)
    search_idxs = rtree.query(rotlocs, k=nnears)[1]
    if acmetric.lower() == 'morans':
        return _global_morans(mvdata, search_idxs, rotlocs)
    elif acmetric.lower() == 'getis':
        return _global_getis(mvdata, search_idxs, rotlocs)
    else:
        raise ValueError("Invalid `acmetric`")


@numba.njit(cache=True)
def _global_morans(mvdata, search_idxs, rotlocs):
    """ global Morans autocorrelation acmetric """
    ndata, nvar = mvdata.shape
    I = np.zeros(nvar, np.float64)
    denom = np.zeros(nvar, np.float64)
    ydata = mvdata.copy()
    for icol in range(mvdata.shape[1]):
        ydata[:, icol] -= mvdata[:, icol].mean()
        denom[icol] = np.sum(ydata[:, icol] ** 2)
    # main loop through neighbors
    for iloc in range(ndata):
        closeix = search_idxs[iloc, :]
        wts = idw_weights(rotlocs[iloc, :], rotlocs[closeix, :])
        for wt, jloc in zip(wts, closeix):
            I += wt * (ydata[iloc, :] * ydata[jloc, :])
    return I / denom


@numba.njit(cache=True)
def _global_getis(mvdata, search_idxs, rotlocs):
    """ Global getis autocorrelation acmetric """
    ndata, nvar = mvdata.shape
    G = np.zeros(nvar, np.float64)
    denom = np.zeros(nvar, np.float64)
    # main loop through neighbors
    for iloc in range(ndata):
        closeix = search_idxs[iloc, :]
        wts = idw_weights(rotlocs[iloc, :], rotlocs[closeix, :])
        for wt, jloc in zip(wts, closeix):
            v = mvdata[iloc, :] * mvdata[jloc, :]
            denom += v
            G += wt * v
    return G / denom


def weights_matrix(locs, nnears, searchparams):
    """
    Calculate the spatial weights matrix

    Parameters
    ----------
    locs : DataFrame or ndarray
        The locations to consider in the weighting
    nnears : int
        The number of nearest neighbors to consider
    searchparams : 6-tuple
        The search anisotropy following GSLIB conventions (ang1, ang2, ang3, r1, r2, r3)

    .. codeauthor:: Ryan Martin - 13-03-2018
    """
    if hasattr(locs, "values"):
        locs = locs.values
    rtree, rotlocs = build_anisotree(searchparams, locs)
    search_idxs = rtree.query(rotlocs, k=nnears)[1]
    return _weights_matrix(search_idxs, searchparams)


@numba.njit(cache=True)
def _weights_matrix(search_idxs, rotlocs):
    """ returns the row standardized spatial weights matrix """
    ndata = search_idxs.shape[0]
    W = np.zeros((ndata, ndata), dtype=np.float64)
    # main loop through neighbors
    for iloc in range(ndata):
        closeix = search_idxs[iloc, :]
        wts = idw_weights(rotlocs[iloc, :], rotlocs[closeix, :])
        for jloc, wt in zip(closeix, wts):
            W[iloc, jloc] = wt
    return W


def local_autocorr(acmetric, mvdata, locs, nnears, searchparams):
    """
    User facing autocorrelation calculation function that takes care of the local search

    Parameters
    ----------
    acmetric: str
        one of `morans` or `getis`, respectively
    mvdata: DataFrame or ndarray
        The multivariate dataset with variables as columns and observations as rows
    locs: DataFrame or ndarray
        The locations of the multivariate samples
    nnears: int
        The number of nearest neighbors to consider
    searchparams: 6-tuple
        A 6-tuple of ang1, ang2, ang3, r1, r2, r3 following GSLIB conventions

    .. codeauthor:: Ryan Martin - 13-03-2018
    """
    if hasattr(mvdata, "values"):
        mvdata = mvdata.values
    if hasattr(locs, "values"):
        locs = locs.values
    rtree, rotlocs = build_anisotree(searchparams, locs)
    search_idxs = rtree.query(rotlocs, k=nnears)[1]
    acmetric = _acmetric_MAPPING[acmetric]
    return _local_autocorr(acmetric, mvdata, search_idxs, rotlocs)


@numba.njit(cache=True)
def _local_autocorr(acmetric, mvdata, search_idxs, rotlocs):
    " Local autocorrelation acmetric "
    ndata, nvar = mvdata.shape
    acor = np.zeros((ndata, nvar), dtype=np.float64)
    # main loop through neighbors
    for iloc in range(ndata):
        # make sure the current location isnt in the search
        closeix = np.array([icl for icl in search_idxs[iloc] if icl != iloc], dtype=np.int_)
        if len(closeix) > 1:
            if acmetric == 1:
                wts = idw_weights(rotlocs[iloc, :], rotlocs[closeix, :])
                acor[iloc, :] = _local_morans(wts, mvdata[iloc, :], mvdata[closeix, :])
            elif acmetric == 2:
                wts = binary_weights(rotlocs[iloc, :], rotlocs[closeix, :])
                acor[iloc, :] = _local_getis(wts, mvdata[iloc, :], mvdata[closeix, :])
        else:
            acor[iloc, :] = mvdata[iloc, :]
    return acor


@numba.njit(cache=True)
def idw_weights(currentloc, otherlocs, regconst=2.0, p=2, mindist=1e-7):
    """
    Calculate the IDW weights between the current location and the other locations

    Parameters
    ----------
    currentloc : ndarray
        `dim` dimensioned ndarray
    otherlocs : ndarray
        `ndata x dim` dimensioned ndarray

    Returns
    -------
    wts : ndarray
        Same size as the `otherlocs` array

    .. codeauthor:: Ryan Martin - 04-07-2018
    """
    dists = pointwise_distance(currentloc, otherlocs)
    dists[dists < mindist] = mindist
    one = 1.0  # https://github.com/numba/numba/issues/3135
    wts = one / (regconst + dists ** p)
    return wts / wts.sum()


@numba.njit(cache=True)
def binary_weights(currentloc, otherlocs, maxdistance=1e10):
    """
    Return the binary weights given current and other locs

    Parameters
    ----------
    currentloc : ndarray
        `dim` dimensioned ndarray
    otherlocs : ndarray
        `ndata x dim` dimensioned ndarray

    Returns
    -------
    wts : ndarray
        Same size as the `otherlocs` array

    .. codeauthor:: Ryan Martin - 04-07-2018
    """
    dists = pointwise_distance(currentloc, otherlocs)
    return (dists <= maxdistance).astype(np.float64)


@numba.njit(cache=True)
def _local_morans(wts, z, zn):
    " local morans assuming the weights are pre-calculated outside "
    morans_denom = weighted_var(zn, wts)
    morans_denom[morans_denom < 1e-8] = 1e-8  # if morans_denom was 0, likely a spike in the data...
    nd, nvar = zn.shape
    res = np.zeros(nvar, dtype=np.float64)
    for idata in range(nd):
        for ivar in range(nvar):
            mod = -1.0 if (z[ivar] < 0 and zn[idata, ivar] < 0) else 1.0
            res[ivar] += mod * wts[idata] * z[ivar] * zn[idata, ivar]
    return res / (morans_denom * wts.sum())


@numba.njit(cache=True)
def _local_getis(wts, z, zn):
    " local getis assuming the weights are pre-calculated outside "
    getis_denom = np.maximum(1e-10, np.sum(z, 0))
    nd, nvar = zn.shape
    res = np.zeros(nvar, dtype=np.float64)
    for idata in range(nd):
        for ivar in range(nvar):
            res[ivar] += wts[idata] * zn[idata, ivar]
    return res / (getis_denom * wts.sum())
