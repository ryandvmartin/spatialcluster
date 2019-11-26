""" (c) Ryan Martin 2018 under GPLv3 license """

import numba
import numpy as np

from .anisokdtree import *
from .statfuncs import *


# --------------------------------------------------------------------------------------------------
# Spatial
# --------------------------------------------------------------------------------------------------
@numba.njit(cache=True)
def spatial_entropy(ucats, categories, search_idxs):
    """
    Sum of the local entropy over all locations

    Parameters
    ----------
    ucats : ndarray
        Unique category codes, ucats = np.unique(categories)
    categories : ndarray
        Cat values for each ndata location
    search_idxs : ndarray
        ndata x nnears array of nearest neighbor idxs given by: _, idxs = cKDTree.query(locs, knn)

    Returns
    -------
    entropy : ndarray
        The entropy values found at each location in the grid
    entropy_sum : float
        The sum of entropy found in local searches around each location in the domain

    .. codeauthor:: Ryan Martin 07-03-2018
    """
    ndata, nvar = search_idxs.shape
    ndata = categories.shape[0]
    ncats = len(ucats)
    pks = np.zeros(ncats, dtype=np.float64)
    win_entropy = np.zeros(ndata, dtype=np.float64)
    for i in range(ndata):
        pks[:] = 0
        for j in search_idxs[i]:
            for icat, cat in enumerate(ucats):
                if categories[j] == cat:
                    pks[icat] += 1.0
        pks /= nvar
        for j in range(ncats):
            if pks[j] > 0:
                win_entropy[i] += -1.0 * pks[j] * np.log(pks[j])
    return win_entropy, win_entropy.sum()


# --------------------------------------------------------------------------------------------------
# Multivariate
# --------------------------------------------------------------------------------------------------
@numba.njit(cache=True)
def tdiff_wcss(ucats, categories, mvdata):
    """
    Euclidean within cluster sum of squares

    Parameters
    ----------
    ucats : ndarray
        Unique category codes, ucats = np.unique(categories)
    categories : ndarray
        Cat values for each ndata location
    mvdata : ndarray
        ndata x nvar array of homotopic multivariate data

    Returns
    -------
    distance : float
        The separation between multivariate populations

    .. codeauthor:: Ryan Martin 07-03-2018
    """
    _, nvar = mvdata.shape
    distsum = np.zeros(len(ucats), dtype=np.float64)
    clusmean = np.zeros(nvar, dtype=np.float64)
    for icat, cat in enumerate(ucats):
        catvals = mvdata[categories == cat, :]
        clusmean[:] = 0.0
        for ivar in range(nvar):
            clusmean[ivar] = catvals[:, ivar].mean()
        distsum[icat] += pointwise_distance_sq(clusmean, catvals).sum()
    return distsum.sum()


@numba.njit(cache=True)
def tdiff_mwcss(ucats, categories, mvdata, regconst=0.001):
    """
    Mahalnobis within cluster sum of squares

    Parameters
    ----------
    ucats : ndarray
        Unique category codes, ucats = np.unique(categories)
    categories : ndarray
        Cat values for each ndata location
    mvdata : ndarray
        ndata x nvar array of homotopic multivariate data

    Returns
    -------
    distance : float
        The separation between multivariate populations

    .. codeauthor:: Ryan Martin - 07-03-2018
    """
    _, nvar = mvdata.shape
    distsum = np.zeros(len(ucats), dtype=np.float64)
    for icat, cat in enumerate(ucats):
        catvals = mvdata[categories == cat, :]
        covmat = cov(catvals, regconst)
        invcov = np.linalg.inv(covmat)
        for ivar in range(nvar):
            catvals[:, ivar] -= (catvals[:, ivar]).mean()
        for j in range(catvals.shape[0]):
            distsum[icat] += mdistance_sq(catvals[j, :], invcov)
    return distsum.sum()


@numba.njit(cache=True)
def tdiff_kde_kld(ucats, categories, mvdata, band=-1):
    """
    KDE-based-KLD sum between populations

    Parameters
    ----------
    ucats : ndarray
        Unique category codes, ucats = np.unique(categories)
    categories : ndarray
        Cat values for each ndata location
    mvdata : ndarray
        ndata x nvar array of homotopic multivariate data
    band : float
        bandwidth of the Gaussian kernels. If -1, data is standardized and a bandwidth of 0.2
        is chosen

    Returns
    -------
    distance : float
        The separation between multivariate populations

    .. codeauthor:: Ryan Martin 03-05-2018
    """
    nclus = ucats.shape[0]
    assert nclus > 2
    ndata, nvar = mvdata.shape
    # get the cov and invcov for each population in the multivariate space
    covmats = np.zeros((nclus, nvar, nvar), dtype=np.float64)
    invmats = np.zeros((nclus, nvar, nvar), dtype=np.float64)
    for i in range(nclus):
        clusdat = mvdata[categories == ucats[i], :]
        covmats[i, :, :] = cov(clusdat)
        invmats[i, :, :] = np.linalg.inv(covmats[i, :, :])
    # if no bandwidth, standardize the data and set band == 0.2
    tdata = mvdata.copy()
    if band == -1:
        for j in range(nvar):
            sd = mvdata[:, j].std()
            m = mvdata[:, j].mean()
            tdata[:, j] = (mvdata[:, j] - m) / sd
        band = 0.2
    dist = 0.0
    for i in range(nclus):
        cat_ix = categories == ucats[i]
        ndata_ix = cat_ix.sum()
        catdata_ix = tdata[cat_ix, :]
        # get the density for pop_i on pop_i
        pqi = np.zeros((2, ndata_ix), dtype=np.float64)
        for k in range(catdata_ix.shape[0]):
            pqi[0, k] = kde_at_point(catdata_ix[k, :], catdata_ix,
                                     covmats[i, :, :], invmats[i, :, :])
        for j in range(i + 1, nclus):
            cat_jx = categories == ucats[j]
            ndata_jx = cat_jx.sum()
            catdata_jx = tdata[cat_jx, :]
            pqj = np.zeros((2, ndata_jx), dtype=np.float64)
            # density of pop_j on pop_i
            for k in range(catdata_ix.shape[0]):
                pqi[1, k] = kde_at_point(catdata_ix[k, :], catdata_jx,
                                         covmats[j, :, :], invmats[j, :, :])
            # density of pop_j on pop_j
            for k in range(catdata_jx.shape[0]):
                pqj[0, k] = kde_at_point(catdata_jx[k, :], catdata_jx,
                                         covmats[j, :, :], invmats[j, :, :])
            # density of pop_i on pop_j
            for k in range(catdata_jx.shape[0]):
                pqj[1, k] = kde_at_point(catdata_jx[k, :], catdata_ix,
                                         covmats[i, :, :], invmats[i, :, :])
            # compute the final sums
            ijdist = np.sum(pqi[0, :] * np.log(pqi[0, :] / pqi[1, :]))
            jidist = np.sum(pqj[0, :] * np.log(pqj[0, :] / pqj[1, :]))
            dist += 0.5 * (ijdist + jidist)
    return dist / ndata


@numba.njit(cache=True)
def tdiff_wards(ucats, categories, mvdata):
    """
    Wards distance sum between populations

    Parameters
    ----------
    ucats : ndarray
        Unique category codes, ucats = np.unique(categories)
    categories : ndarray
        Cat values for each ndata location
    mvdata : ndarray
        ndata x nvar array of homotopic multivariate data

    Returns
    -------
    distance : float
        The separation between multivariate populations

    .. codeauthor:: Ryan Martin 05-07-2018
    """
    ndata, nvar = mvdata.shape
    nclus = len(ucats)
    # standardize the inputs in a copy to be sure
    tdata = mvdata.copy()
    for j in range(nvar):
        sd = mvdata[:, j].std()
        m = mvdata[:, j].mean()
        tdata[:, j] = (mvdata[:, j] - m) / sd
    dist = 0.0
    for i in range(nclus):
        clus_ix = categories == ucats[i]
        catdata_ix = mvdata[clus_ix, :]
        for j in range(i + 1, nclus):
            clus_jx = categories == ucats[j]
            catdata_jx = tdata[clus_jx, :]
            dist += wards_distance(catdata_ix, catdata_jx)
    return dist / ndata


@numba.njit(cache=True)
def tdiff_wards_mdist(ucats, categories, mvdata):
    """
    Mahalanobis-Wards distance sum between populations

    Parameters
    ----------
    ucats : ndarray
        Unique category codes, ucats = np.unique(categories)
    categories : ndarray
        Cat values for each ndata location
    mvdata : ndarray
        ndata x nvar array of homotopic multivariate data

    Returns
    -------
    distance : float
        The separation between multivariate populations

    .. codeauthor:: Ryan Martin 05-07-2018
    """
    ndata, nvar = mvdata.shape
    nclus = len(ucats)
    # standardize the inputs in a copy to be sure
    tdata = mvdata.copy()
    for j in range(nvar):
        sd = mvdata[:, j].std()
        m = mvdata[:, j].mean()
        tdata[:, j] = (mvdata[:, j] - m) / sd
    dist = 0.0
    for i in range(nclus):
        clus_ix = categories == ucats[i]
        catdata_ix = mvdata[clus_ix, :]
        for j in range(i + 1, nclus):
            clus_jx = categories == ucats[j]
            catdata_jx = tdata[clus_jx, :]
            dist += wards_distance_mdist(catdata_ix, catdata_jx)
    return dist / ndata


# --------------------------------------------------------------------------------------------------
# Driver Functions
# --------------------------------------------------------------------------------------------------
def cluster_metrics(mvdata, locations, clusterings, nnears, searchparams, mvfunc=tdiff_mwcss):
    """
    Returns a MV and spatial measure of compactness given the input arrays

    Parameters
    ----------
    mvdata: ndarray
        nvar x ndata dimensioned array of multivariate data
    locations: ndarray
        dim x ndata dimensioned array of locations
    clusterings: ndarray
        ndata x nreal array of clusterings, individual realizations as columns
    nnears: int
        number of neighbors to consider in a local search
    searchparams: tuple
        (ang1, ang2, ang3, ahmax, ahmin, avert) defining the search anisotropy
    radius: float
        optional radius search instead of NN
    mvfunc: function
        one of `tdiff_wcss`, `tdiff_mwcss` or `tdiff_kde_kld`, `tdiff_wards` or `tdiff_wards_mdist`.
        OR: a custom function that takes: `ucats`, `clusdefs` and `mvdata` as arguments and returns
        a single value

    Returns
    -------
    mvdata, spdata : ndarray
        Arrays with the multivariate and spatial metrics for all columns of `clusterings`

    .. codeauthor:: Ryan Martin - 15-01-2018
    """
    from scipy.spatial import cKDTree
    from .utils import log_range
    assert(mvdata.shape[1] < mvdata.shape[0]), "`mvdata` must be nd x dim dimensioned "
    assert(locations.shape[1] < locations.shape[0]
           ), "`locations` must be nd x dim dimensioned"
    assert(clusterings.shape[0] == locations.shape[0]
           ), "mismatch in `clusterings` and `locations`"
    if hasattr(mvdata, "values"):
        mvdata = mvdata.values
    if hasattr(locations, "values"):
        locations = locations.values
    if hasattr(clusterings, "values"):
        clusterings = clusterings.values

    unique_cats = np.unique(clusterings[:, 0])
    locations = locations.copy()
    mvdata = mvdata.copy()
    # deal with the legacy 5-long tuple
    if len(searchparams) == 5:
        searchparams = (searchparams[0], searchparams[1],
                        searchparams[2], 1, searchparams[3], searchparams[4])

    aniso_kdtree, rotlocs = build_anisotree(searchparams, locations)
    _, spidxs = aniso_kdtree.query(rotlocs, nnears)

    if mvfunc.__name__ == "spatial_entropy":
        mvidxs = cKDTree(mvdata).query(mvdata, k=nnears)[1]
        mvdata = mvidxs

    nreal = clusterings.shape[1]
    multivariate_diff = np.zeros(nreal, dtype=np.float64)
    spatial_diff = np.zeros(nreal, dtype=np.float64)

    clusterings_t = clusterings.T
    for ireal in log_range(nreal):
        thisreal = clusterings_t[ireal, :]
        refmv = mvfunc(unique_cats,
                       np.random.permutation(thisreal),
                       mvdata)
        refsp = spatial_entropy(unique_cats,
                                np.random.permutation(thisreal),
                                spidxs)[1]
        multivariate_diff[ireal] = mvfunc(unique_cats, thisreal, mvdata)
        multivariate_diff[ireal] /= refmv
        spatial_diff[ireal] = spatial_entropy(unique_cats, thisreal, spidxs)[1]
        spatial_diff[ireal] /= refsp

    return multivariate_diff, spatial_diff


def cluster_metrics_single(mvdata, locations, clusdefs, nnears, searchparams, mvfunc=tdiff_wcss):
    """
    Returns a MV and spatial measure of compactness given the input arrays

    Parameters
    ----------
    mvdata: np.ndarray
        nvar x ndata dimensioned array of multivariate data
    locations: np.ndarray
        dim x ndata dimensioned array of locations
    clusdefs: np.ndarray
        ndata - long array of cluster definitions
    nnears: int
        number of neighbors to consider in a local search
    searchparams: tuple
        (ang1, ang2, ang3, ahmax, ahmin, avert) defining the search anisotropy
    radius: float
        optional radius search instead of NN
    mvfunc: function
        one of `tdiff_wcss`, `tdiff_mwcss` or `tdiff_kde_kld`, `tdiff_wards` or `tdiff_wards_mdist`.
        OR: a custom function that takes: `ucats`, `clusdefs` and `mvdata` as arguments and returns
        a single value

    Returns
    -------
    mvdata, spdata : float
        multivariate and spatial metric values for `clusdefs`

    .. codeauthor:: Ryan Martin - 15-01-2018
    """
    from scipy.spatial import cKDTree
    assert(mvdata.shape[1] < mvdata.shape[0]), "`mvdata` must be nd x dim dimensioned "
    assert(locations.shape[1] < locations.shape[0]
           ), "`locations` must be nd x dim dimensioned"
    if hasattr(mvdata, "values"):
        mvdata = mvdata.values
    if hasattr(locations, "values"):
        locations = locations.values
    if hasattr(clusdefs, "values"):
        clusdefs = clusdefs.values
    # deal with the legacy 5-long tuple
    if len(searchparams) == 5:
        searchparams = (searchparams[0], searchparams[1],
                        searchparams[2], 1, searchparams[3], searchparams[4])

    aniso_kdtree, rotlocs = build_anisotree(searchparams, locations)
    _, spidxs = aniso_kdtree.query(rotlocs, nnears)

    if mvfunc.__name__ == "spatial_entropy":
        mvidxs = cKDTree(mvdata).query(mvdata, k=nnears)[1]
        mvdata = mvidxs

    # get the stats
    unique_cats = np.unique(clusdefs)
    multivariate_diff = mvfunc(unique_cats, clusdefs, mvdata)
    random_multivar = mvfunc(unique_cats,
                             np.random.permutation(clusdefs),
                             mvdata)

    spatial_diff = spatial_entropy(unique_cats, clusdefs, spidxs)[1]
    random_spatial = spatial_entropy(unique_cats,
                                     np.random.permutation(clusdefs),
                                     spidxs)[1]
    if unique_cats.size > 0:
        return multivariate_diff / random_multivar, spatial_diff / random_spatial
    return multivariate_diff / random_multivar, 0


# --------------------------------------------------------------------------------------------------
# For Plotting  Differences
# --------------------------------------------------------------------------------------------------
def label_cluster_stats(mvdata, locations, clusdefs, nnears, searchparams, ax, coords=(0.99, 0.99),
                        ha='right', va='top', fontsize=7, mvfunc=tdiff_wcss, **kwargs):
    """
    Calculate and plot the cluster statistics for a given configuration

    Parameters
    ----------
    mvdata: np.ndarray
        nvar x ndata dimensioned array of multivariate data
    locations: np.ndarray
        dim x ndata dimensioned array of locations
    clusdefs: np.ndarray
        ndata - long array of cluster definitions
    nnears: int
        number of neighbors to consider in a local search
    searchparams: tuple
        (ang1, ang2, ang3, ahmax, ahmin, avert) defining the search anisotropy
    ax: mpl.axis
        plotting axis for labeling
    coords: tuple
        xy-coordinates of the calculated statistic labels
    ha, va: str
        the horz, vert alignment of the statistic labels
    fontsize: int
        size of the label
    mvfunc: function
        one of `tdiff_wcss`, `tdiff_mwcss` or `tdiff_kde_kld`, `tdiff_wards` or `tdiff_wards_mdist`.
        OR: a custom function that takes: `ucats`, `clusdefs` and `mvdata` as arguments and returns
        a single value
    **kwargs: dict
        passed to `ax.annotate` function
    """
    if hasattr(mvdata, 'values'):
        mvdata = mvdata.values.astype(float)
    if hasattr(locations, 'values'):
        locations = locations.values.astype(float)
    if hasattr(clusdefs, 'values'):
        clusdefs = clusdefs.values.astype(float)
    wcss, spentropy = cluster_metrics_single(
        mvdata, locations, clusdefs, nnears, searchparams, mvfunc)

    annot = 'MV: %.2f\nSP: %.2f' % (wcss, spentropy)
    ax.annotate(annot, xy=coords, xycoords='axes fraction',
                fontsize=fontsize, va=va, ha=ha, **kwargs)
