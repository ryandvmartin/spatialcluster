"""
Base Ensemble Spatial Clustering Object that parses basic multivariate and spatial inputs
and implements functions common to all ensemble clusteirng objects

(c) Ryan Martin 2018 under GPLv3 license
"""

import numpy as np

from ..cluster_utils import consensus
from ..statfuncs import columnwise_standardize


class BaseEnsemble:
    """
    Parameters
    ----------
    mvdata : ndarray
        The ndata x nvar dataset, assumed to be homotopic
    locations : ndarray
        The ndata x dim set of locations
    nreal: int
        number of clusdefs iterations to run
    nnears: int
        number of nearest neighbors to consider in the spatial search
    rseed: int
        the rseed to set the random state
    minfound : float
        the minimum proportion allowed in a single cluster
    maxfound : float
        the maximum proprtion allowed in a single cluster
    searchparams: 6-tuple
        (ang1, ang2, ang3, r1, r2, r3) anisotropic properties following the GSLIB rotation
        conventions
    """

    def __init__(self, mvdata, locations, nreal=100, nnears=10, rseed=69069, minfound=0.001,
                 maxfound=0.999, searchparams=(0, 0, 0, 500, 500, 500)):
        try:
            mvdata = mvdata.values
        except AttributeError:
            pass
        try:
            locations = locations.values
        except AttributeError:
            pass
        if mvdata.ndim == 1:
            mvdata = mvdata[:, np.newaxis]
        if locations.ndim == 1:
            locations = locations[:, np.newaxis]
        assert mvdata.ndim == 2, "mvdata must be 2 dimensional with nd x nvar"
        assert mvdata.shape[0] > mvdata.shape[1], "mvdata.shape[0] must be > mvdata.shape[1]"
        assert locations.ndim == 2, "locations must be 2 dimensional with nd x dimension"
        assert locations.shape[0] > locations.shape[1], (
            "locations.shape[0] must be > locations.shape[1]")
        assert isinstance(searchparams, (tuple, list)), 'Pass list or tuple for `searchparams`'
        if len(searchparams) != 5 and len(searchparams) != 6:
            raise ValueError('Must pass a 5-long tuple of (ang1, ang2, ang3, r1, r2, r3) for '
                             '`searchparams`')
        # ensure each variable (column) has unit variance
        self.mvdata = columnwise_standardize(mvdata)
        self.locations = locations
        if len(searchparams) == 5:
            searchparams = (searchparams[0], searchparams[1], searchparams[2],
                            1.0, searchparams[3], searchparams[4])
        self.pars = dict(nreal=nreal, nnears=nnears, rseed=rseed, searchparams=searchparams,
                         minfound=minfound, maxfound=maxfound)
        self.clusterings = None
        self.mvstat, self.spstat = None, None

    def fit(self, *args, **kwargs):
        """
        Subtype specific `fit` subroutine, generates the clustering ensemble,
        e.g. generates self.clusterings
        """
        raise NotImplementedError("`fit` is not implemented for `{}`".format(type(self)))

    def predict(self, nclus, weights=None, method='spec', refclus=None):
        """
        Predict the clusters given the ensemble contained in `self.clusterings`
        Parameters
        ----------
        weights : ndarray
            nclus-long array of weights for each clustering
        method : str
            either `spec` or `hier`
        refclus : ndarray
            A reference clustering for this dataset that the target will be recoded too
        """
        self.labels, self.clusprob, self.pairings = \
            consensus(self.clusterings, nclus, weights, method=method, refclus=refclus)
        return self.labels

    def computestats(self, func=None):
        """
        For each clusdefs in the ensemble compute the within cluster sum of squares and the
        spatial entropy metric
        """
        from .. import clustermetrics as clm
        if func is None:
            func = clm.tdiff_wcss

        self.mvstat, self.spstat = clm.cluster_metrics(
            self.mvdata, self.locations, self.clusterings,
            self.pars['nnears'], self.pars['searchparams'], func)

    def geostat_subensemble(self, nclus=100, minprop=0.001, startpt=(0.5, 0.0), endpt=(0.0, 0.5)):
        """
        Choose the best clusterings from the ensemble using the clustering metrics from
        `computestats`

        Parameters
        ----------
        nclus : int
            the number of clusters from the ensemble to select
        minprop : float
            reject clusterings that have at least 1 cluster that is < this proportion
        startpt : tuple
            the starting coordinate of the line to search
        endpt : tuple
            the end coordinate of the line to search
        """
        import copy
        from scipy.spatial import cKDTree
        if self.spstat is None or self.mvstat is None:
            self.computestats()
        ndata = len(self.mvdata)

        # set the start and end location of the line to search along
        xstart, ystart = startpt
        xfin, yfin = endpt

        # drop points from the search nodes that are overlapping
        searchpts = np.c_[self.spstat, self.mvstat]
        keepidxs = pairwise_exclude(searchpts)
        droppedpts = searchpts[keepidxs, :]
        droppedclusterings = self.clusterings[:, keepidxs]
        droppedse = self.spstat[keepidxs]
        droppedwcss = self.mvstat[keepidxs]

        # setup the kdtree:
        dx, dy = ((xfin - xstart), (yfin - ystart))
        rtree = cKDTree(droppedpts)
        # points along the line, rotated to match the search anisotropy
        linepts = np.array([[xstart + t * dx, ystart + t * dy] for t in np.arange(0, 1, 1 / nclus)])

        drawn_idx = []
        drawn_sp = np.zeros(nclus, dtype=np.float64)
        drawn_mv = np.zeros(nclus, dtype=np.float64)
        drawn_clusterings = np.zeros((ndata, nclus), dtype=np.int32)
        for idraw, pt in enumerate(linepts):
            _, idxs = rtree.query(pt, k=self.clusterings.shape[1] - 1)
            for ix in idxs:
                uclus = np.unique(droppedclusterings[:, ix])
                props = [(self.clusterings[:, ix] == c).sum() / ndata for c in uclus]
                if ix not in drawn_idx and all(p > minprop for p in props):
                    drawn_idx.append(ix)
                    drawn_sp[idraw] = droppedse[ix]
                    drawn_mv[idraw] = droppedwcss[ix]
                    drawn_clusterings[:, idraw] = droppedclusterings[:, ix]
                    break
        nfound = len(drawn_idx)
        if nfound == 0:
            raise ValueError("ERROR: no clusterings found! Check minprop ...")
        elif nfound < nclus:
            print('Warning: fewer than the target number of clusterings were found!')
        else:
            print("found {} clusterings".format(nfound))
        newensclus = copy.deepcopy(self)
        newensclus.spstat = drawn_sp
        newensclus.mvstat = drawn_mv
        newensclus.clusterings = drawn_clusterings
        return newensclus


def pairwise_exclude(searchpts, rseed=69321):
    """ returns a list of idxs that are not duplicates based on the distance between points """
    from scipy.spatial.distance import cdist
    nd, dim = searchpts.shape
    idxs = np.arange(searchpts.shape[0])
    np.random.seed(rseed)
    np.random.shuffle(idxs)
    keepidxs = [idxs[0]]
    for ix in idxs[1:]:
        distsum = (cdist(searchpts[ix - 1:ix, :], searchpts[ix:, :]) == 0).sum()
        if distsum == 0:
            keepidxs.append(ix)
    return keepidxs
