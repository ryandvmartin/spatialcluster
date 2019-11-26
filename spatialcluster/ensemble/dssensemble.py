"""
Dual Space Search Ensemble Clustering

(c) Ryan Martin 2018 under GPLv3 license
"""

import numpy as np
from numba import njit

from .. import statfuncs
from ..anisokdtree import build_anisotree
from ..utils import log_progress, rseed_list
from .baseensemble import BaseEnsemble


class DSSEnsemble(BaseEnsemble):
    """
    Clustering by the Dual Space Search (DSS) agglomeration
    {}

    DSS Parameters
    --------------
    numtake : int
        number of nearest neighbors to merge in the spatial search, numtake <= nnears

    .. codeauthor:: Ryan Martin - 04-07-2018
    """
    __doc__ = __doc__.format(BaseEnsemble.__doc__)

    def __init__(self, mvdata, locations, nreal=100, nnears=10, rseed=69069, minfound=0.001,
                 maxfound=0.999, searchparams=(0, 0, 0, 500, 500, 500), numtake=5):
        super().__init__(mvdata, locations, nreal, nnears, rseed, minfound, maxfound, searchparams)
        # add the extra parameters for this style of ensemble clustering
        self.pars['minfound'] = minfound
        self.pars['maxfound'] = maxfound
        self.pars['numtake'] = numtake

    def fit(self, target_nclus, verbose=True, nprocesses=None):
        """
        Parameters
        ----------
        target_nclus : int
            the target number of clusters to merge to, could be larger than the target number of
            clusters for the domain so that this parameter can be inferred from the pairings matrix
        verbose : bool
            whether or not to write a progress bar to the jupyter notebook
        nprocesses : int
            specifies how many parallel processes to use

        Returns
        -------
        clusterings : ndarray

        """
        import multiprocessing as mp
        assert nprocesses is None or nprocesses >= 1, 'invalid nprocesses'
        mvdata = self.mvdata
        locations = self.locations
        ndata, _ = mvdata.shape
        seeds = rseed_list(self.pars['nreal'], self.pars['rseed'])
        self.clusterings = np.zeros((ndata, self.pars['nreal']))
        # setup the parallel args list
        if nprocesses is not None:
            if verbose:
                pbar = log_progress(seeds, name='Clusterings')

                def updatefunc(*args, **kwargs):
                    pbar.update()

            else:
                updatefunc = None
            pool = mp.Pool(processes=nprocesses)
            results = {}
            for iproc, seed in enumerate(seeds):
                args = (mvdata, locations, self.pars['nnears'], self.pars['numtake'], target_nclus,
                        self.pars['searchparams'], int(seed), self.pars['minfound'],
                        self.pars['maxfound'])
                results[iproc] = pool.apply_async(dss_single, args, callback=updatefunc)
            pool.close()
            pool.join()
            pbar.close()
            for iproc in range(self.pars['nreal']):
                self.clusterings[:, iproc] = results[iproc].get()
        else:
            if verbose:
                iterable = log_progress(seeds, name="Clusterings")
            else:
                iterable = seeds
            for iproc, seed in enumerate(iterable):
                args = (mvdata, locations, self.pars['nnears'], self.pars['numtake'], target_nclus,
                        self.pars['searchparams'], int(seed), self.pars['minfound'],
                        self.pars['maxfound'])
                self.clusterings[:, iproc] = dss_single(*args)


# ---------------------------------
# supporting functions and classes
# ---------------------------------
# flag to print extra information to the terminal for debugging
_DEBUG = False


class OneCluster:
    """
    Structure for a single cluster with locations, mvdata and original data indexes

    Parameters
    ----------
    xyz, mvdata : ndarray
        ndata x (dim, nvar) arrays with data for each cluster
    idxs : ndarray
        the 1D original index values for the samples contained within this cluster

    """

    def __init__(self, xyz, mvdat, idxs):
        self.xyz = xyz
        self.center = self.xyz.mean(axis=0)
        self.mvdat = mvdat
        self.mvcenter = self.mvdat.mean(axis=0)
        self.idxs = idxs

    def mergewith(self, otherclus):
        """
        Merge `self` with `otherclus` returning a new `OneCluster` object
        """
        return OneCluster(np.r_[self.xyz, otherclus.xyz],
                          np.r_[self.mvdat, otherclus.mvdat],
                          np.append(self.idxs, otherclus.idxs))


class ClusterSet(dict):
    """
    Class to hold all clusterings as a dictionary of OneCluster objects

    Parameters
    ----------
    xyz, mvdata : ndarray
        ndata x (dim, nvar) arrays with data for each cluster
    idxs : ndarray
        the 1D original index values for the samples contained within this cluster
    clustering : ndarray
        1D array of cluster definitions

    """

    def __init__(self, allxyz, allmvdat, clustering):
        datrange = np.arange(allxyz.shape[0])
        unique_clusters = np.unique(clustering).astype(int)
        for iclus in unique_clusters:
            clusidx = clustering == iclus
            self[iclus] = OneCluster(allxyz[clusidx, :],    # locations
                                     allmvdat[clusidx, :],  # mvdata
                                     datrange[clusidx])     # original data indexes
        # properties of the passed dataset
        self._ndata, self._dim = allxyz.shape
        _, self._nvar = allmvdat.shape
        self._nclus = len(unique_clusters)

    def merge_clusters(self, clusix, clusjx, newid):
        if _DEBUG:
            print('Merging {} and {} to {}'.format(clusix, clusjx, newid))
        try:
            # remove the two clusters from `self`
            clus1 = self.pop(clusix)
            clus2 = self.pop(clusjx)
            # generate a new merged cluster with the new id
            self[newid] = clus1.mergewith(clus2)
        except Exception:
            print("Failed merging {} with {} to {}".format(clusix, clusjx, newid))
            raise

    @property
    def centers(self):
        return np.array([c.center for c in self.values()])

    @property
    def mvcenters(self):
        return np.array([c.mvcenter for c in self.values()])

    @property
    def clustering(self):
        """
        Generate the current clustering labels for this dataset given the idxs in each entry
        Always return [0, nclus) as the cluster labels
        """
        cluslabels = np.zeros(self._ndata, dtype=int)
        for iclus, (_, cluster) in enumerate(self.items()):
            cluslabels[cluster.idxs] = iclus
        return cluslabels

    @property
    def categories(self):
        """ Unique categories is (0, nclus] """
        return np.arange(len(self))

    @property
    def clustercodesleft(self):
        """ The codes left in `self`, available for merging """
        return np.array(list(self.keys()))


@njit(cache=True)
def _primary_merge(closeidxs, mvdata, mvdist, assigned, num_take):
    """ Given the closeidxs, mvdata, assign data to different clusters """
    ndata = mvdata.shape[0]
    clustering = np.zeros(ndata, dtype=np.int32)
    nnears = closeidxs.shape[1] - 1
    mvdist = np.zeros(nnears, dtype=np.float64)
    clusnumber = 0
    for iloc in range(ndata):
        ix = closeidxs[iloc][0]
        didx = closeidxs[iloc][1:]
        # for each neighbor, compute the sqr eucldist and sort based on closest
        statfuncs.distance_sq_itemwise(ix, didx, mvdata, mvdist)
        didx = didx[np.argsort(mvdist)]
        # merge closest nearby points
        nassigned = 0
        for jx in didx:
            if assigned[ix] and not assigned[jx]:
                clustering[jx] = clustering[ix]
                assigned[jx] = True
            elif assigned[jx] and not assigned[ix]:
                clustering[ix] = clustering[jx]
                assigned[ix] = True
            elif not assigned[ix] and not assigned[jx]:
                clustering[ix] = clusnumber
                clustering[jx] = clusnumber
                assigned[ix] = True
                assigned[jx] = True
                clusnumber += 1
            nassigned += 1
            if nassigned >= num_take:
                break
    return clustering, clusnumber


def dss_single(mvdata, xyzlocs, nnears, num_take, target_nclus, searchparams=(0, 0, 0, 1, 1, 1),
               rseed=69069, minfound=0.01, maxfound=0.99):
    """
    Find a single dual-space clustering, separated out so that it can be easily parallelized

    Parameters
    ----------
    mvdata, xyzlocs : ndarray
        ndata x dim arrays of the multivariate data and locations, respectively
    nnears, num_take, target_nclus : int
        Number of nearest neighbors to search, number to merge, and target nclus for final merging
    searchparams : 6-tuple
        Tuple of (ang1, ang2, ang3, r1, r2, r2)
    rseed : int
        Random seed
    minfound : float
        Reject clusterings if any of the found clusters have < this prop
    maxfound : float
        Reject clusterings if any of the found clusters have > this prop

    Returns
    -------
    clustering : ndarray
        Integer array of cluster codes from 1 -> nclus

    """
    from scipy.spatial import cKDTree
    # init and allocate
    rng = np.random.RandomState(rseed)
    ndata, nvar = mvdata.shape
    mvdist = np.zeros(nnears, dtype=np.float64)
    assigned = np.zeros(ndata, dtype=bool)
    # make the kdtree
    anisotree, rotlocs = build_anisotree(searchparams, xyzlocs)
    # loop until explicitly return
    ireject = 0
    while True:
        assigned[:] = False
        idx = rng.permutation(ndata)
        closeidxs = anisotree.query(rotlocs[idx, :], nnears + 1)[1]

        # ------ primary merging stage, based on the spatial neighbors -----------------------------
        clustering, iclus = _primary_merge(closeidxs, mvdata, mvdist, assigned, num_take)

        # build the dictionary of clusters for merging
        allclusters = ClusterSet(xyzlocs, mvdata, clustering)

        # ------ secondary merging based on the MV differences of the populations ------------------
        nclus_left = 1e21
        while nclus_left > target_nclus + 1:
            clusters = allclusters.clustercodesleft
            nclus_left = len(clusters)
            iclus = clusters.max() + 1
            # choose a random cluster center, store the data indexes to labels as clusix
            ix = rng.randint(0, nclus_left)
            # update the center array
            centers = allclusters.mvcenters
            center_kdtree = cKDTree(centers)
            cidx = center_kdtree.query(centers[ix, :], k=nnears + 1)[1]
            cidx = cidx[1:max(nclus_left, 0)]
            # index into the dictionary with the cluster key
            x = allclusters[clusters[ix]].mvdat
            # get the divergence between x (ix) and the rest of the clusters in the search
            mv_popdist = []
            mv_centeridx = []
            for i, jx in enumerate(cidx):
                if jx == ix:
                    continue
                y = allclusters[clusters[jx]].mvdat
                # based on the number of data in each vector, compute the divergence
                if x.shape[0] > 25 and y.shape[0] > 25:
                    diffs = statfuncs.wards_distance_mdist(x, y, 1e-3)
                else:
                    diffs = statfuncs.wards_distance(x, y)
                mv_centeridx.append(jx)
                mv_popdist.append(diffs)
            jx = np.array(mv_centeridx)[np.argsort(mv_popdist)][0]
            allclusters.merge_clusters(clusters[ix], clusters[jx], iclus)

        # ------ do some checks and return the clustering if its `good` ----------------------------
        final_categories = allclusters.categories
        final_clustering = allclusters.clustering
        props = np.array([np.count_nonzero(final_clustering == cat) / ndata
                          for cat in final_categories])
        if (props > minfound).all() and (props < maxfound).all():
            return final_clustering
        else:
            ireject += 1
            if ireject > 1000:
                raise ValueError('Rejected too many clusterings, check parameters!')
