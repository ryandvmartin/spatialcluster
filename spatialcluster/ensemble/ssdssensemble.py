"""
Dual Space Search Ensemble Clustering

(c) Ryan Martin 2018 under GPLv3 license
"""

import numpy as np

from .. import statfuncs
from ..anisokdtree import build_anisotree
from ..utils import log_progress, rseed_list
from .baseensemble import BaseEnsemble
from .dssensemble import ClusterSet


class SSDSSEnsemble(BaseEnsemble):
    """
    Semi-Supervised Dual-Space-Search Ensemble Clustering
    {}

    DSS Parameters
    --------------
    numtake : int
        number of nearest neighbors to merge in the spatial search, numtake <= nnears

    SSDSS Parameters
    ----------------
    domaincodes : ndarray
        A set of integers of the same length as `mvdata` that assign each location to one of K
        target stationary domains
    seedprop : float
        The proportion of `known` domain codes to seed each ensemble clustering run with

    .. codeauthor:: Ryan Martin - 04-07-2018
    """
    __doc__ = __doc__.format(BaseEnsemble.__doc__)

    def __init__(self, mvdata, locations, nreal=100, nnears=10, rseed=69069, minfound=0.001,
                 maxfound=0.999, searchparams=(0, 0, 0, 500, 500, 500), numtake=5,
                 domaincodes=None, seedprop=0.25):
        assert domaincodes is not None, 'ERROR: Must pass `domaincodes` to this object!'
        super().__init__(mvdata, locations, nreal, nnears, rseed, minfound, maxfound, searchparams)
        try:
            domaincodes = domaincodes.values
        except AttributeError:
            pass
        if domaincodes.ndim == 2:
            if domaincodes.shape[1] == 1:
                domaincodes = domaincodes.flatten()
            else:
                raise ValueError("ERROR: `domaincodes` must be a 1D array of domain codes")
        assert domaincodes.ndim == 1, 'ERROR: must pass a 1D array of domain codes'
        assert domaincodes.shape[0] == self.mvdata.shape[0], (
            'ERROR: size mismatch between data and domain codes')
        # generate the code map and the new vector from (0->nclus]
        codemap = {}
        reclassified = domaincodes.copy().astype(int)
        for icode, code in enumerate(np.unique(domaincodes)):
            reclassified[domaincodes == code] = icode
            codemap[icode] = code
        self._tempcodes = reclassified  # mapping (0->nclus)
        self.domaincodes = domaincodes
        self.pars['codemap'] = codemap
        # add the extra parameters for this style of ensemble clustering
        assert 0 < seedprop <= 1, "ERROR: `seedprop` should be in the interval [0, 1)"
        self.pars['nseed'] = int(np.ceil(seedprop * mvdata.shape[0]))
        self.pars['numtake'] = numtake

    def fit(self, verbose=True, nprocesses=None):
        """
        Generate the clustering ensemble
        Parameters
        ----------
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
        domaincodes = self._tempcodes
        target_nclus = len(np.unique(domaincodes))
        ndata, _ = mvdata.shape
        seeds = rseed_list(self.pars['nreal'], self.pars['rseed'])
        self.clusterings = np.zeros((ndata, self.pars['nreal']))
        self._miniclus = self.clusterings.copy()
        # setup the parallel args list
        if nprocesses is not None:
            if verbose:
                pbar = log_progress(range(len(seeds)), name='Clusterings')

                def pbar_update(*args, **kwargs):
                    pbar.update()
            else:
                pbar_update = None
            pool = mp.Pool(processes=nprocesses)
            results = {}
            for iproc, seed in enumerate(seeds):
                args = (mvdata, locations, domaincodes, self.pars['nseed'],
                        self.pars['nnears'], self.pars['numtake'], target_nclus,
                        self.pars['searchparams'], int(seed), self.pars['minfound'],
                        self.pars['maxfound'])
                results[iproc] = pool.apply_async(ssdss_single, args, callback=pbar_update)
            pool.close()
            pool.join()
            pbar.close()
            for iproc in range(self.pars['nreal']):
                self.clusterings[:, iproc], self._miniclus[:, iproc] = results[iproc].get()
        else:
            iterable = log_progress(seeds) if verbose else seeds
            for iproc, seed in enumerate(iterable):
                args = (mvdata, locations, domaincodes, self.pars['nseed'],
                        self.pars['nnears'], self.pars['numtake'], target_nclus,
                        self.pars['searchparams'], int(seed), self.pars['minfound'],
                        self.pars['maxfound'])
                self.clusterings[:, iproc], self._miniclus[:, iproc] = ssdss_single(*args)

    def predict(self, weights=None, method='spec'):
        """
        Override the superclass predict method because `nclus` is specified by the input dataset
        """
        from ..cluster_utils import reclass_clusters
        from copy import deepcopy
        nclus = len(np.unique(self._tempcodes))
        nreal = self.clusterings.shape[1]
        self.labels = super().predict(nclus, weights, method)
        # since we know the `true codes` (0->nclus],
        # automatically relcassify the final and all in ensemble
        self.labels, _ = reclass_clusters(self._tempcodes, self.labels)  # _tempcodes is 0->nclus
        self.clusterings, _ = reclass_clusters(self._tempcodes, self.clusterings)
        # for each iclus count the number of times each location is equal to that
        self.clusprob[:] = 0.0
        for clusreal in self.clusterings.T:
            for iclus in range(nclus):
                self.clusprob[clusreal == iclus, iclus] += 1.0
        self.clusprob /= nreal
        # finally map everything back to the output codes
        codemap = self.pars['codemap']
        temp = deepcopy(self.labels)
        for iclus in range(nclus):
            self.labels[temp == iclus] = codemap[iclus]
        for ireal in range(nreal):
            temp = deepcopy(self.clusterings[:, ireal])
            for iclus in range(nclus):
                self.clusterings[temp == iclus, ireal] = codemap[iclus]
        return self.labels


def ssdss_single(mvdata, xyzlocs, domaincodes, nseed, nnears, num_take, target_nclus,
                 searchparams, rseed, minfound, maxfound):
    """
    Find a semi-supervised dual-space clustering

    Parameters
    ----------
    mvdata, xyzlocs : ndarray
        ndata x dim arrays of the multivariate data and locations, respectively
    domaincodes : ndarray
        1D array of stationary domain codes that are checked
    nseed : int
        Number of samples to give starting `domaincode`s
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
    ndata, _ = mvdata.shape
    mvdist = np.zeros(nnears, dtype=np.float64)
    assigned = np.zeros(ndata, dtype=bool)
    # make the kdtree
    anisotree, rotlocs = build_anisotree(searchparams, xyzlocs)
    # loop until explicitly return
    ireject = 0
    while True:
        assigned[:] = False
        clustering = np.zeros(ndata, dtype=int)
        # track the number of clusters that are seeded
        nseeded = 0
        iclus = target_nclus + 1
        # --------------------------------------------------
        # Seed a minimum of 1 from each input code
        idx = rng.permutation(ndata)
        for i, c in enumerate(np.unique(domaincodes)):
            clustering[idx[i]] = c
            nseeded += 1
        idx = rng.permutation(ndata)
        closeidxs = anisotree.query(rotlocs[idx, :], nnears + 1)[1]
        # closeidxs now contains the [ndata, nnears] idxs to randomly permuted data
        # ------ primary merging stage, based on the spatial neighbors -----------------------------
        for idx in closeidxs:
            ix = idx[0]
            didx = idx[1:]
            # for each neighbor, compute the sqr eucldist and sort based on closest
            statfuncs.distance_sq_itemwise(ix, didx, mvdata, mvdist)
            didx = didx[np.argsort(mvdist)]
            # merge closest nearby points
            nassigned = 0
            for jx in didx:
                if assigned[ix] and not assigned[jx]:
                    if nseeded < nseed:
                        # only seed with merges where original codes are the same at each
                        # location
                        if domaincodes[ix] == domaincodes[jx]:
                            clustering[jx] = domaincodes[ix]
                            assigned[jx] = True
                            nseeded += 1
                            nassigned += 1
                    else:
                        clustering[jx] = clustering[ix]
                        assigned[jx] = True
                        nassigned += 1
                elif assigned[jx] and not assigned[ix]:
                    if nseeded < nseed:
                        if domaincodes[jx] == domaincodes[ix]:
                            clustering[ix] = domaincodes[jx]
                            assigned[ix] = True
                            nseeded += 1
                            nassigned += 1
                    else:
                        clustering[ix] = clustering[jx]
                        assigned[ix] = True
                        nassigned += 1
                elif not assigned[ix] and not assigned[jx]:
                    if nseeded < nseed:
                        if domaincodes[ix] == domaincodes[jx]:
                            clustering[ix] = clustering[jx] = domaincodes[ix]
                            assigned[ix] = assigned[jx] = True
                            nseeded += 1
                            nassigned += 1
                    else:
                        clustering[ix] = clustering[jx] = iclus
                        iclus += 1
                        assigned[ix] = assigned[jx] = True
                        nassigned += 1
                nassigned += 1
                if nassigned >= num_take:
                    break
        # copy this miniclustering to return with the final
        miniclustering = clustering.copy()

        # build the dictionary of clusters for merging
        allclusters = ClusterSet(xyzlocs, mvdata, clustering)

        # ------ secondary merging based on the MV differences of the populations ------------------
        nclus_left = 1e21
        while nclus_left > target_nclus:
            clusters = allclusters.clustercodesleft
            nclus_left = len(clusters)
            iclus = np.unique(clusters).max() + 1
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
                if ix == jx:
                    continue
                y = allclusters[clusters[jx]].mvdat
                # based on the number of data in each vector, compute the divergence
                if x.shape[0] < 50 or y.shape[0] < 50:
                    diffs = statfuncs.wards_distance(x, y)
                else:
                    diffs = statfuncs.wards_distance_mdist(x, y, 1e-4)
                mv_centeridx.append(jx)
                mv_popdist.append(diffs)
            jx = np.array(mv_centeridx)[np.argsort(mv_popdist)][0]
            # ----------------------------------------------------------------------
            # MODIFICATION FOR SEMI-SUPERVISED
            # Avoid merging or changing the codes of the original clusters (0,nclus]
            if clusters[ix] < target_nclus and clusters[jx] < target_nclus:
                continue
            elif clusters[ix] < target_nclus:
                iclus = clusters[ix]
            elif clusters[jx] < target_nclus:
                iclus = clusters[jx]
            # -----------------------------------------------------------------------
            allclusters.merge_clusters(clusters[ix], clusters[jx], iclus)

        # ------ do some checks and return the clustering if its `good` ----------------------------
        final_categories = allclusters.categories
        final_clustering = allclusters.clustering
        props = np.array([np.count_nonzero(final_clustering == cat) /
                          ndata for cat in final_categories])
        if all(props > minfound) and all(props < maxfound):
            # print("rejected {} clusterings!".format(ireject))
            return final_clustering, miniclustering
        else:
            ireject += 1
            if ireject > 1000:  # try to have an upper limit on the number of rejected clusters
                raise ValueError('Rejected too many clusterings, check parameters!')
