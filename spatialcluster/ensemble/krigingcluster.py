"""
Kriging Random Path Clustering

(c) Ryan Martin 2018 under GPLv3 license
"""
import numpy as np

from ..anisokdtree import build_anisotree
from .baseensemble import BaseEnsemble
from ..fastcova import kriging_matrices, ordkriging_matrices
from ..utils import log_progress, rseed_list


class KrigeCluster(BaseEnsemble):
    """
    Clustering by Kriging on a random path
    {}

    KrigeCluster Parameters
    -----------------------
    numtake : int
        number of nearest neighbors to merge in the spatial search, numtake <= nnears

    .. codeauthor:: Ryan Martin - 04-07-2018
    """
    __doc__ = __doc__.format(BaseEnsemble.__doc__)

    def __init__(self, locations, mvdata, seedcodes, varios, nreal=100, rseed=51231, krigtype="ok",
                 minfound=0.001, maxfound=0.999, knsearch=50, kaniso=(0, 0, 0, 100, 100, 20),
                 ensearch=100, eaniso=(0, 0, 0, 100, 100, 20), randprop=0.9):
        super().__init__(mvdata, locations, nreal, rseed, minfound, maxfound)
        assert(isinstance(varios, dict)), \
            "ERROR: variodict must be a dictionary with keys vdict[cat, var] = gg.VarModel()"
        # super() column-wise standardizes the mvdata by default
        self.mvdata = mvdata
        self.locations = locations
        self.seedcodes = seedcodes
        self.varios = varios
        self.pars.update(dict(krigtype=krigtype, knsearch=knsearch, randprop=randprop,
                              kaniso=kaniso, ensearch=ensearch, eaniso=eaniso))

    def fit(self, nprocesses=4, verbose=True):
        """
        Parameters
        ----------
        nprocesses : int
            specifies how many parallel processes to use

        Returns
        -------
        clusterings : ndarray

        """
        import multiprocessing as mp
        assert nprocesses is None or nprocesses >= 1, 'invalid nprocesses'
        ndata = self.mvdata.shape[0]
        seeds = rseed_list(self.pars['nreal'], int(self.pars['rseed']))
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
                args = (self.locations, self.mvdata, self.seedcodes, self.varios,
                        self.pars['krigtype'], self.pars['knsearch'], self.pars['ensearch'],
                        self.pars['kaniso'], self.pars['eaniso'], self.pars['randprop'], int(seed))
                results[iproc] = pool.apply_async(krigecluster, args, callback=updatefunc)
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
                args = (self.locations, self.mvdata, self.seedcodes, self.varios,
                        self.pars['krigtype'], self.pars['knsearch'], self.pars['ensearch'],
                        self.pars['kaniso'], self.pars['eaniso'], self.pars['randprop'], int(seed))
                self.clusterings[:, iproc] = krigecluster(*args)


def rotatedlocations(locations, aniso=(0, 0, 0, 1, 1, 1)):
    """
    Return a single or dictionary of rotated locations according to the
    given anisotropy
    """
    if isinstance(aniso, dict):
        rotlocs = {}
        for cat, apar in aniso.items():
            _, rotlocs[cat] = build_anisotree(apar, locations)
    else:
        _, rotlocs = build_anisotree(aniso, locations)
        return rotlocs


def updatetrees(locations, newcodes, aniso=(0, 0, 0, 1, 1, 1)):
    """
    generate a new set of kdtree's to search for a given location
    """
    trees = {}
    treelocs = {}
    treevals = {}
    for cat in np.unique(newcodes):
        if isinstance(aniso, dict):
            anisopars = aniso[cat]
        else:
            anisopars = aniso
        idxs = newcodes == cat
        trees[cat], treelocs[cat] = build_anisotree(anisopars, locations[idxs, :])
        treevals[cat] = np.arange(locations.shape[0])[idxs]
    return treelocs, treevals, trees


def krigecluster(locations, mvdata, seedcodes, varios, ktype='sk', knsearch=45, ensearch=45,
                 kaniso=(0, 0, 0, 100, 100, 100), eaniso=(0, 0, 0, 100, 100, 100), randprop=0.9,
                 rseed=6341112, verbose=False):
    """
    Re-assign categories and update the local search and data populations used for kriging.
    Optionally allow 'poor' allocations (those that are not the majority in the local search) with
    the randprop parameter the amount of influence from the kriging system with randprop

    Parameters
    ----------
    locations : 2d ndarray
        The locations of the samples
    mvdata : 2d ndarray
        The multivariate dataset
    seedcodes : 1d ndarray
        The 1d array of codes to seed the kriging run with
    knsearch : int
        the number of neighbors to use for kriging
    kaniso : 6-tuple
        The search anisotropy for kriging
    ensearch : int
        The number of neighbors to consider for local proportions and updating categories
    eaniso : 6-tuple
        The search anisotropy for the proportions
    varios : dict
        Dictionary of cat, var: gg.VarModel
    ktype : str
        one of 'ok' or 'sk'
    randprop : float
        A value between 0 and 1, high values result in less random chance to have
        categories reassigned

    .. codeauthor:: Ryan Martin - 29-08-2018
    """
    import numpy as np
    rng = np.random.RandomState(rseed)

    ndata, nvar = mvdata.shape
    currentcodes = seedcodes.copy()
    ucats = np.unique(seedcodes)
    ncats = len(ucats)

    # get the variable names from the dictionary
    variables = []
    for (c, v), _ in varios.items():
        if c not in ucats:
            raise ValueError(
                "ERROR: `varios` must be a dictionary with keys (cat, var): gg.VarModel")
        if v not in variables:
            variables.append(v)

    # setup the trees
    treelocs, index2orig, trees = updatetrees(locations, currentcodes, kaniso)
    rotlocs = rotatedlocations(locations, kaniso)
    setree, selocs = build_anisotree(eaniso, locations)

    # go through the random path
    randompath = rng.permutation(ndata)
    ests = np.zeros((ncats, nvar), dtype=np.float64)

    for iloc in log_progress(randompath) if verbose else randompath:
        ests[:] = 0.0
        for icat, c in enumerate(ucats):
            u = rotlocs[c][iloc, :] if isinstance(rotlocs, dict) else rotlocs[iloc, :]
            maxdis = kaniso[c][3] if isinstance(kaniso, dict) else kaniso[3]
            nsearch = min(treelocs[c].shape[0], knsearch)
            dists, idxs = trees[c].query(u, k=nsearch)
            idxs = idxs[dists < maxdis]
            # omit this location if estimating with the same category
            if currentcodes[iloc] == c:
                idxs = idxs[1:]
            if idxs.shape[0] > 2:
                for ivar, var in enumerate(variables):
                    if ktype == 'ok':
                        K, k = ordkriging_matrices(u, treelocs[c][idxs, :], varios[c, var])
                        ests[icat, ivar] = \
                            np.linalg.solve(K, k)[:-1] @ mvdata[index2orig[c][idxs], ivar]
                    else:
                        K, k = kriging_matrices(u, treelocs[c][idxs, :], varios[c, var])
                        ests[icat, ivar] = \
                            np.linalg.solve(K, k) @ mvdata[index2orig[c][idxs], ivar]
            else:
                ests[icat, :] = 0.0
        # re-assign iloc to the category that has the max estimate
        mvdiffs = np.abs((ests - mvdata[iloc, :]) / np.maximum(1e-5, ests)).mean(axis=1)
        assignedcat = ucats[np.argmin(mvdiffs)]
        if assignedcat != currentcodes[iloc]:
            # _, idxs = setree.query(selocs[iloc, :], ensearch)
            # localcats = currentcodes[idxs]
            # props = [np.count_nonzero(localcats == c) / ensearch
            #          for c in np.unique(localcats)]
            # if assignedcat == localcats[np.argmax(props)] or np.random.rand() > randprop:
            currentcodes[iloc] = assignedcat
            # now we need to update the trees
            try:
                treelocs, index2orig, trees = \
                    updatetrees(locations, currentcodes, kaniso)
                ucats = np.array(list(trees.keys()))
            except:
                pass
    return currentcodes
