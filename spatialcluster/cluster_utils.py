""" (c) Ryan Martin 2018 under MIT license """

import multiprocessing as mp

import numba
import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture

from .statfuncs import wards_distance
from .clustermetrics import tdiff_wards


def gmm_cluster(nclus, mvdata):
    """ sklearn GMM clustering """
    gmm = GaussianMixture(nclus)
    gmm.fit(mvdata)
    return gmm.predict(mvdata) + 1, gmm


def km_cluster(nclus, mvdata):
    """ sklearn KM clustering """
    km = KMeans(nclus, n_init=1)
    km.fit(mvdata)
    return km.predict(mvdata) + 1, km


def hier_cluster(nclus, mvdata):
    """ sklearn ward clustering """
    agg = AgglomerativeClustering(n_clusters=nclus)
    agg.fit(mvdata)
    return agg.labels_ + 1, agg


def predict_cluster_labels(trainlocs, traindata, clusdefs, testlocs, testdata, nestimators=10):
    """
    Given a set of training locations, data and cluster definitions, train a random forest
    classifier to predict the cluster labels for a set of data sampled at test locations

    Parameters
    ----------
    trainloc, traindata : ndarray
        ndata x dim, ndata x nvar arrays of data
    clusdefs : ndarray
        1D array of cluster labels
    testlocs, testdata : ndarray
        ndata x dim, ndata x nvar arrays of data
    nestimators : int
        The number of trees found in the random forest
    """
    clf = RandomForestClassifier(n_estimators=nestimators)
    tr = np.c_[trainlocs, traindata]
    tst = np.c_[testlocs, testdata]
    return clf.fit(tr, clusdefs).predict(tst)


def classify_clusters(model, mvsamples):
    " predict the labels for the samples given the model "
    return model.predict(mvsamples) + 1


def cluster(nclus, dfs, n_init=1, variables=None, method='kmeans', rseed=None):
    """
    Parameters
    ----------
    nclus : int
        number of clusters
    dfs : DataFrame or ndarray
        the data
    n_init : int
        the number of times to initialize the clusterer
    variables : list
        the list of variables to take from the dataframe
    method : str
        one of `kmeans`, `hier` or `gmm`

    Returns
    -------
    clusdef : ndarray
        ndata long array of cluster codes
    model : sklearn model
        One of `KMeans`, `AgglomerativeClustering` or `GaussianMixture`

    """
    from sklearn.cluster import KMeans, hierarchical
    from sklearn.mixture import GaussianMixture
    # get the data for clustering:
    if variables is not None:
        nd_pts = dfs[variables].values
    else:
        nd_pts = dfs
    nclus = int(nclus)
    if method.lower() == 'kmeans':
        model = KMeans(n_clusters=nclus, n_init=n_init, random_state=rseed)
        clusdef = model.fit(nd_pts).labels_
    elif method.lower() == 'gmm':
        model = GaussianMixture(n_components=nclus, n_init=n_init, random_state=rseed)
        model = model.fit(nd_pts)
        clusdef = model.predict(nd_pts)
    elif method.lower() == 'hier':
        model = hierarchical.AgglomerativeClustering(n_clusters=nclus, linkage='ward')
        clusdef = model.fit_predict(nd_pts)
    else:
        raise ValueError("Invalid clustering `method`, try one of `kmeans`, `gmm` or `hier`")
    return clusdef, model


def reclass_clusters_byvar(base, clusters, variables):
    """
    Recode `clusters` to have the greatest match to `base` based on the contained continuous
    `variable`

    Parameters
    ----------
    base : ndarray, pd.DataFrame
        The base classification that all `clusters` are reclassified against
    clusters : ndarray, pd.DataFrame
        Either a 1D with the same length as `base`, or a 2D array with realizations as columns
    variables : ndarray, pd.DataFrame
        The variable(s) used to determine similarity between clusters

    Returns
    -------
    reclassed : ndarray
        Reclassified clusters according to `clusters`
    diffs : ndarray
        Missmatch scores between the clusterings and the targets
    """
    import itertools
    if hasattr(base, 'values'):
        base = base.values
    if hasattr(clusters, 'values'):
        clusters = clusters.values
    if hasattr(variables, 'values'):
        variables = variables.values
    uclus = np.unique(base).astype(int)
    perms = np.array([p for p in itertools.permutations(uclus)])
    clusters = clusters.astype(int)

    if clusters.ndim == 1 or (clusters.ndim == 2 and clusters.shape[1] == 1):
        return _reclass_single_byvar(uclus, perms, base, clusters, variables)

    # else multiple clusterings were passed
    nreals = clusters.shape[1]
    reclassed = np.zeros_like(clusters, dtype=int)
    scores = []
    for ireal in range(nreals):
        reclassed[:, ireal], score = _reclass_single_byvar(uclus, perms, base, clusters[:, ireal],
                                                           variables)
        scores.append(score)
    return reclassed, scores


@numba.njit(cache=True)
def _reclass_single_byvar(uclus, permclus, target, single, variables):
    """
    Recodes a `single` such that the pairwise distance between base and clusters is minimized based
    on the multivariate distances from wards connectivity
    """
    nperm = permclus.shape[0]
    ndata = target.shape[0]
    permuted = np.zeros((nperm, ndata), dtype=np.int_)
    scores = []
    for iperm in range(nperm):
        # convert the single clustering to the new permutation e.g. orig : new mappings
        for orig, new in zip(uclus, permclus[iperm, :]):
            oidx = single == orig
            permuted[iperm][oidx] = new
        # get the sum of pairwise similaritity
        scores.append(_prop_equal_byvar(uclus, target, permuted[iperm, :], variables))
    scores = np.array(scores)
    ireclass = np.argsort(scores)[0]  # sorts smallest to largest, and want the smallest
    return permuted[ireclass, :], scores[ireclass]


@numba.njit(cache=True)
def _prop_equal_byvar(uclus, clus1, clus2, variables):
    """ jit friendly counter of pairwise MV distances """
    assert len(clus1) == len(clus2)
    dist = 0.0
    for cluscode in uclus:
        p1 = variables[clus1 == cluscode]
        p2 = variables[clus2 == cluscode]
        dist += wards_distance(p1, p2)
    return dist / len(uclus)


def reclass_clusters(base, clusters):
    """
    Recode `clusters` to have the greatest match to `base`
    Parameters
    ----------
    base : ndarray, pd.DataFrame
        The base classification that all `clusters` are reclassified against
    clusters : ndarray, pd.DataFrame
        Either a 1D with the same length as `base`, or a 2D array with realizations as columns

    Returns
    -------
    reclassed : ndarray
        Reclassified clusters according to `clusters`
    diffs : ndarray
        Missmatch scores between the clusterings and the targets
    """
    import itertools
    if hasattr(base, 'values'):
        base = base.values
    if hasattr(clusters, 'values'):
        clusters = clusters.values
    uclus_base = np.unique(base).astype(int)
    uclus_clus = np.unique(clusters).astype(int)
    assert len(uclus_base) == len(uclus_clus), 'ERROR: cluster number mismatch between clusterings'
    perms = np.array([p for p in itertools.permutations(uclus_base)])
    clusters = clusters.astype(int)

    # if only a single clustering is passed for reclassification
    if clusters.ndim == 1 or (clusters.ndim == 2 and clusters.shape[1] == 1):
        return _reclass_single_perms(uclus_clus, perms, base, clusters)

    # else multiple clusterings were passed
    clusters_t = clusters.T
    nreals = clusters.shape[1]
    reclassed = np.zeros_like(clusters_t, dtype=int)
    scores = []
    for ireal in range(nreals):
        reclassed[ireal, :], score = _reclass_single_perms(uclus_clus,
                                                           perms, base, clusters_t[ireal, :])
        scores.append(score)
    return reclassed.T, scores


@numba.njit(cache=True)
def _prop_equal(clus1, clus2):
    """ jit friendly counter """
    assert len(clus1) == len(clus2)
    count = 0.0
    for c1, c2 in zip(clus1, clus2):
        if c1 == c2:
            count += 1.0
    return count / len(clus1)


@numba.njit(cache=True)
def _reclass_single_perms(uclus, permutations, target, single):
    """
    Permutes a `single` clustering to have the greatest match to the `target` clustering.
    """
    nperm = permutations.shape[0]
    ndata = target.shape[0]
    permuted = np.zeros((nperm, ndata), dtype=np.int_)
    scores = []
    for iperm in range(nperm):
        # convert the single clustering to the new permutation e.g. orig : new mappings
        for orig, new in zip(uclus, permutations[iperm, :]):
            oidx = single == orig
            permuted[iperm][oidx] = new
        # get the sum of pairwise similaritity
        scores.append(_prop_equal(permuted[iperm, :], target))
    scores = np.array(scores)
    ireclass = np.argsort(scores)[-1]  # sorts smallest to largest, and want the largest
    return permuted[ireclass, :], scores[ireclass]


def pairwise_cluster_similarity(clusterings, nprocesses=None):
    """
    For a large clusterings matrix (e.g., ndata x nclusterings) recode all clusterings to find the
    pairwise maximum similarity and return the similarity score. Effectively the pairwise similarity
    between the clusterings, but not the global clustering between all clusterings

    Parameters
    ----------
    clusterings: np.ndarray
        ndata x nclusterings array of different cluster codings, integer values assigning each
        data location to a cluster during the run. Permutations permitted since the jaccard index
        is permutation independent
    nprocesses: int
        number of parallel processes to spawn

    Returns
    -------
    dist : ndarray
        nd x nd array of pairwise similarities after reclassifying each clustering

    """
    from .utils import log_progress
    nreal = np.size(clusterings, 1)
    differencematrix = np.zeros((nreal, nreal))
    res = {}
    if nprocesses is None or nprocesses == 1:
        for i in range(nreal):
            # keep the unique dictionary key for each parallel run
            res[i] = reclass_clusters(clusterings[:, i], clusterings)
            differencematrix[i, :] = res[i][1]
    else:
        # start the parallel processing:
        pool = mp.Pool(processes=nprocesses)
        # start the parallel processes:
        for i in range(nreal):
            # keep the unique dictionary key for each parallel run
            res[i] = pool.apply_async(reclass_clusters, (clusterings[:, i], clusterings))
        [res[key].get() for key in log_progress(res.keys(), name='Clusterings Processed')]
        pool.close()
        pool.join()

        # collect the results:
        for i in range(nreal):
            _, diffs = res[i].get()
            differencematrix[i, :] = diffs
    return differencematrix


def pairings_matrix(clusterings, weights=None):
    """
    Return the pairwise matrix of 'number of times paired' over all clustering runs

    Parameters
    ----------
    clusterings : ndarray or cluster object
        ndata x nreal array of clustercodes
    weights : ndarray
        optional nreal-long clustering weights

    Returns
    -------
    pairings : ndarray
        ndata x ndata array of pairings
    """
    try:
        clusterings = clusterings.clusterings
    except AttributeError:
        pass
    if weights is None:
        return _pairings_matrix(clusterings)
    return _pairings_matrix_weights(clusterings, weights)


@numba.njit(parallel=True)
def _pairings_matrix_weights(clusterings, weights):
    """ numba jit compiled function to return the pairings matrix given the unordered set of
    clustering  # realiations in the clustering array
    NOTE: tested with numba 0.26.xx; conda install numba = 0.26.0
    Clusterings: ndata x nreal array of clusterings
    weights: real - long array of weights associated with each clustering
    """
    ndata, nreal = clusterings.shape
    numpairings = np.zeros((ndata, ndata), dtype=np.float64)
    for ireal in numba.prange(nreal):
        cluscodes = clusterings[:, ireal]
        clustering_weight = weights[ireal]
        for i in range(ndata):
            for j in range(ndata):
                if cluscodes[i] == cluscodes[j]:
                    numpairings[i, j] += clustering_weight
    numpairings /= weights.sum()
    return numpairings


@numba.njit(parallel=True)
def _pairings_matrix(clusterings):
    """ numba jit compiled function to return the pairings matrix given the unordered set of
    clustering  # realiations in the clustering array
    """
    ndata, nreal = clusterings.shape
    numpairings = np.ones((ndata, ndata), dtype=np.float64)
    for i in numba.prange(ndata):
        for j in range(i + 1, ndata):
            for ireal in range(nreal):
                if clusterings[i, ireal] == clusterings[j, ireal]:
                    numpairings[i, j] += 1
    numpairings = numpairings / nreal
    for i in range(ndata):
        numpairings[i, i] = 1.0
        for j in range(i + 1, ndata):
            numpairings[j, i] = numpairings[i, j]
    return numpairings


def cluster_probability(clusdefs, pairingsmatrix):
    """ Using the pairwise - pairings matrix, return the probability of each point to belong to
    each cluster in the form of ndata x nclus columns
    Parameters:
        clusdefs(array): nd integer array of cluster definitions
        pairingsmatrix(array): nd x nd array of the pair count per data combination
    Returns:
        ndarray(nd x nclus) of the probability for each point to belong to each clustering
    """
    ucats = np.unique(clusdefs)
    return _cluster_probability(ucats, clusdefs, pairingsmatrix)


@numba.njit(cache=True)
def _cluster_probability(clusids, clusdefs, pairingsmatrix):
    """
    Parameters
    ----------
    clusids : ndarray
        The unique cluster ids in clusdefs, clusids = np.unique(clusdefs)
    clusdefs : ndarray
        The array of ids for each ndata location
    pairingsmatrix : ndarray
        ndata x ndata pairings obtained with pairings_matrix

    Returns
    -------
    clusterprobs : ndarray
        pairwise probability to be in each cluster
    """
    assert clusdefs.ndim == 1, 'ERROR: clusdefs should be a 1-D array of cluster definitions'
    ndata = clusdefs.shape[0]
    alldatidx = np.arange(ndata)
    clusterprobs = np.ones((ndata, len(clusids)))
    for iclus, clusid in enumerate(clusids):
        # indexes to data inside this cluster
        idxin = alldatidx[clusdefs == clusid]
        # get the ratio
        for jx in alldatidx:
            probability_inside = 0.0
            for ix in idxin:
                if jx == ix:
                    continue
                probability_inside += pairingsmatrix[jx, ix]
            clusterprobs[jx, iclus] = probability_inside / pairingsmatrix[ix, :].sum()
    return clusterprobs


def cluster_probability_bycount(clusdefs, clusterings):
    """
    Parameters
    ----------
    clusdefs : ndarray
        The array of ids for each ndata location
    clusterings : ndarray
        ndata x nclus matrix of clusterings, assume they are not recoded, and recode

    Returns
    -------
    clusterprobs : ndarray
        pairwise probability to be in each cluster
    """
    ndata, nreal = clusterings.shape
    uclus = np.unique(clusdefs)
    nclus = len(uclus)
    # make sure the clusterings all have the same nclus
    if any(len(np.unique(c)) != nclus for c in clusterings.T):
        return np.zeros((ndata, nclus), dtype=np.float64)
    # else continue with calcs
    clusreclassed, _ = reclass_clusters(clusdefs, clusterings)
    for u1, u2 in zip(uclus, np.unique(clusreclassed)):
        assert u1 == u2, 'ERROR: incompatible clusters between `clusdefs` and `clusterings`'
    clusterprobs = np.zeros((ndata, nclus), dtype=np.float64)
    for idata in range(ndata):
        for iclus, clusid in enumerate(uclus):
            clusterprobs[idata, iclus] = (clusreclassed[idata, :] == clusid).sum() / nreal
    return clusterprobs


def hierarchical_clustering(pairingsmatrix, nclus, method='ward'):
    """
    Parameters
    ----------
    pairingsmatrix : ndarray
        ndata x ndata array of pairings calculated with `pairings_matrix` or similar
    nclus : int
        number of clusters to generate
    method : str
        linakge method, `ward` or `average` recommended

    """
    from scipy.cluster import hierarchy
    linkage = hierarchy.linkage(pairingsmatrix, method=method)
    clusdefs = hierarchy.fcluster(linkage, nclus, criterion='maxclust')
    return clusdefs


def consensus(clusterings, nclus, weights=None, method='hier', refclus=None):
    """
    Consensus by clustering of the pairings matrix, using hierarchical or spectral clustering

    Parameters
    ----------
    clusterings : ndarray
        ndata x nreal array of cluster realizations
    nclus : int
        the number of clusters to generate
    weights : ndarray
        nreal-long array of weights for each clustering
    method : str
        clustering method for the pairings matrix, either `hier` or `spec`
    refclus : ndarray
        A reference clustering for this dataset that the target will be recoded too

    Returns
    -------
    final_clusterings : ndarray
        1D array of final cluster labels given the passed parameters
    clusterprobs : ndarray
        ndata x nclus array of likelihood to be in each cluster

    """
    from sklearn.cluster import spectral_clustering
    try:
        clusterings = clusterings.clusterings
    except AttributeError:
        pass
    pairings = pairings_matrix(clusterings, weights)
    # use the selected nd x nd matrix clustering method
    if method == 'hier':
        final_clusters = hierarchical_clustering(pairings, nclus, method='ward')
    else:
        final_clusters = spectral_clustering(pairings, n_clusters=nclus)

    if refclus is not None:
        final_clusters, _ = reclass_clusters(refclus, final_clusters)
        final_ensemble, _ = reclass_clusters(refclus, clusterings)
        # if a reference clustering is passed also recode the passed ensemble
        for i in range(final_ensemble.shape[1]):
            clusterings[:, i] = final_ensemble[:, i]

    clusterprobs = cluster_probability_bycount(final_clusters, clusterings)

    return final_clusters, clusterprobs, pairings


# ---------------------------------------------------------------
# MODIFIED CONSENSUS FUNCTION with the voting for the majority
# and supporting functions
# ---------------------------------------------------------------
@numba.njit(cache=True)
def _choose(N, K):
    """ The binomial coefficient, `N choose K` """
    if K > N:
        return 0.0
    elif K == 0:
        return 1.0
    elif K > (N / 2.0):
        return _choose(N, N - K)
    else:
        return N * _choose(N - 1.0, K - 1.0) / K


@numba.njit(cache=True)
def _unique(vector):
    """ replacement for `np.unique` in jitted functions """
    vector_sorted = np.sort(vector)
    idxs = np.concatenate((np.ones(1, dtype=np.bool_), vector_sorted[:-1] != vector_sorted[1:]))
    return vector_sorted[idxs]


@numba.njit(cache=True)
def _onehot(clustering):
    """
    Parameters
    ----------
    clustering : ndarray
        1D clustering object contaning integers to recode.

    Returns
    -------
    encoding : ndarray
        (nclus x ndata) dimensioned array of one-hot encodings where rows correspond to low->high
        sorted values from `_unique`
    """
    uids = _unique(clustering)
    nclus = len(uids)
    ndata = clustering.shape[0]
    encoding = np.zeros((nclus, ndata), dtype=np.float_)
    for i, clid in enumerate(uids):
        encoding[i][clustering == clid] = 1
    return encoding, nclus


@numba.jit(cache=True)
def adjusted_rand_index(target, clustering):
    """
    JITd adjusted rand index implementation
    """
    ndata = clustering.shape[0]
    b1, num1 = _onehot(target)
    b2, num2 = _onehot(clustering)
    # the correspondance matrix, results in cmat[num1, num2]
    cmat = b1.dot(b2.T)
    # compute the different accumulators
    sumi = 0.0
    two = 2.0
    for i in range(num1):
        sumi += _choose(cmat[i, :].sum(), two)
    sumj = 0.0
    for j in range(num2):
        sumj += _choose(cmat[:, j].sum(), two)
    sumij = 0.0
    for i in range(num1):
        for j in range(num2):
            sumij += _choose(cmat[i, j], two)
    chance_factor = (sumi * sumj) / _choose(ndata, two)
    return (sumij - chance_factor) / (0.5 * (sumi + sumj) - chance_factor)


def consensus_agg(base, clusterings):
    """
    Get the fuzzy clustering corresponding to the ensemble given the input dataset
    From: Rathore et al 2017 "Ensemble Fuzzy Clusteirng using Cumulative Aggregation...."
    """
    nreal = clusterings.shape[1]
    # copy, transpose and get the similarity score between ref and sort the clusterings
    clus_t = clusterings.copy().T
    diffs = [adjusted_rand_index(base, clus_t[i, :]) for i in range(nreal)]
    sortorder = np.argsort(diffs)[::-1]  # highest to lowest
    clus_t = clus_t[sortorder, :]
    # The base encoding
    Ub_s = [_onehot(base)[0]]
    for z, clus in enumerate(clus_t):
        i = z + 2
        Ui = _onehot(clus)[0]
        Wi_b = Ub_s[-1] @ np.linalg.pinv(Ui)
        Ui_b = Wi_b @ Ui
        updated = ((i - 1) / i) * Ub_s[-1] + (1 / i) * Ui_b
        Ub_s.append(updated)
    return Ub_s, diffs
