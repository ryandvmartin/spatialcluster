""" (c) Ryan Martin 2018 under GPLv3 license """

import numpy as np

from .. import cluster_utils as cutils
from .. import statfuncs as fstat
from ..accluster import _local_autocorr
from ..anisokdtree import build_anisotree
from .baseensemble import BaseEnsemble

_CLUS_STR_INT = dict(kmeans=0, gmm=1, hier=2)
_CLUS_INT_STR = {0: "kmeans", 1: "gmm", 2: "hier"}
_CLUSFUNCS = [cutils.km_cluster, cutils.gmm_cluster, cutils.hier_cluster]
_AUTOCORR_INT = dict(morans=1, getis=2)


class ACEnsemble(BaseEnsemble):
    """
    Generates an ensemble of clusterings using the random subspace and local autocorrelation stats
    {}

    Autocorrelation Parameters
    --------------------------
    cluster_method: str
        can be one of 'kmeans', 'gmm', or 'hier'
    acmetric: str or list
        the autocorrelation acmetric to consider, options are 'morans', 'getis', specify with
        1, 2 or [1, 2]
    acprop: float or list
        list of floats of the same size as `acmetric`, higher values ensure that the corresponding
        method is chosen more frequently in the ensemble generation, defaults to 1 / len(acmetric)

    Random Subspace Parameters
    --------------------------
    dhs: ndarray
        Array of the same length as `mvdata` containing the unique drill hole identifiers for each
        sample
    minvars: int
        Specify the minimum number of variables to use for random subspace
    minsearch: int
        Minimum number of data to consider in the knn search
    minremove: float
        Minimum proportion of samples to remove for random subspace
    maxremove: float
        Maximum proportion of samples to remove for random subspace

    .. codeauthor:: Ryan Martin - 28-01-2018
    """
    __doc__ = __doc__.format(BaseEnsemble.__doc__)

    def __init__(self, mvdata, locations, nreal=100, nnears=10, rseed=69069, minfound=0.001,
                 maxfound=0.999, searchparams=(0, 0, 0, 500, 500, 500), dhs=None, minvars=-1,
                 minsearch=10, cluster_method='kmeans', acmetric=[1, 2], acprop=None,
                 minremove=0, maxremove=0.15):
        super().__init__(mvdata, locations, nreal, nnears, rseed, minfound, maxfound, searchparams)

        minvars = mvdata.shape[1] if minvars == -1 else minvars
        if dhs is not None:
            assert len(dhs) == len(mvdata), 'ERROR: mismatch between dhs and other data'
        self.dhs = dhs

        # get the different cluster methods to randomize through
        if isinstance(cluster_method, str):
            cluster_method = [_CLUS_STR_INT[cluster_method]]
        elif isinstance(cluster_method, list):
            _cluster_method = []
            for cltype in cluster_method:
                if isinstance(cltype, int):
                    _cluster_method.append(cltype)
                else:
                    _cluster_method.append(_CLUS_STR_INT[cltype])
            cluster_method = _cluster_method
        else:
            cluster_method = [0]

        if isinstance(acmetric, str):
            acmetric = [_AUTOCORR_INT[acmetric]]
        elif isinstance(acmetric, list):
            _acmetric = []
            for acm in acmetric:
                if isinstance(acm, str):
                    _acmetric.append(_AUTOCORR_INT[acm])
                else:
                    _acmetric.append(acm)
            acmetric = _acmetric

        if acprop is None:
            nautocorr = len(acmetric)
            acprop = [1 / nautocorr for _ in range(nautocorr)]

        # update the main dictionary with the class-specific parameters
        acpars = dict(dhs=dhs, minvars=minvars, minsearch=minsearch, cluster_method=cluster_method,
                      acmetric=acmetric, acprop=acprop, minremove=minremove, maxremove=maxremove)
        self.pars.update(acpars)

        # information that is saved during ensemble construction
        self.pars["training_test_idx"] = {}
        self.pars["autocorrtype"] = np.zeros(nreal, dtype=np.int_)
        self.pars["clusterer"] = np.zeros(nreal).astype(object)
        self.pars["prop_remove"] = np.zeros(nreal, dtype=np.float64)

    def _gen_testtraining(self, udh=None, dhs=None):
        " generate the test, training and variable indexes, optionally removing full dhs at a time "
        # required pars
        maxnnears = self.pars["nnears"]
        mnpr, mxpr = self.pars["minremove"], self.pars["maxremove"]
        mnvars = self.pars["minvars"]
        nvar = self.mvdata.shape[1]

        if maxnnears is not None:
            nnears = max(self.pars["minsearch"], int(
                (0.5 + 0.5 * np.random.rand()) * maxnnears))
        else:
            nnears = None
        if udh is None:
            ndata = self.mvdata.shape[0]
        else:
            ndata = len(udh)
        # permute and take the ranomd sample
        prop_remove = np.random.rand() * (mxpr - mnpr) + mnpr
        ndata_include = int((1 - prop_remove) * ndata)
        perm = np.random.permutation(ndata)
        # the 2d idxs
        training_idx = perm[:ndata_include]
        test_idx = perm[ndata_include:]
        vidx = np.random.permutation(np.random.randint(mnvars, nvar + 1))[:nvar]
        if udh is not None:
            # modify test & training to remove complete ddhs
            dhrange = np.arange(dhs.shape[0])
            training_dhs = udh[training_idx]
            training_idx = []
            for tid in training_dhs:
                training_idx.extend(dhrange[dhs == tid].tolist())
            training_idx = np.array(training_idx)
            test_dhs = udh[test_idx]
            test_idx = []
            for tid in test_dhs:
                test_idx.extend(dhrange[dhs == tid].tolist())
            test_idx = np.array(test_idx)
        return training_idx, test_idx, vidx, nnears

    def fit(self, target_nclus, verbose=True):
        # parse the rest of the things from self.pars
        parlist = ["nreal", "nnears", "cluster_method", "rseed", "acmetric", "acprop",
                   "searchparams", "minfound", "maxfound"]
        nreal, nnears, cluster_method, rseed, \
            acmetric, acprop, searchparams, \
            minfound, maxfound = [self.pars[p] for p in parlist]

        # simple checks
        if nnears is None:
            nnears = 25
        # setup the random generator and make a `grnd` function
        np.random.seed(rseed)
        grnd = np.random.rand

        # coerce things into arrays
        acmetric = np.array(acmetric)
        acprop = np.array(acprop) / np.sum(acprop)
        acprop = np.cumsum(acprop)

        if self.dhs is None:
            udhs, dhids = None, None
        else:
            udhs, dhids = np.unique(self.dhs), self.dhs

        # setup the arrays and dictionary of storage
        ndata, nvar = self.mvdata.shape
        clusterings = np.zeros((ndata, nreal), dtype=np.int_)

        cats = [i + 1 for i in range(target_nclus)]
        locations = self.locations
        mvdata = self.mvdata
        # the rotated locations
        rot_tree, rot_xyz = build_anisotree(searchparams, locations)
        _, allsearch_idx = rot_tree.query(rot_xyz, k=nnears)

        # calculate the autocorrelation dataset(s)
        autocorrdatasets = {0: fstat.columnwise_nscore(mvdata.copy())}
        for metric in acmetric:
            z_autocorr = fstat.columnwise_nscore(_local_autocorr(metric, mvdata, allsearch_idx,
                                                                 rot_xyz))
            autocorrdatasets[metric] = z_autocorr
        ireal = 0
        irealold = -1
        nfound = 0
        nskip = 0
        irepo = nreal / 10
        while nfound < nreal:
            if verbose and ireal != irealold and ireal % irepo == 0:
                print("Working on %i, skipped %i" % (ireal, nskip))
                irealold = ireal

            # determine which autocorrelation metric to use, calculate the autocorrelation
            training_idx, test_idx, vidx, nnears = self._gen_testtraining(udhs, dhids)

            metric = acmetric[acprop.searchsorted(grnd())]
            metric = np.random.permutation(acmetric)[0]
            mvdata = autocorrdatasets[metric]

            # get the test and training datasets
            z_training = mvdata[training_idx, :]
            z_training = z_training[:, vidx]
            trainlocs = rot_xyz[training_idx, :]
            z_test = mvdata[test_idx, :]
            z_test = z_test[:, vidx]
            testlocs = rot_xyz[test_idx, :]

            if len(cluster_method) == 1:  # if 1 is passed, use it, else randomly choose between them
                clusteridx = cluster_method[0]
            else:
                clusteridx = cluster_method[np.random.randint(len(cluster_method))]
            clusterer = _CLUSFUNCS[clusteridx]
            # cluster the autocorrelated data
            training_clusdef, sk_cluster_obj = clusterer(target_nclus, z_training)
            clusterings[training_idx, ireal] = training_clusdef

            # classify the test_data if necessary
            if z_test.size > 0:
                clusterings[test_idx, ireal] = cutils.predict_cluster_labels(
                    trainlocs, z_training, training_clusdef, testlocs, z_test)

            props = np.array([np.sum(clusterings[:, ireal] == cat) / ndata for cat in cats])
            if (props > minfound).all() and (props < maxfound).all():
                self.pars["training_test_idx"][ireal] = [training_idx.copy(), test_idx.copy()]
                self.pars["autocorrtype"][ireal] = metric
                self.pars["prop_remove"][ireal] = z_test.shape[0]
                self.pars["clusterer"][ireal] = _CLUS_INT_STR[clusteridx]
                ireal += 1
                nfound += 1
            else:
                nskip += 1
                if nskip > 1000:
                    raise RuntimeError("`minfound` is likely too small! ")
        print('Finished! Skipped %i clusterings from insufficient proportions' % nskip)
        self.autocorrdatasets = autocorrdatasets
        self.clusterings = clusterings
