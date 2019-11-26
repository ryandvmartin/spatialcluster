""" wrapper for MDS related functions """
import numpy as np


def mdsplot(data, mtype='MDS', ndim=2, rseed=69069, c='k', cmap=None, catdata=False,
            pltstyle=None, ax=None, cax=None, figsize=None, s=15, lw=0.1, cbar=None, grid=None,
            legend_loc='lower right', title=None, vlim=None, legstr='Cluster', xlabel=None,
            ylabel=None):
    """
    MDS Plotting function, embeds the given dataset in ndim-dimensional space, then plots. Really
    only makes sense for 2D. Consider initializing an `MDSPlot` object if repeatedly plotting the
    projected values with different colors, etc.

    Parameters
    ----------
    data : ndarray or dataframe
        Variables as columns, with observations as rows. Assumes homotopic sampling
    """
    if hasattr(data, 'plot'):
        mdsplotter = data
    else:
        mdsplotter = MDSPlot(data, mtype, ndim, rseed)
        mdsplotter.embed()
    return mdsplotter.plot(c, cmap, catdata, pltstyle, ax, cax, figsize, s, lw, cbar, grid,
                           legend_loc, title, vlim, legstr, xlabel, ylabel)


def get_catcmap(continuous, N, offset=0.5):
    """ get the `N` categorical colors from `continuous` cmap """
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    cm = plt.cm.get_cmap(continuous)
    colors = [cm((i + offset) / N) for i in range(N)]
    return mpl.colors.ListedColormap(colors, name=continuous)


def cmap_handling(cmap, ncats=None, catdict=None):
    """
    Parameters
    ----------
    cmap : None, string, dictionary
        defaults to `viridis`
    ncats : int
        the number of categories, if `None` then continuous cmap is assumed
    catdict : dict
        Contains the mapping {k: name} where k is the integer code and name is the name of
        the category
    """
    import matplotlib as mpl
    if cmap is None:
        cmap = "Viridis"
    if ncats is not None:
        if isinstance(cmap, str):
            try:
                colors = get_catcmap(cmap.lower(), ncats)
            except:
                colors = get_catcmap(cmap, ncats)
            cmap = [mpl.colors.rgb2hex(c) for c in colors.colors]
        elif isinstance(cmap, dict):
            if catdict is None:
                cmap = [cmap[k] for k in cmap][::-1]
            else:
                try:
                    cmap = [cmap[k] for k in catdict]
                except KeyError:
                    cmap = [cmap[k] for k in catdict.values()]
        elif hasattr(cmap, "colors"):
            cmap = [mpl.colors.rgb2hex(c) for c in cmap.colors]
    return cmap


class MDSPlot:
    """
    Wrapper for MDS plotting for a given set of multivariate data, passed on init
    Parameters:
        data (dataframe or np.ndarray): the data to embed with MDS
        mtype (str): the embedding type, valid are 'umap', 'mds' and 'tsne'
        ndim (int): the number of dimensions to embed the coordinates in
        rseed (int): the random state for embedding
    """

    def __init__(self, data, mtype='umap', ndim=2, rseed=69069, verbose=False):
        # make sure data is standardized
        from .statfuncs import columnwise_standardize
        if hasattr(data, 'values'):
            data = data.values
        self.data = columnwise_standardize(data)
        self.mtype = mtype
        self.ndim = ndim
        self.rseed = rseed
        self.verbose = verbose
        self.coords = None

    def embed(self, **kwargs):
        """
        Quick wrapper around some sklearn manifold techniques

        Parameters:
            data (dataframe or np.ndarray):
        """
        from sklearn import manifold
        data = self.data
        np.random.seed(self.rseed)
        if self.mtype.lower() == 'mds':
            model = manifold.MDS(n_components=self.ndim, random_state=self.rseed,
                                 verbose=self.verbose, **kwargs)
        elif self.mtype.lower() == 'tsne':
            model = manifold.t_sne.TSNE(n_components=self.ndim, random_state=self.rseed,
                                        verbose=self.verbose, **kwargs)
        elif self.mtype.lower() == 'umap':
            import umap
            model = umap.UMAP(n_components=self.ndim, **kwargs)
        elif self.mtype.lower() == 'isomap':
            import umap
            model = manifold.Isomap(n_components=self.ndim, **kwargs)
        nrow, ncol = data.shape
        if ncol > nrow:
            data = data.T
        model = model.fit(data)
        self.coords = model.embedding_  # the N-dimensional embedding of the data
        return self

    def plot(self, c='k', cmap=None, catdata=None, ax=None, cax=None, figsize=None, s=15, lw=0.1,
             cbar=None, grid=True, legend_loc='lower right', title=None, vlim=None,
             legstr='Cluster', xlabel=None, ylabel=None):
        """
        Parameters
        ----------
        c : str, ndarray
            a single color or a ndata-long array of colors
        cmap : str, dict
            Either a mpl compatible cmap string, or if `catdata` a dictionary of {k: color} mapping
            each category to a specific color
        catdata : bool, dict
            If a dictionary is passed, the mapping {k: name} is expected
        """
        colors = c
        try:
            import pygeostat as gs
        except:
            raise ImportError("ERROR: this function requires pygeostat!")
        coords = self.coords
        # setup the figure
        fig, ax, cax = gs.setup_plot(ax, cbar=cbar, cax=cax, figsize=figsize)
        if vlim is None:
            if colors is not None and not isinstance(colors, str):
                # if vlim is `None` get the 95 percentile as the max
                vlim = (colors.min(), gs.cdf(colors, bins=100)[0][95])
            else:
                vlim = (None, None)
        # deal with non-array input
        if hasattr(colors, 'values'):
            colors = colors.values
        if catdata is None and not isinstance(colors, str) and len(np.unique(colors)) <= 12:
            catdata = True
        # plot categories
        if catdata:
            if isinstance(catdata, dict):
                catdict = catdata
            else:
                catdict = None
            ucolors = np.unique(colors)
            ncat = len(ucolors)
            cmap = cmap_handling(cmap, ncat, catdict)
            for i in range(ncat):
                thiscolor = cmap[i]
                if catdict is None:
                    label = '{} {}'.format(legstr, ucolors[i])
                else:
                    label = catdict[ucolors[i]]
                idx = colors == ucolors[i]
                ax.scatter(coords[idx, 0], coords[idx, 1], c=thiscolor, s=s, lw=lw,
                           label=label, zorder=10)
            if isinstance(legend_loc, str):
                ax.legend(loc=legend_loc, scatterpoints=1, handletextpad=0.05)
            elif isinstance(legend_loc, tuple):
                ax.legend(loc='upper left', bbox_to_anchor=legend_loc, scatterpoints=1,
                          handletextpad=0.05)
        # plot continous data with a colorbar
        else:
            plot = ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=s, lw=lw, cmap=cmap,
                              vmin=vlim[0], vmax=vlim[1], zorder=10)
            if cbar:
                vlim, ticklocs, ticklabels = gs.get_contcbarargs(colors, 2, vlim, nticks=8)
                cbar = fig.colorbar(plot, cax=cax, ticks=ticklocs)
                cbar.ax.set_yticklabels(ticklabels, ha='left')
        ax.grid(grid)
        if ylabel is None:
            ax.set_ylabel('$MDS_2$')
        else:
            ax.set_ylabel(ylabel)
        if xlabel is None:
            ax.set_xlabel('$MDS_1$')
        else:
            ax.set_xlabel(xlabel)
        if title:
            ax.set_title(title)
        return ax


def elbowplt(dataframe, variables, ninit=10, maxclust=20, ax=None, gmm=False, figsize=None,
             **plotkws):
    """
    For a number of cluster numbers, fit the clustering, calculate the WCSS, and generate a plot
    showing the relationship over the range of K

    Parameters
    ----------
    dataframe : DataFrame-like
        Data container to be indexed by the variables
    variables : str or list
        List of variables comprising the multivariate dataset
    ninit : int
        number of random initializations of the clustering algorithm
    maxclust : int
        The range [2, maxclust] are evaluated
    ax : mpl.Axis
        A plotting axis to provide
    **plotkws
        Keywords passed on to plt.plot
    """
    import matplotlib.pyplot as plt
    from .cluster_utils import cluster
    from .clustermetrics import tdiff_wcss, tdiff_mwcss

    # get the transposed array of values for clustering
    values = dataframe[variables].values

    clusnums = np.arange(2, maxclust + 1)
    wcss_km_m = np.zeros(len(clusnums))
    wcss_gmm_m = np.zeros(len(clusnums))
    for i, nclus in enumerate(clusnums):
        if gmm:
            nav_gmm = 0
        nav_km = 0
        cats = np.arange(nclus) + 1
        for init in range(ninit):
            if gmm:
                gmmclus = cluster(nclus, values, n_init=1, method="gmm")[0] + 1
                wcss_gmm_m[i] += tdiff_mwcss(cats, gmmclus, values)
                nav_gmm += 1

            # kmeans
            kmclus = cluster(nclus, values, n_init=1, method="kmeans")[0] + 1
            wcss_km_m[i] += tdiff_wcss(cats, kmclus, values)
            nav_km += 1

        # finalize
        if gmm:
            wcss_gmm_m[i] /= nav_gmm
        wcss_km_m[i] /= nav_km

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.plot(clusnums, wcss_km_m, c='blue', **plotkws, label='WCSS')
    ax.set_ylabel('WCSS')
    if gmm:
        ax2 = ax.twinx()
        ax2.plot(clusnums, wcss_gmm_m, c='red', **plotkws, label='GMM')
        ax.plot(np.nan, c='red', **plotkws, label='GMM')
        # ax.set_ybound(0)
        ax2.set_ylabel('M-WCSS')
        ax2.grid(False)
    ax.set_xlabel('Number of Clusters, `K`')
    ax.grid(False)
    ax.legend()
    ax.xaxis.set_ticks(clusnums.astype(int))
    ax.xaxis.set_ticklabels(clusnums.astype(int))
    ax.set_xbound(2)
    return ax
