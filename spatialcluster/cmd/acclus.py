""" A GSLIB-style command-line callable script implementing DSSEnseble clustering """
import os
import sys
import warnings
from argparse import ArgumentParser

import spatialcluster as sp
from spatialcluster.utils import (parstr_sectionindexes, readparfile, writeparfile,
                                  write_gslib, read_gslib)

warnings.simplefilter(action='ignore', category=FutureWarning)


defaultparstr = """\t\t  ACCluster
\t\t  ----------

START OF DATA:
datafile.dat        - file with the input dataset
1  2  0             - columns for dh, x, y, z data
3  4  5  6          - columns for variables (implicit nvar)
outfile.out         - file for clustering output
1                   - append output to input file? (0=No, 1=Yes)

START OF AC:        # autocorrelation settings
25                  - number of nearest neighbors
0  0  0             - search anisotropy/ ang1, ang2, ang3
500 500 500         - range1, range2, range3
kmeans              - cluster method: one of `kmeans`, `gmm`, `hier`
1                   - autocorrelation metrics (0=None, 1=Morans, 2=Getis)
5                   - number of clusters

"""


def parsepars(parstring):
    """ parse parameters specific to the ACEnsemble parstring """
    import numpy as np
    pars = {}
    sectiontitles = ["START OF DATA:", "START OF AC:"]
    startidxs, finidxs = parstr_sectionindexes(parstring, sectiontitles)
    alllines = parstring.splitlines()
    # parse the data section --------------------------------------------------------
    lines = alllines[startidxs[0]:finidxs[0]]
    # data
    pars['datafile'] = lines[0].split()[0]
    try:
        data = read_gslib(pars['datafile'])
    except:
        print("ERROR: `{}` must be geoeas-formatted!".format(pars['datafile']))
        raise
    cols = data.columns
    # locations
    ifx, ify, ifz = [int(i) for i in lines[1].split()[:3]]
    assert all(c > 0 for c in [ifx, ify]), "ERROR: ifx and ify cols must be > zero"
    pars['locations'] = data[[cols[ifx - 1], cols[ify - 1]]].values
    if ifz > 0:
        pars['locations'] = np.c_[pars['locations'], data[cols[ifz - 1]].values]
    # mvdata
    varcols = [int(c) for c in lines[2][:lines[2].rfind('-')].split()]
    pars['mvdata'] = data[[cols[vc - 1] for vc in varcols]].values
    # output options
    pars['outfile'] = lines[3].split()[0]
    pars['appendout'] = bool(int(lines[4].split()[0]))
    # parse the ac section --------------------------------------------------------
    lines = alllines[startidxs[1]:finidxs[1]]
    # search stuff
    pars['nnears'] = int(lines[0].split()[0])
    angs = [float(a) for a in lines[1].split()[:3]]
    ranges = [float(r) for r in lines[2].split()[:3]]
    pars['searchparams'] = tuple(angs + ranges)
    # cluster stuff, metrics and proportions
    pars['cluster_method'] = lines[3].split()[0]
    pars['acmetric'] = sp.accluster._acmetric_MAPPING_R[int(lines[4].split()[0])]
    pars['nclus'] = int(lines[5].split()[0])
    return pars


def main():
    # some defaults to make parts of this generic
    cluster_object = sp.ACCluster
    thisname = cluster_object.__name__
    shortname = thisname[:3].lower()
    # setup the argparser and check for no args
    parser = ArgumentParser(description="Runner for {}".format(thisname))
    parser.add_argument('parfile', type=str, nargs='?',
                        help="the parfile to call `{}` with".format(thisname))
    args = parser.parse_args()
    if args.parfile is None:  # if nothing passed, print the default parfile
        writeparfile(defaultparstr, '{}.par'.format(thisname.lower()))
        sys.exit(0)
    else:
        assert os.path.isfile(args.parfile), "ERROR: {} does not exist!".format(args.parfile)
    # parse the parameter file
    pars = parsepars(readparfile(args.parfile))
    # collect pars that are not sent to the cluster object init
    datafile = pars.pop('datafile')
    outfile = pars.pop('outfile')
    appendout = pars.pop('appendout')
    # setup the output datafile
    if appendout:
        data = read_gslib(datafile)
    else:
        from pandas import DataFrame
        data = DataFrame()
    # generate the clustering object
    model = cluster_object(**pars)
    model.fit()
    labels = model.predict()
    data['{}_clusters'.format(shortname)] = labels
    # write the output
    write_gslib(data, outfile)


if __name__ == "__main__":
    main()
