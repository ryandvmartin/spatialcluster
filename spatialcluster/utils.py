""" (c) Ryan Martin 2018 under MIT license """
import pandas as pd


def read_gslib(filename, filesize=None, chunksize=100000, engine="c"):
    """ read the geoeas formatted file """
    with open(filename, "r") as outfl:
        outfl.readline()
        nvar = int(outfl.readline().split()[0])
        columns = [outfl.readline().strip() for _ in range(nvar)]
    data = pd.read_csv(filename, delim_whitespace=True, chunksize=chunksize, nrows=filesize,
                       skiprows=nvar + 2, skipinitialspace=True, header=None, engine=engine)
    data = pd.concat(data, ignore_index=True)
    data.columns = columns
    return data


def write_gslib(data, filename, fmt="%.8g", chunksize=None):
    """ write the geoeas formatted file """
    header = "{}\n{}\n{}\n".format("saved data",
                                   len(data.columns),
                                   "\n".join(data.columns))
    with open(filename, "w") as outfl:
        outfl.write(header)
        data.to_csv(outfl, header=False, index=False, line_terminator="\n",
                    chunksize=chunksize, sep=" ", float_format=fmt)


def readparfile(parfile):
    """ read and return the parstring """
    print('Reading parfile from `{}`'.format(parfile))
    with open(parfile, 'r') as fh:
        parstr = fh.read()
    return parstr


def writeparfile(parstr, parfile):
    """ save the `parstr` to the `parfile` """
    print("Writing parfile to `{}`".format(parfile))
    with open(parfile, 'w') as fh:
        fh.write(parstr)


def parstr_sectionindexes(parstring, sectiontitles):
    """
    Return the indexes pointing to the first and last line of the parstring corresponding
    to the section indicated by `sectiontitles`
    """
    startidxs = []
    finidxs = []
    alllines = parstring.splitlines()
    if alllines[-1] != '':
        alllines.append('')
    if isinstance(sectiontitles, str):
        sectiontitles = [sectiontitles]
    # get the startidx for each section
    for sectitle in sectiontitles:
        for iline, line in enumerate(alllines):
            if sectitle in line:
                startidxs.append(iline + 1)
                for iline2 in range(iline + 1, len(alllines)):
                    thisline = alllines[iline2]
                    # if we get to an empty line or another section title
                    if thisline == '' or any(s in thisline for s in sectiontitles):
                        finidxs.append(iline2)
                        break
    return startidxs, finidxs


def rseed_list(n=100, seed=73021):
    import numpy as np
    np.random.seed(seed)
    step = 3
    return np.random.permutation(np.arange(101, n * step * 50, step))[:n].tolist()


def log_range(*rangeint, **kwargs):
    """ quick wrapper around log_progress when an integer is passed """
    from numpy import arange
    if isinstance(rangeint[0], (tuple, list)):
        rangeint = rangeint[0]
    for i in log_progress(arange(*rangeint), **kwargs):
        yield i


def log_enum(iterable, **kwargs):
    """ quick wrapper around logprogress and enumerate when an iterable collection is passed """
    for i, item in enumerate(log_progress(iterable, **kwargs)):
        yield (i, item)


def log_progress(sequence, size=None, name='Items'):
    """
    Return the status iterator
    """
    from tqdm import tqdm, tqdm_notebook
    try:
        get_ipython()
        return tqdm_notebook(sequence, desc=name, total=size)
    except:
        return tqdm(sequence, name, total=size, ascii=True)
