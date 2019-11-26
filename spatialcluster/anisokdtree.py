""" (c) Ryan Martin 2018 under GPLv3 license """

import numpy as np
from scipy.spatial import cKDTree
from . statfuncs import setrot


def build_anisotree(anisotropy, locations, searchlocs=None):
    """
    Rotates locations and constructs an anisotropic kdtree in the rotated space.

    Parameters
    ----------
    anisotropy : list, tuple
        should be a 6-long iterable of (ang1, ang2, ang3, r1, r2, r3)
    locations : (nata, dim) ndarray
        array of locations to consider
    searchlocs : (nsearch, dim) ndarray
        array of locations for subsequent searching, e.g., the rotated searchlocs are returned

    Returns
    -------
    anisotree : cKDTree
        Anisotropic tree constructed on the rotated points
    rotlocs : ndarray
        Rotated locations for further queries

    .. codeauthor:: Ryan Martin - 03-07-2018
    """
    assert len(anisotropy) == 6, 'ERROR: pass (a1, a2, a3, r1, r2, 3) for `anisotropy`'
    if hasattr(locations, 'values'):
        locations = locations.values
    if hasattr(searchlocs, 'values'):
        searchlocs = searchlocs.values
    dim = locations.shape[1]
    rotmat = setrot(*anisotropy)
    rotlocs = np.dot(rotmat[:dim, :dim], locations.T).T
    rottree = cKDTree(rotlocs)
    if searchlocs is None:
        return rottree, rotlocs
    return rottree, np.dot(rotmat[:dim, :dim], searchlocs.T).T


def query_anisotree(anisotropy, locations, knn=25):
    """
    Automates the build and query of an anisotropic tree for the same locations as passed

    Parameters
    ----------
    anisotropy : list, tuple
        should be a 6-long iterable of (ang1, ang2, ang3, r1, r2, r3)
    locations : (nata, dim) ndarray
        array of locations to consider
    knn : int
        The number of neighbors to search

    Returns
    -------
    idxs : ndarray
        The indexes corresponding to the `knn` neighbors around each location

    .. codeauthor:: Ryan Martin - 03-07-2018
    """
    anisotree, rotatedlocs = build_anisotree(anisotropy, locations)
    return anisotree.query(rotatedlocs, knn)[1]
