# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)

import numpy as np
import scipy


# Based on https://github.com/espg/OPTICS

class setOfObjects(object):

    def __init__(self, distance_pairs):
        """
        Builds and holds the data structure for OPTICS processing.

        Parameters
        ----------
        distance_pairs : float ndarray
            NxN distance matrix.

        """


        self.data = distance_pairs
        self._n = len(self.data)
        self._processed = scipy.zeros((self._n, 1), dtype=bool)
        self._reachability = scipy.ones(self._n) * scipy.inf
        self._core_dist = scipy.ones(self._n) * scipy.nan
        self._index = scipy.array(range(self._n))
        self._nneighbors = scipy.ones(self._n, dtype=int)*self._n
        self._cluster_id = -scipy.ones(self._n, dtype=int)
        self._is_core = scipy.ones(self._n, dtype=bool)
        self._ordered_list = []


def prep_optics(SetofObjects, epsilon):
    """
    Prep data set for main OPTICS loop.

    Parameters
    ----------
    SetofObjects : setOfObjects object
        Instantiated and prepped instance.
    epsilon : float
        Determines maximum object size that can be extracted. Smaller epsilons
        reduce run time.

    """

    for j in SetofObjects._index:
        # Find smallest nonzero distance
        SetofObjects._core_dist[j] = np.sort(SetofObjects.data[j,:])[1]


def build_optics(SetOfObjects, epsilon):
    """
    Builds OPTICS ordered list of clustering structure.

    Parameters
    ----------
    SetofObjects : setOfObjects object
        Instantiated and prepped instance.
    epsilon : float
        Determines maximum object size that can be extracted. Smaller epsilons
        reduce run time.

    """

    for point in SetOfObjects._index:
        if not SetOfObjects._processed[point]:
            expand_cluster_order(SetOfObjects, point, epsilon)


def expand_cluster_order(SetOfObjects, point, epsilon):
    """
    Expands OPTICS ordered list of clustering structure

    Parameters
    ----------
    SetofObjects : setOfObjects object
        Instantiated and prepped instance.
    point : int
        Index of event to process.
    epsilon : float
        Determines maximum object size that can be extracted. Smaller epsilons
        reduce run time.

    """

    if SetOfObjects._core_dist[point] <= epsilon:
        while not SetOfObjects._processed[point]:
            SetOfObjects._processed[point] = True
            SetOfObjects._ordered_list.append(point)
            point = set_reach_dist(SetOfObjects, point, epsilon)
    else:
        SetOfObjects._processed[point] = True


def set_reach_dist(SetOfObjects, point, epsilon):
    """
    Sets reachability distance and ordering.

    Parameters
    ----------
    SetofObjects : setOfObjects object
        Instantiated and prepped instance.
    point : int
        Index of event to process.
    epsilon : float
        Determines maximum object size that can be extracted. Smaller epsilons
        reduce run time.

    Returns
    -------
    point : int, list int
        Index of event processed, or list of unprocessed points.

    """

    row = [SetOfObjects.data[point,:]]
    indices = np.argsort(row)
    distances = np.sort(row)

    if scipy.iterable(distances):

        unprocessed = indices[(SetOfObjects._processed[indices] < 1)[0].T]
        rdistances = scipy.maximum(distances[
            (SetOfObjects._processed[indices] < 1)[0].T],
            SetOfObjects._core_dist[point])
        SetOfObjects._reachability[unprocessed] = scipy.minimum(
            SetOfObjects._reachability[unprocessed], rdistances)

        if unprocessed.size > 0:
            return unprocessed[np.argsort(np.array(SetOfObjects._reachability[
                unprocessed]))[0]]
        else:
            return point
    else:
        return point
