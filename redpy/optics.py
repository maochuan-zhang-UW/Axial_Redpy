# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
OPTICS - Ordering the Points to Investigate the Clustering Structure.

This class handles the processing REDPy uses for determining 'core'
events and for ordering events based on their similarity.
"""
import numpy as np


# Based on https://github.com/espg/OPTICS
class OPTICS():
    """Object for holding data and processing for OPTICS ordering."""

    def __init__(self, distance_pairs):
        """
        Build and hold the data structure for OPTICS processing.

        Parameters
        ----------
        distance_pairs : float ndarray
            NxN distance matrix.

        """
        self.data = distance_pairs
        self._n = len(self.data)
        self._processed = np.zeros((self._n, 1), dtype=bool)
        self._reachability = np.ones(self._n) * np.inf
        self._core_dist = np.ones(self._n) * np.nan
        self._index = np.array(range(self._n))
        self._nneighbors = np.ones(self._n, dtype=int)*self._n
        self._cluster_id = -np.ones(self._n, dtype=int)
        self._is_core = np.ones(self._n, dtype=bool)
        self._ordered_list = []

    def prep_optics(self):
        """
        Prep data set for main OPTICS loop.

        Parameters
        ----------
        epsilon : float
            Determines maximum object size that can be extracted.
            Smaller epsilons reduce run time.

        """
        for j in self._index:
            # Find smallest nonzero distance
            self._core_dist[j] = np.sort(self.data[j, :])[1]

    def build_optics(self, epsilon):
        """
        Build OPTICS ordered list of clustering structure.

        Parameters
        ----------
        epsilon : float
            Determines maximum object size that can be extracted.
            Smaller epsilons reduce run time.

        """
        for point in self._index:
            if not self._processed[point]:
                self.expand_cluster_order(point, epsilon)

    def expand_cluster_order(self, point, epsilon):
        """
        Expand OPTICS ordered list of clustering structure.

        Parameters
        ----------
        point : int
            Index of event to process.
        epsilon : float
            Determines maximum object size that can be extracted.
            Smaller epsilons reduce run time.

        """
        if self._core_dist[point] <= epsilon:
            while not self._processed[point]:
                self._processed[point] = True
                self._ordered_list.append(point)
                point = self.set_reach_dist(point)
        else:
            self._processed[point] = True

    def set_reach_dist(self, point):
        """
        Set reachability distance and ordering.

        Parameters
        ----------
        point : int
            Index of event to process.

        Returns
        -------
        int, list int
            Index of event processed, or list of unprocessed points.

        """
        row = [self.data[point, :]]
        indices = np.argsort(row)
        distances = np.sort(row)
        if np.iterable(distances):
            unprocessed = indices[(self._processed[indices] < 1)[0].T]
            rdistances = np.maximum(distances[
                (self._processed[indices] < 1)[0].T],
                self._core_dist[point])
            self._reachability[unprocessed] = np.minimum(
                self._reachability[unprocessed], rdistances)
            if unprocessed.size > 0:
                return unprocessed[np.argsort(np.array(self._reachability[
                    unprocessed]))[0]]
            return point
        return point

    def run(self, epsilon):
        """
        Run OPTICS.

        Parameters
        ----------
        epsilon : float
            Determines maximum object size that can be extracted.
            Smaller epsilons reduce run time.

        Returns
        -------
        int ndarray
            Ordered list.
        int
            Index of core (minimum reachability).

        """
        self.prep_optics()
        self.build_optics(epsilon)
        return np.array(self._ordered_list), np.argmin(self._reachability)
