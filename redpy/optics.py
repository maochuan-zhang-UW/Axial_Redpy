# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)
"""
OPTICS - Ordering the Points to Investigate the Clustering Structure.

This class handles the processing REDPy uses for determining 'core'
events and for ordering events based on their similarity.
"""
import warnings

import numpy as np
import sklearn.neighbors
from sklearn.cluster import OPTICS

import redpy.correlation


def prep_distance_matrix(detector, members):
    """
    Prepare the sparse distance matrix for OPTICS for a family.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    members : int ndarray
        List of rows of family members in Repeaters table.

    Returns
    -------
    float csr_matrix
        Sparse distance matrix.
    int ndarray
        Reordered list of family members, sorted by sum of rows in
        correlation matrix.

    """
    ccc_sparse = redpy.correlation.subset_matrix(
        detector.get('rtable', 'id', members), detector.get_matrix()[1],
        'sparse')
    order = np.argsort(np.squeeze(np.asarray(ccc_sparse.sum(axis=0))))
    members = members[order]
    ccc_sparse = ccc_sparse[order, :]
    ccc_sparse = ccc_sparse[:, order]
    ccc_sparse.data -= 1
    ccc_sparse = sklearn.neighbors.sort_graph_by_row_values(
            np.abs(ccc_sparse), warn_when_not_sorted=False)
    return ccc_sparse, members


def run_optics(detector, members, distance_matrix=None):
    """
    Run the OPTICS algorithm.

    Can use either sparse or dense distance matrices. Note that this
    function also returns a sorted version of members, which will
    correspond to the ordering of the cluster object's attributes.

    Parameters
    ----------
    detector : Detector object
        Primary interface for handling detections.
    members : int ndarray
        List of rows of family members in Repeaters table.
    distance_matrix : float csr_matrix or float ndarray, optional
        Distance matrix (1-ccc).

    Returns
    -------
    sklearn.cluster.OPTICS object
        Fitted clustering object.
    int ndarray
        Reordered list of family members, sorted by sum of rows in
        correlation matrix.

    """
    if distance_matrix is None:
        distance_matrix, members = prep_distance_matrix(detector, members)
    with warnings.catch_warnings():
        # Every once in a while, OPTICS() throws an EfficiencyWarning
        warnings.simplefilter("ignore")
        return OPTICS(metric='precomputed', min_samples=0, max_eps=1).fit(
            distance_matrix), members
