# REDPy - Repeating Earthquake Detector in Python
# Copyright (C) 2016-2020  Alicia Hotovec-Ellis (ahotovec-ellis@usgs.gov)
# Licensed under GNU GPLv3 (see LICENSE.txt)

import sys

import numpy as np
from tables import *
from scipy.sparse import coo_matrix

from redpy.optics import *


def update_family(rtable, ctable, ftable, fnum, config, merge=1):
    """
    Decides whether to run OPTICS and then updates the Families table.

    OPTICS is (currently) extremely expensive for large families, so we update
    the core less frequently as the family grows. The merge ratio allows
    larger families to have their cores updated if less than
    config.get('merge_percent') of the total new family length is contained in a
    single family.

    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    ctable : Table object
        Handle to the Correlation table.
    ftable : Table object
        Handle to the Families table.
    fnum : int
        Family number.
    config : Config object
        Describes the run parameters.
    merge : float, optional
        Ratio of the largest family to the total new family length from merge.

    """

    # Update longest family
    famlen = len(ftable[fnum]['members'])
    if famlen > ftable.attrs.current_max_famlen:
        ftable.attrs.current_max_famlen = famlen

    fam = np.fromstring(ftable[fnum]['members'], dtype=int, sep=' ')

    if (len(fam) in (3, 4, 5, 6, 10, 15, 25, 50, 100, 250, 500, 1000, 2500,
        5000, 10000, 25000, 50000, 100000, 250000, 500000)) or (
        merge <= config.get('merge_percent')):

        run_optics(rtable, ctable, ftable, fnum, fam, config)

    update_ftable(rtable, ftable, fnum, fam, config)


def run_optics(rtable, ctable, ftable, fnum, fam, config):
    """
    Sets up distance matrix and runs OPTICS ordering to determine core event.

    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    ctable : Table object
        Handle to the Correlation table.
    ftable : Table object
        Handle to the Families table.
    fnum : int
        Family to run OPTICS on.
    fam : int ndarray
        List of family member locations in rtable.
    config : Config object
        Describes the run parameters.

    """

    # Get ids
    ids = rtable[fam]['id']

    # Get data from ctable
    id1 = ctable.cols.id1[:]
    ix = np.where(np.in1d(id1,ids))
    id1 = id1[ix]
    id2 = ctable[ix]['id2']
    ccc = ctable[ix]['ccc']
    maxid = np.max((np.max(ids),np.max(id2)))+1

    # Ensure no duplicates
    rc = np.vstack([id1,id2]).T.copy()
    dt = rc.dtype.descr * 2
    i = np.unique(rc.view(dt), return_index=True)[1]

    # Create sparse correlation matrix
    ccc_sparse = coo_matrix((ccc[i], (id1[i],id2[i])),
                             shape=(maxid,maxid)).tocsr()
    ccc_sparse = ccc_sparse[ids,:]
    ccc_sparse = ccc_sparse[:,ids]
    ccc_sparse += ccc_sparse.transpose()

    # Sort so most connected event is always considered for core
    s = np.argsort(np.squeeze(np.asarray(ccc_sparse.sum(axis=0))))
    ccc_sparse = ccc_sparse[s,:]
    ccc_sparse = ccc_sparse[:,s]
    fam = fam[s]

    # Create dense distance matrix
    D = np.ones(len(ids)) - ccc_sparse
    D = np.squeeze(np.asarray(D))
    D[range(len(ids)),range(len(ids))] = 0

    # Run OPTICS
    ttree = setOfObjects(D)
    prep_optics(ttree,1)
    build_optics(ttree,1)
    order = np.array(ttree._ordered_list)
    core = fam[np.argmin(ttree._reachability)]

    # Write to ftable
    np.set_printoptions(threshold=sys.maxsize)
    np.set_printoptions(linewidth=sys.maxsize)
    ftable.cols.members[fnum] = np.array2string(fam[order])[1:-1]
    ftable.cols.core[fnum] = core


def update_ftable(rtable, ftable, fnum, fam, config):
    """
    Updates the Families table after a new member has been added.

    Primarily in charge of keeping track of times and printing.

    Parameters
    ----------
    rtable : Table object
        Handle to the Repeaters table.
    ftable : Table object
        Handle to the Families table.
    fnum : int
        Family number to update.
    fam : int ndarray
        List of family member locations in rtable.
    config : Config object
        Describes the run parameters.

    """

    # !!! This function should not need to call from rtable (scales poorly)

    startTimes = rtable[fam]['startTimeMPL']

    ftable.cols.startTime[fnum] = np.min(startTimes)
    ftable.cols.longevity[fnum] = np.max(startTimes) - np.min(startTimes)
    ftable.cols.printme[fnum] = 1
    ftable.cols.printme[-1] = 1
    ftable.flush()

