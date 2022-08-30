"""
Module that loads parts of the SubLink merger trees as required.

"""

import numpy as np

import simulation_box
import locate_object

def load_single_subhalo(subhaloid, fields, sim):
    """ Loads specified columns in the SubLink catalog for a given subhalo.

    Parameters
    ----------
    ... : type
        ....

    Returns
    -------
    ... : float
        ...

    """

    rownum, chunknum = locate_object.row_in_chunk(subhaloid, sim)
    catkey = 'SubLink' + str(chunknum)
    sim.load_data('SubLink', chunknum, fields)

    subhalo = {}
    subhalo['Number'] = 1
    subhalo['ChunkNumber'] = chunknum
    subhalo['IndexInChunk'] = rownum
    for field in fields:
        subhalo[field] = sim.preloaded[catkey][field][rownum]

    return subhalo


def load_group_subhalos(subhaloid, fields, sim, numlimit=0):
    """ Loads specified columns in the SubLink catalog for all subhalos in
    the same FOF group as the given subhalo.

    Parameters
    ----------
    ... : type
        ....

    numlimit : int, optional
        The number of subhalos to load, ordered by the MassHistory column.
        Default is 0, in which case all subhalos in the group are loaded.

    Returns
    -------
    ... : float
        ordered by the MassHistory column.

    """

    rownum, chunknum = locate_object.row_in_chunk(subhaloid, sim)
    catkey = 'SubLink' + str(chunknum)
    fields_ = list(set(fields).union(set(['SubhaloGrNr',
                                          'MassHistory',
                                          'SnapNum'])))
    sim.load_data('SubLink', chunknum, fields_)

    grnr = sim.preloaded[catkey]['SubhaloGrNr'][rownum]
    snap = sim.preloaded[catkey]['SnapNum'][rownum]
    mask = np.logical_and(sim.preloaded[catkey]['SubhaloGrNr'] == grnr,
                          sim.preloaded[catkey]['SnapNum'] == snap)
    idx = np.arange(len(sim.preloaded[catkey]['SnapNum']))[mask]
    numtot = len(idx)
    order = np.argsort(sim.preloaded[catkey]['MassHistory'][mask])[::-1]
    if numlimit and numlimit < numtot:
        order = order[: numlimit + 1]
        numtot = numlimit

    groupsubs = {}
    groupsubs['Number'] = numtot
    groupsubs['ChunkNumber'] = chunknum
    groupsubs['IndexInChunk'] = idx[order]
    for field in fields:
        groupsubs[field] = sim.preloaded[catkey][field][mask][order]
        
    return groupsubs


def load_single_tree(subhaloid, fields, sim,
                     main_branch_only=False,
                     include_progenitors=True,
                     include_descendants=False):
    """ Loads specified columns in the SubLink catalog for the subhalo-
    based merger tree, including progenitors and/or descendants of the
    given subhalo.

    Parameters
    ----------
    ... : type
        ....

    Returns
    -------
    ... : float
        ...

    """

    # no progenitors and no descendants
    if not include_descendants and not include_progenitors:
        raise ValueError('If only the subhalo itself is needed, use the ' +
                         '`load_sublink.load_single_subhalo` function.')

    fields_ = list(set(fields).union(set(['RootDescendantID',
                                          'DescendantID',
                                          'MainLeafProgenitorID',
                                          'FirstProgenitorID',
                                          'NextProgenitorID',
                                          'LastProgenitorID'])))
    subhalo = load_single_subhalo(subhaloid, fields_, sim)
    rownum, chunknum = locate_object.row_in_chunk(subhaloid, sim)
    catkey = 'SubLink' + str(chunknum)

    # progenitors only
    if not include_descendants:
        start = rownum
        if main_branch_only:
            end = rownum + (subhalo['MainLeafProgenitorID'] -
                            subhalo['SubhaloID'])
        else:
            end = rownum + (subhalo['LastProgenitorID'] -
                            subhalo['SubhaloID'])
        tree = {}
        for field in fields:
            tree[field] = sim.preloaded[catkey][field][start : end + 1]
        return tree

    # all descendants
    if not main_branch_only and not include_progenitors:
        desc = subhalo['DescendantID']
        idx = [rownum]
        while desc != -1:


def load_group_tree():
    """ Loads specified columns in the SubLink catalog for the subhalo-
    based merger tree, including progenitors and/or descendants of all
    subhalos in the same FOF group as the given subhalo..

    Parameters
    ----------
    ... : type
        ....

    Returns
    -------
    ... : float
        ...

    """
    pass
