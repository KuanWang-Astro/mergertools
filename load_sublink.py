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
    subhalo = {}
    sim.load_data('SubLink', chunknum, fields)
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
    fields_ = list(set(fields).union(set(['SubhaloGrNr', 'MassHistory'])))
    sim.load_data('SubLink', chunknum, fields_)
    grnr = sim.preloaded[catkey]['SubhaloGrNr'][rownum]
    mask = sim.preloaded[catkey]['SubhaloGrNr'] == grnr

    groupsubs = {}
    for field in fields:
        order = np.argsort(sim.preloaded[catkey]['MassHistory'][mask])[::-1]
        groupsubs[field] = sim.preloaded[catkey][field][mask][order]
        if numlimit:
            groupsubs[field] = groupsubs[field][:numlimit]

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
    chunknum = locate_object.chunk_num(subhaloid)[0]
    catkey = 'SubLink' + str(chunknum)

    # mb progenitors
    if main_branch_only and not include_descendants:
        start = rownum
        end = rownum + subhalo['MainLeafProgenitorID']\
                     - subhalo['SubhaloID']
        tree = {}
        for field in fields:
            tree[field] = sim.preloaded[catkey][field][start : end + 1]
        return tree

    # all descendants
    #if not main_branch_only and not include_progenitors:
    #    desc =
    #    while

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
