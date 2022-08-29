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

    Notes
    -----
    ....

    """

    rownum, chunknum = locate_object.row_in_chunk(subhaloid, sim)
    subhalo = {}
    sim.load_data('SubLink', chunknum, fields)
    for field in fields:
        subhalo[field] = sim.preloaded['SubLink' + str(chunknum)]\
                                      [field][rownum]

    return subhalo


def load_group_subhalos(subhaloid, fields, sim):
    """ Loads specified columns in the SubLink catalog for all subhalos in
    the same FOF group as the given subhalo.

    Parameters
    ----------
    ... : type
        ....

    Returns
    -------
    ... : float
        ...

    Notes
    -----
    ....

    """

    rownum, chunknum = locate_object.row_in_chunk(subhaloid, sim)

    fields_ = list(set(fields).union(['SubhaloGrNr']))
    sim.load_data('SubLink', chunknum, fields_)
    grnr = sim.preloaded['SubLink' + str(chunknum)]['SubhaloGrNr'][rownum]
    mask = sim.preloaded['SubLink' + str(chunknum)]['SubhaloGrNr'] == grnr

    groupsubs = {}
    for field in fields:
        groupsubs[field] = sim.preloaded['SubLink' + str(chunknum)]\
                                        [field][mask]

    return groupsubs


def load_single_tree():
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

    Notes
    -----
    ....

    """
    pass

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

    Notes
    -----
    ....

    """
    pass
