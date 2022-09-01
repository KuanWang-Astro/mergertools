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
    subhaloid :  int
      The SubhaloID of the subhalo to load. SubhaloID is the SubLink
      index and is unique throughout all snapshots.

    fields : list of str
      The columns to load from the table.

    sim : class obj
      Instance of the simulation_box.SimulationBox class, which specifies
      the simulation box to work with.

    Returns
    -------
    subhalo : dict
      Dictionary containing the specified fields for the given subhalo.
      Entries are stored as single-element arrays. Also includes the number
      of subhalos (1 in this case), the SubLink chunk number in which the
      subhalo is stored, and the row index of the given subhalo in the chunk.

    """

    rownum, chunknum = locate_object.row_in_chunk(subhaloid, sim)
    catkey = 'SubLink' + str(chunknum)
    sim.load_data('SubLink', chunknum, fields)

    subhalo = {}
    subhalo['Number'] = 1
    subhalo['ChunkNumber'] = chunknum
    subhalo['IndexInChunk'] = np.array([rownum])
    for field in fields:
        subhalo[field] = sim.preloaded[catkey][field][rownum : rownum + 1]

    return subhalo


def walk_tree(subhaloid, fields, sim, pointer, numlimit=0):
    """ Walks the tree following a given pointer (e.g., FirstProgenitorID,
    DescendantID, etc.) iteratively, starting from the given subhalo, with
    an optional number limit of subhalos to load.

    Parameters
    ----------
    subhaloid :  int or array_like
      The SubhaloID of the subhalo to start from. SubhaloID is the SubLink
      index and is unique throughout all snapshots.

    fields : list of str
      The columns to load from the table.

    sim : class obj
      Instance of the simulation_box.SimulationBox class, which specifies
      the simulation box to work with.

    pointer : str
      The name of the pointer to follow. Choose from the given set of pointer
      names in https://www.tng-project.org/data/docs/specifications/#sec4a.

    numlimit : int, optional
      The maximum number of subhalos to load along the pointer direction,
      including the starting point. Default is 0, in which case the function
      follows the pointer until there are no more subhalos ahead.


    Returns
    -------
    chain : dict
        Dictionary containing the specified fields for the chain of subhalos,
        ordered in the direction of the pointer. Entries are stored as numpy
        arrays. Also includes the number of subhalos, the SubLink chunk number
        in which the subhalos are stored, and the row indices of the loaded
        subhalos in the chunk.

    """

    iter_pointers = ['FirstProgenitorID', 'NextProgenitorID',
                     'DescendantID', 'NextSubhaloInFOFGroupID']
    noniter_pointers = ['LastProgenitorID', 'MainLeafProgenitorID',
                        'RootDescendantID', 'FirstSubhaloInFOFGroupID']

    if pointer not in iter_pointers + noniter_pointers:
        raise ValueError('Unknown pointer: {}. '.format(pointer) +
                         'Choose from {}.'.format(iter_pointers +
                                                  noniter_pointers))

    fields_ = list(set(fields).union(set([pointer])))
    subhalo = load_single_subhalo(subhaloid, fields_, sim)
    rownum, chunknum = locate_object.row_in_chunk(subhaloid, sim)
    catkey = 'SubLink' + str(chunknum)

    head = subhalo['SubhaloID']
    head_row = rownum
    idx = [head_row]

    if pointer in noniter_pointers:
        jump = sim.preloaded[catkey][pointer][head_row] - head
        if jump != 0:
            head_row += jump
            idx.append(head_row)
            head = sim.preloaded[catkey][pointer][head_row]

    else:
        while True:
            jump = sim.preloaded[catkey][pointer][head_row] - head
            head_row += jump
            if head_row < 0:
                break
            idx.append(head_row)
            if numlimit and len(idx) == numlimit:
                break
            head = sim.preloaded[catkey]['SubhaloID'][head_row]
    print(idx)
    chain = {}
    chain['Number'] = len(idx)
    chain['ChunkNumber'] = chunknum
    chain['IndexInChunk'] = np.array(idx)
    for field in fields:
        chain[field] = sim.preloaded[catkey][field][idx]
    return chain


def load_group_subhalos(subhaloid, fields, sim, numlimit=0):
    """ Loads specified columns in the SubLink catalog for all subhalos in
    the same FOF group as the given subhalo.

    Parameters
    ----------
    subhaloid :  int or array_like
      The SubhaloID of the subhalo whose group to load. SubhaloID is the
      SubLink index and is unique throughout all snapshots.

    fields : list of str
      The columns to load from the table.

    sim : class obj
      Instance of the simulation_box.SimulationBox class, which specifies
      the simulation box to work with.

    numlimit : int, optional
        The maximum number of subhalos to load, ordered by the MassHistory
        column. Default is 0, in which case all subhalos in the group are
        loaded.

    Returns
    -------
    groupsubs : dict
        Dictionary containing the specified fields for the subhalos in the
        FOF group, ordered by the MassHistory column. Note that the input
        subhalo is not guaranteed to be the first in the loaded dictionary
        or guaranteed to be included if a number limit of subhalos is set.
        Entries are stored as numpy arrays. Also includes the number of
        subhalos, the SubLink chunk number in which the subhalos are stored,
        and the row indices of the loaded subhalos in the chunk.

    """

    primary_subhaloid = walk_tree(subhaloid, ['SubhaloID'], sim,
                                  'FirstSubhaloInFOFGroupID')['SubhaloID'][0]

    groupsubs = walk_tree(primary_subhaloid, fields, sim,
                          'NextSubhaloInFOFGroupID')

    """rownum, chunknum = locate_object.row_in_chunk(subhaloid, sim)
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

    return groupsubs"""


def load_tree_progenitors(subhaloid, fields, sim,
                          main_branch_only=False):
    """ Loads specified columns in the SubLink catalog for the subhalo-
    based merger tree, including progenitors of the given subhalo.

    Parameters
    ----------
    ... : type
        ....

    Returns
    -------
    ... : float
        ...

    """

    fields_ = list(set(fields).union(set(['MainLeafProgenitorID',
                                          'LastProgenitorID'])))
    subhalo = load_single_subhalo(subhaloid, fields_, sim)
    rownum, chunknum = locate_object.row_in_chunk(subhaloid, sim)
    catkey = 'SubLink' + str(chunknum)

    start = rownum
    if main_branch_only:
        end = rownum + (subhalo['MainLeafProgenitorID'] -
                        subhalo['SubhaloID'])
    else:
        end = rownum + (subhalo['LastProgenitorID'] -
                        subhalo['SubhaloID'])

    tree = {}
    tree['Number'] = end - start + 1
    tree['ChunkNumber'] = chunknum
    tree['IndexInChunk'] = np.arange(start, end + 1)
    for field in fields:
        tree[field] = sim.preloaded[catkey][field][start : end + 1]
    return tree


def load_tree_descendants(subhaloid, fields, sim,
                          main_branch_only=False):
    """ Loads specified columns in the SubLink catalog for the subhalo-
    based merger tree, including descendants of the given subhalo.

    Parameters
    ----------
    ... : type
        ....

    Returns
    -------
    ... : float
        ...

    """

    fields_ = list(set(fields).union(set(['DescendantID'])))
    subhalo = load_single_subhalo(subhaloid, fields_, sim)
    rownum, chunknum = locate_object.row_in_chunk(subhaloid, sim)
    catkey = 'SubLink' + str(chunknum)

    head = subhalo['SubhaloID']
    head_row = rownum
    idx = [head_row]

    while True:
        jump = sim.preloaded[catkey]['DescendantID'][head_row] - head
        if main_branch_only and jump != -1:
            break
        head_row += jump
        if head_row < 0:
            break
        idx.append(head_row)
        head = sim.preloaded[catkey]['SubhaloID'][head_row]

    tree = {}
    tree['Number'] = len(idx)
    tree['ChunkNumber'] = chunknum
    tree['IndexInChunk'] = np.array(idx)[::-1]
    for field in fields:
        tree[field] = sim.preloaded[catkey][field][idx[::-1]]
    return tree


def load_group_tree():
    """ Loads specified columns in the SubLink catalog for the subhalo-
    based merger tree, including progenitors and/or descendants of all
    subhalos in the same FOF group as the given subhalo.

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
