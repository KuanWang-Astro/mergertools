"""
Module that loads parts of the SubLink merger trees as required.

"""

import numpy as np

import simulation_box
import locate_object

def load_subhalo_only(subhaloid, fields, sim):
    """ Loads specified columns in the SubLink catalog for a given subhalo.

    Parameters
    ----------
    subhaloid :  int or array_like
      The SubhaloID(s) of the subhalo(s) to load. SubhaloID is the SubLink
      index and is unique throughout all snapshots.

    fields : list of str
      The columns to load from the table.

    sim : class obj
      Instance of the simulation_box.SimulationBox class, which specifies
      the simulation box to work with.

    Returns
    -------
    subhalo : dict
      Dictionary containing the specified fields for the given subhalo(s).
      Entries are stored as single-element arrays. Also includes the number
      of subhalos, the SubLink chunk number of the subhalo(s), and the row
      index(es) of the subhalo(s) in the chunk.

    """

    if np.isscalar(subhaloid):
        subhaloid = np.array([subhaloid])

    rownum, chunknum = locate_object.row_in_chunk(subhaloid, sim)
    catkey = 'SubLink' + str(chunknum)
    sim.load_data('SubLink', chunknum, fields)

    subhalo = {}
    subhalo['Number'] = len(subhaloid)
    subhalo['ChunkNumber'] = chunknum
    subhalo['IndexInChunk'] = rownum
    for field in fields:
        subhalo[field] = sim.preloaded[catkey][field][rownum]

    return subhalo


def walk_tree(subhaloid, fields, sim, pointer):
    """ Walks the SubLink tree following a given pointer (e.g., DescendantID,
    FirstProgenitorID, etc.) iteratively, starting from the given subhalo.

    Parameters
    ----------
    subhaloid :  int
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


    Returns
    -------
    chain : dict
        Dictionary containing the specified fields for the chain of subhalos,
        ordered in the direction of the pointer. Entries are stored as numpy
        arrays. Also includes the number of subhalos, the SubLink chunk number
        in which the subhalos are stored, and the row indices of the loaded
        subhalos in the chunk.

    """

    if not np.isscalar(subhaloid):
        raise TypeError('Process one subhalo at a time.')

    iter_pointers = ['FirstProgenitorID', 'NextProgenitorID',
                     'DescendantID', 'NextSubhaloInFOFGroupID']
    noniter_pointers = ['LastProgenitorID', 'MainLeafProgenitorID',
                        'RootDescendantID', 'FirstSubhaloInFOFGroupID']

    if pointer not in iter_pointers + noniter_pointers:
        raise ValueError('Unknown pointer: {}. '.format(pointer) +
                         'Choose from {}.'.format(iter_pointers +
                                                  noniter_pointers))

    fields_ = list(set(fields).union(set([pointer])))
    subhalo = load_subhalo_only(subhaloid, fields_, sim)
    chunknum = subhalo['ChunkNumber']
    rownum = subhalo['IndexInChunk'][0]
    catkey = 'SubLink' + str(chunknum)

    head = subhalo['SubhaloID'][0]
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
            head = sim.preloaded[catkey]['SubhaloID'][head_row]
    chain = {}
    chain['Number'] = len(idx)
    chain['ChunkNumber'] = chunknum
    chain['IndexInChunk'] = np.array(idx)
    for field in fields:
        chain[field] = sim.preloaded[catkey][field][idx]
    return chain


def load_group_subhalos(subhaloid, fields, sim):
    """ Loads specified columns in the SubLink catalog for all subhalos in
    the same FOF group as the given subhalo, ordered by the MassHistory
    column. A wrapper around `load_sublink.walk_tree`.

    Parameters
    ----------
    subhaloid :  int
      The SubhaloID of the subhalo whose group to load. SubhaloID is the
      SubLink index and is unique throughout all snapshots.

    fields : list of str
      The columns to load from the table.

    sim : class obj
      Instance of the simulation_box.SimulationBox class, which specifies
      the simulation box to work with.

    Returns
    -------
    groupsubs : dict
        Dictionary containing the specified fields for the subhalos in the
        FOF group, ordered by the MassHistory column. Note that the input
        subhalo is not guaranteed to be the first in the loaded dictionary.
        Entries are stored as numpy arrays. Also includes the number of
        subhalos, the SubLink chunk number in which the subhalos are stored,
        and the row indices of the loaded subhalos in the chunk.

    """

    if not np.isscalar(subhaloid):
        raise TypeError('Process one subhalo at a time.')

    fields_ = list(set(fields).union(set(['SubhaloID'])))
    primary_subhaloid = walk_tree(subhaloid, fields_, sim,
                                  'FirstSubhaloInFOFGroupID')['SubhaloID'][-1]

    groupsubs = walk_tree(primary_subhaloid, fields, sim,
                          'NextSubhaloInFOFGroupID')

    return groupsubs


def load_tree_progenitors(subhaloid, fields, sim, main_branch_only):
    """ Loads specified columns in the SubLink catalog for the progenitors
    of the given subhalo.

    Parameters
    ----------
    subhaloid :  int
      The SubhaloID of the subhalo whose progenitors to load. SubhaloID is
      the SubLink index and is unique throughout all snapshots.

    fields : list of str
      The columns to load from the table.

    sim : class obj
      Instance of the simulation_box.SimulationBox class, which specifies
      the simulation box to work with.

    main_branch_only : bool
      If True, only load main branch progenitors of the given subhalo, if
      False, load all progenitors.

    Returns
    -------
    progenitors : dict
        Dictionary containing the specified fields for the progenitors of
        the given subhalo, including itself. Subhalos are sorted starting
        from the input subhalo, in the same order as in the original tree,
        i.e., in a depth-first manner. Entries are stored as numpy arrays.
        Also includes the number of subhalos, the SubLink chunk number in
        which the subhalos are stored, and the row indices of the loaded
        subhalos in the chunk.

    """

    if not np.isscalar(subhaloid):
        raise TypeError('Process one subhalo at a time.')

    fields_ = list(set(fields).union(set(['MainLeafProgenitorID',
                                          'LastProgenitorID'])))
    subhalo = load_subhalo_only(subhaloid, fields_, sim)
    chunknum = subhalo['ChunkNumber']
    rownum = subhalo['IndexInChunk'][0]
    catkey = 'SubLink' + str(chunknum)

    start = rownum
    if main_branch_only:
        end = rownum + (subhalo['MainLeafProgenitorID'][0] -
                        subhalo['SubhaloID'][0])
    else:
        end = rownum + (subhalo['LastProgenitorID'][0] -
                        subhalo['SubhaloID'][0])

    progenitors = {}
    progenitors['Number'] = end - start + 1
    progenitors['ChunkNumber'] = chunknum
    progenitors['IndexInChunk'] = np.arange(start, end + 1)
    for field in fields:
        progenitors[field] = sim.preloaded[catkey][field][start : end + 1]
    return progenitors


def load_tree_descendants(subhaloid, fields, sim, main_branch_only):
    """ Loads specified columns in the SubLink catalog for the descendants
    of the given subhalo. A wrapper around `load_sublink.walk_tree`.

    Parameters
    ----------
    subhaloid :  int
      The SubhaloID of the subhalo whose descendants to load. SubhaloID is
      the SubLink index and is unique throughout all snapshots.

    fields : list of str
      The columns to load from the table.

    sim : class obj
      Instance of the simulation_box.SimulationBox class, which specifies
      the simulation box to work with.

    main_branch_only : bool
      Each subhalo has at most one immediate descendant, but a subhalo is
      not necessarily the main branch progenitor of its descendant. If True,
      only load descendants for which the input subhalo is a main branch
      progenitor. If False, load all descendants.

    Returns
    -------
    descendants : dict
        Dictionary containing the specified fields for the descendants of
        the given subhalo, including itself. Subhalos are sorted from early
        to late and start from the input subhalo. Entries are stored as numpy
        arrays. Also includes the number of subhalos, the SubLink chunk number
        in which the subhalos are stored, and the row indices of the loaded
        subhalos in the chunk.

    """

    if not np.isscalar(subhaloid):
        raise TypeError('Process one subhalo at a time.')

    descendants = walk_tree(subhaloid, fields, sim, 'DescendantID')

    if main_branch_only:
        not_main_branch = (descendants['IndexInChunk'][:-1] - 1 !=
                           descendants['IndexInChunk'][1:])
        if np.any(not_main_branch):
            truncate = np.where(not_main_branch)[0][0]
            descendants['Number'] = truncate + 1
            descendants['IndexInChunk'] = descendants['IndexInChunk']\
                                                     [:truncate + 1]
            for field in fields:
                descendants[field] = descendants[field][:truncate + 1]

    return descendants
