"""
Module with functions that locate subhalos and convert between different ids.

"""

import numpy as np

import simulation_box

def subfind_central(groupnum, snapnum, sim):
    """ Finds the SubfindID of the primary (most massive) subhalo in a
    given FOF group. Processes any number of groups in the same snapshot.

    Parameters
    ----------
    groupnum : int or array_like
      The group number(s) to find the primary subhalo(s) for. Group number
      is the index of the FOF group in the group catalog. The group number
      is only unique within each snapshot and not throughout the history.

    snapnum : int
      The snapshot number that contains the group(s). Should be one number.

    sim : class obj
      Instance of the simulation_box.SimulationBox class, which specifies
      the simulation box to work with.

    Returns
    -------
    central : int or array_like
      The SubfindID(s) of primary subhalo(s) of the given groups. Has same
      shape as the input groupnum. The SubfindID is only unique within
      each snapshot and not throughout the history.

    """

    if not np.isscalar(snapnum):
        raise TypeError('The input snapnum should be one number.')

    if isinstance(groupnum, tuple):
        groupnum = np.array(groupnum)

    sim.load_data('Group', snapnum, fields = ['GroupFirstSub'])
    central = sim.preloaded['Group'
                            + str(snapnum)]['GroupFirstSub'][groupnum]

    return central

def sublink_id(subfindid, snapnum, sim):
    """ Converts subhalo SubfindID(s) to SubhaloID(s) given the snapshot.
    Processes any number of subhalos in the same snapshot.

    Parameters
    ----------
    subfindid : int or array_like
      The SubfindID(s) of subhalo(s) to find the SubhaloID(s) for. SubfindID
      is only unique within each snapshot and not throughout the history.

    snapnum : int
      The snapshot number that contains the subhalo(s). Should be one number.

    sim : class obj
      Instance of the simulation_box.SimulationBox class, which specifies
      the simulation box to work with.

    Returns
    -------
    subhaloid : int or array_like
      The SubhaloID(s) of the given subhalos. Has same shape as the input
      subfindid. SubhaloID is the SubLink index and is unique throughout all
      snapshots.

    """

    if not np.isscalar(snapnum):
        raise TypeError('The input snapnum should be one number,' +
                        ' process subhalos in different snapshots separately.')

    if isinstance(subfindid, tuple):
        subfindid = np.array(subfindid)

    sim.load_data('SubLinkOffsets', snapnum, fields = ['SubhaloID'])
    subhaloid = sim.preloaded['SubLinkOffsets'
                              + str(snapnum)]['SubhaloID'][subfindid]

    return subhaloid

def chunk_num(subhaloid):
    """ Identifies the chunk of the SubLink tree for given subhalo(s).

    Parameters
    ----------
    subhaloid :  int or array_like
      The SubhaloID(s) of the subhalos to locate. SubhaloID is the SubLink
      index and is unique throughout all snapshots.

    Returns
    -------
    chunknum : int or array_like
      The chunk number of the given subhalos. Has same shape as the input
      subhaloid.

    samechunk : bool
      True if all the given subhalos are in the same tree file chunk, False
      otherwise.

    """

    # doesn't depend on box
    if np.isscalar(subhaloid):
        return subhaloid // int(1e16), True

    chunknum = np.array(subhaloid) // int(1e16)
    samechunk = len(np.unique(chunknum)) == 1

    return chunknum, samechunk

def row_in_chunk(subhaloid, sim):
    """ Locates subhalo(s) within a tree file chunk.

    Parameters
    ----------
    subhaloid :  int or array_like
      The SubhaloID(s) of the subhalos to locate. SubhaloID is the SubLink
      index and is unique throughout all snapshots.

    sim : class obj
      Instance of the simulation_box.SimulationBox class, which specifies the
      simulation box to work with.

    Returns
    -------
    rownum : int or array_like
      The row indices of the given subhalos in the tree chunk. Has same shape
      as the input subhaloid.

    chunknum : int
      The chunk number of the given subhalos. Subhalos required to be in the
      same chunk.

    """
    chunknum, samechunk = chunk_num(subhaloid)
    if not samechunk:
        raise Exception('The subhalos should be in the same tree chunk,' +
                        ' process subhalos in different chunks separately.' +
                        ' Index of first subhalo in a different chunk ' +
                        'is {}.'.format(np.sort(np.unique(chunknum,
                        return_index = True)[1])[1]))
    if not np.isscalar(chunknum):
        chunknum = chunknum[0]
    sim.load_data('SubLink', chunknum, fields = ['SubhaloID'])
    rownum = np.searchsorted(sim.preloaded['SubLink' + str(chunknum)]\
                                          ['SubhaloID'],
                             subhaloid)

    return rownum, chunknum


def subfind_id(subhaloid, sim):
    """ Converts subhalo SubhaloID(s) to SubfindID(s) and SnapNum(s).
    Processes any number of subhalos in the same tree chunk.

    Parameters
    ----------
    subhaloid :  int or array_like
      The SubhaloID(s) of the subhalos to convert. SubhaloID is the SubLink
      index and is unique throughout all snapshots.

    sim : class obj
      Instance of the simulation_box.SimulationBox class, which specifies the
      simulation box to work with.

    Returns
    -------
    subfindid : int or array_like
      The SubfindID(s) of the given subhalo(s). SubfindID is only unique
      within each snapshot and not throughout the history. Has same shape
      as the input subhaloid.

    snapnum : int or array_like
      The snapshot number(s) of the subhalo(s). Has same shape as the input
      subhaloid.

    """

    rownum, chunknum = row_in_chunk(subhaloid, sim)
    sim.load_data('SubLink', chunknum,
                  fields = ['SubfindID', 'SnapNum'])
    subfindid = sim.preloaded['SubLink' + str(chunknum)]['SubfindID'][rownum]
    snapnum = sim.preloaded['SubLink' + str(chunknum)]['SnapNum'][rownum]

    return subfindid, snapnum
