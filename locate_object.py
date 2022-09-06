"""
Module with functions that locate subhalos and convert between different ids.

"""

import numpy as np

import simulation_box

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

    sim.load_by_file('SubLinkOffsets', snapnum, fields = ['SubhaloID'])
    subhaloid = sim.loaded['SubLinkOffsets'
                              + str(snapnum)]['SubhaloID'][subfindid]

    return subhaloid


def chunk_num(subhaloid):
    """ Identifies the chunk of the SubLink tree for given subhalo(s). This
    information is available from the subhaloid alone and does not depend on
    the simulation box.

    Parameters
    ----------
    subhaloid :  int or array_like
      The SubhaloID(s) of the subhalos to locate. SubhaloID is the SubLink
      index and is unique throughout all snapshots.

    Returns
    -------
    chunknum : int or array_like
      The SubLink chunk number of the given subhalos. Has same shape as the
      input subhaloid.

    samechunk : bool
      True if all the given subhalos are in the same tree file chunk, False
      otherwise.

    """

    if np.isscalar(subhaloid):
        return subhaloid // 10000000000000000, True

    chunknum = np.array(subhaloid) // 10000000000000000
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
      The SubLink chunk number of the given subhalos. Subhalos required to be
      in the same chunk.

    """
    chunknum, samechunk = chunk_num(subhaloid)
    if not samechunk:
        raise ValueError('The subhalos should be in the same tree chunk,' +
                         ' process subhalos in different chunks separately.' +
                         ' Index of first subhalo in a different chunk ' +
                         'is {}.'.format(np.sort(np.unique(chunknum,
                         return_index = True)[1])[1]))
    if not np.isscalar(chunknum):
        chunknum = chunknum[0]
    sim.load_by_file('SubLink', chunknum, fields = ['SubhaloID'])
    rownum = np.searchsorted(sim.loaded['SubLink' + str(chunknum)]\
                                          ['SubhaloID'],
                             subhaloid)
    if subhaloid != sim.loaded['SubLink' + str(chunknum)]['SubhaloID']\
                                 [rownum]:
        raise ValueError('SubhaloID does not exist.')

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
    sim.load_by_file('SubLink', chunknum,
                  fields = ['SubfindID', 'SnapNum'])
    subfindid = sim.loaded['SubLink' + str(chunknum)]['SubfindID'][rownum]
    snapnum = sim.loaded['SubLink' + str(chunknum)]['SnapNum'][rownum]

    return subfindid, snapnum


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

    sim.load_by_file('Group', snapnum, fields = ['GroupFirstSub'])
    central = sim.loaded['Group'
                            + str(snapnum)]['GroupFirstSub'][groupnum]

    return central


def group_num(subhaloid, sim):
    """ Identifies the index(es) of the halo(s) hosting the input subhalo(s)
    given the SubhaloID(s) and snapshot. Processes any number of subhalos in
    the same tree chunk.

    Parameters
    ----------
    subhaloid :  int or array_like
      The SubhaloID(s) of the subhalos to find group numbers for. SubhaloID
      is the SubLink index and is unique throughout all snapshots.

    sim : class obj
      Instance of the simulation_box.SimulationBox class, which specifies
      the simulation box to work with.

    Returns
    -------
    groupnum : int or array_like
      The SubhaloGrNr(s) of the given subhalos. Has same shape as the input
      subhaloid. SubhaloGrNr is the index of the halo in the group catalog.

    snapnum : int
      The snapshot number that contains the subhalo(s). Should be one number.

    """

    rownum, chunknum = row_in_chunk(subhaloid, sim)
    sim.load_by_file('SubLink', chunknum,
                  fields = ['SubfindID', 'SnapNum'])
    groupnum = sim.loaded['SubLink' + str(chunknum)]['SubhaloGrNr'][rownum]
    snapnum = sim.loaded['SubLink' + str(chunknum)]['SnapNum'][rownum]

    return groupnum, snapnum


def is_subfind_central():
    pass


def is_sublink_central():
    pass
