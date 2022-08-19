"""
Module with functions that locate subhalos convert between different ids.

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
      The SubfindID(s) of primary subhalo(s) of the given groups. In same
      format as the input groupnum. The SubfindID is only unique within
      each snapshot and not throughout the history.

    """

    if not np.isscalar(snapnum):
        raise TypeError('The input snapnum should be one number.')

    sim.load_data('Group', snapnum, keys = ['GroupFirstSub'])
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
      The SubhaloID(s) of the given subhalos. In same format as the input
      subfindid. SubhaloID is the SubLink index and is unique throughout all
      snapshots.

    """

    if not np.isscalar(snapnum):
        raise TypeError('The input snapnum should be one number.')

    sim.load_data('SubLinkOffsets', snapnum, keys = ['SubhaloID'])
    subhaloid = sim.preloaded['SubLinkOffsets'
                              + str(snapnum)]['SubhaloID'][subfindid]

    return subhaloid

def chunk_num(subhaloid):
    # doesn't depend on box
    if np.isscalar(subhaloid):
        return subhaloid // int(1e16)
    uniq = np.unique(np.array(subhaloid) // int(1e16))
    if len(uniq) == 1:
        return uniq[0]
    return np.array(subhaloid) // int(1e16)

def row_in_chunk(subhaloid, tree_dict = None, **box_kwargs):
    chunknum = chunk_num(subhaloid)
    if not np.isscalar(chunknum):
        raise Exception('Subhalos must be in the same tree chunk!')
    if tree_dict is None:
        tree_dict = load_data.load_data('SubLink', chunknum,
                             keys = ['SubhaloID'],
                             **box_kwargs)
    return np.searchsorted(tree_dict['SubhaloID'], subhaloid)


def subfind_id(subhaloid, tree_dict = None, **box_kwargs):
    """
    converts SubhaloID to SubfindID and SnapNum for subhalo(s)
    1 subhaloid or multiple subhaloid in same tree chunk
    """
    # add check for needed columns
    if tree_dict is None:
        tree_dict = load_data.load_data('SubLink', Chunknum(subhaloid),
                             keys = ['SubhaloID', 'SubfindID', 'SnapNum'],
                             **box_kwargs)
    idx = np.searchsorted(tree_dict['SubhaloID'], subhaloid)
    return tree_dict['SubfindID'][idx], tree_dict['SnapNum'][idx]
