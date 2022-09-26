"""
Module that constructs halo-based merger trees from the subhalo-based trees.

"""

import numpy as np

import simulation_box
import locate_object
import load_sublink

_grsn_combo_const = 1000000000000

def _groupnum_sn_1to2(groupsnapnum):
    return [groupsnapnum % _grsn_combo_const,
            groupsnapnum // _grsn_combo_const]


def _groupnum_sn_2to1(groupnum, snapnum):
    return snapnum * _grsn_combo_const + groupnum


def immediate_progenitor_halos(groupnum, snapnum, sim,
                               subhalo_numlimit=0,
                               previous_snapshot_only=True):
    """ Finds the immediate progenitor halos of the given halo,
    sorted by virial mass.

    """

    central_sfid = locate_object.subfind_central(groupnum, snapnum, sim)
    central_shid = locate_object.sublink_id(central_sfid, snapnum, sim)
    fields = ['SubhaloID', 'Group_M_TopHat200', 'SubhaloGrNr', 'SnapNum']
    subhalos = load_sublink.load_group_subhalos(central_shid,
                    fields, sim, numlimit = subhalo_numlimit)['SubhaloID']

    immediate_progenitor_subhalos = load_sublink.merge_tree_dicts(
                                 [load_sublink.load_immediate_progenitors(
                                                     subhalo, fields, sim)
                                  for subhalo in subhalos],
                                 fields = fields)
    if immediate_progenitor_subhalos['Number'] == 0:
        return np.array([], dtype = 'int64'), np.array([], dtype = 'int64'),\
               np.array([], dtype = '<f4')

    immediate_progenitors = _groupnum_sn_2to1(
                            immediate_progenitor_subhalos['SubhaloGrNr'],
                            immediate_progenitor_subhalos['SnapNum'])
    immediate_progenitors, idx = np.unique(immediate_progenitors,
                                           return_index = True)
    progenitor_masses = immediate_progenitor_subhalos['Group_M_TopHat200'][idx]
    order = np.argsort(-progenitor_masses)

    immediate_progenitors = _groupnum_sn_1to2(immediate_progenitors[order])
    progenitor_masses = progenitor_masses[order]

    if previous_snapshot_only:
        mask = immediate_progenitors[1] == snapnum - 1
        immediate_progenitors[0] = immediate_progenitors[0][mask]
        immediate_progenitors[1] = immediate_progenitors[1][mask]
        progenitor_masses = progenitor_masses[mask]

    return immediate_progenitors[0], immediate_progenitors[1],\
           progenitor_masses


def immediate_descendant_halos(groupnum, snapnum, sim,
                               subhalo_numlimit=20,
                               next_snapshot_only=True):
    """ Finds the immediate descendant halos of the given halo,
    sorted by virial mass.

    """

    central_sfid = locate_object.subfind_central(groupnum, snapnum, sim)
    central_shid = locate_object.sublink_id(central_sfid, snapnum, sim)
    fields = ['SubhaloID', 'Group_M_TopHat200', 'SubhaloGrNr']
    subhalos = load_sublink.load_group_subhalos(central_shid,
                    fields, sim, numlimit = subhalo_numlimit)['SubhaloID']

    immediate_descendant_subhalos = load_sublink.merge_tree_dicts(
                                 [load_sublink.load_immediate_descendant(
                                                    subhalo, fields, sim)
                                  for subhalo in subhalos],
                                 fields = fields)
    if immediate_descendant_subhalos['Number'] == 0:
        return np.array([], dtype = 'int64'), np.array([], dtype = 'int64'),\
               np.array([], dtype = '<f4')

    immediate_descendants = _groupnum_sn_2to1(*locate_object.group_num(
                                immediate_descendant_subhalos['SubhaloID'],
                                sim, internal = True))
    immediate_descendants, idx = np.unique(immediate_descendants,
                                           return_index = True)
    descendant_masses = immediate_descendant_subhalos['Group_M_TopHat200'][idx]
    order = np.argsort(-descendant_masses)

    immediate_descendants = _groupnum_sn_1to2(immediate_descendants[order])
    descendant_masses = descendant_masses[order]

    if next_snapshot_only:
        mask = immediate_descendants[1] == snapnum + 1
        immediate_descendants[0] = immediate_descendants[0][mask]
        immediate_descendants[1] = immediate_descendants[1][mask]
        descendant_masses = descendant_masses[mask]

    return immediate_descendants[0], immediate_descendants[1],\
           descendant_masses


def _immediate_progenitor_halos(subhaloid, sim,
                                subhalo_numlimit=0,
                                previous_snapshot_only=True):
    """ Finds the immediate progenitor halos of the given halo,
    sorted by virial mass.

    """

    fields = ['SubhaloID', 'Group_M_TopHat200', 'SubhaloGrNr', 'SnapNum']
    group_subhalos = load_sublink.load_group_subhalos(subhaloid, fields,
                                       sim, numlimit = subhalo_numlimit)
    subhalos = group_subhalos['SubhaloID']
    snapnum = group_subhalos['SnapNum'][0]

    immediate_progenitor_subhalos = load_sublink.merge_tree_dicts(
                                 [load_sublink.load_immediate_progenitors(
                                                     subhalo, fields, sim)
                                  for subhalo in subhalos],
                                 fields = fields)
    if immediate_progenitor_subhalos['Number'] == 0:
        return np.array([], dtype = 'int64'), np.array([], dtype = 'int64'),\
               np.array([], dtype = '<f4')

    immediate_progenitors = _groupnum_sn_2to1(
                            immediate_progenitor_subhalos['SubhaloGrNr'],
                            immediate_progenitor_subhalos['SnapNum'])
    immediate_progenitors, idx = np.unique(immediate_progenitors,
                                           return_index = True)
    progenitor_subs = immediate_progenitor_subhalos['SubhaloID'][idx]
    progenitor_masses = immediate_progenitor_subhalos['Group_M_TopHat200'][idx]
    order = np.argsort(-progenitor_masses)

    immediate_progenitors = _groupnum_sn_1to2(immediate_progenitors[order])
    progenitor_subs = progenitor_subs[order]
    progenitor_masses = progenitor_masses[order]

    if previous_snapshot_only:
        mask = immediate_progenitors[1] == snapnum - 1
        immediate_progenitors[0] = immediate_progenitors[0][mask]
        immediate_progenitors[1] = immediate_progenitors[1][mask]
        progenitor_subs = progenitor_subs[mask]
        progenitor_masses = progenitor_masses[mask]

    return immediate_progenitors[0], immediate_progenitors[1],\
           progenitor_subs, progenitor_masses


def main_merger_tree(subhaloid, sim):
    """


    """
    pass

def progenitor_halos(groupnum, snapnum, sim):
    """ Loads the progenitor halos (FOF groups) of the given halo. Progenitor
    halos are defined as halos that contain subhalos that are progenitors of
    the subhalos in the input halo.

    Parameters
    ----------
    groupnum : int
      The group number to find the progenitor halos for. Group number
      is the index of the FOF group in the group catalog. The group number
      is only unique within each snapshot and not throughout the history.

    Returns
    -------
    breadth-first.

    """

    central_sfid = locate_object.subfind_central(groupnum, snapnum, sim)
    central_shid = locate_object.sublink_id(central_sfid, snapnum, sim)
    fields = ['SubhaloID', 'DescendantID', 'LastProgenitorID',
              'Group_M_TopHat200']
    subhalos = load_sublink.load_group_subhalos(central_shid,
                                                fields, sim)['SubhaloID']

    progenitor_subhalos = load_sublink.merge_tree_dicts(
                                 [load_sublink.load_tree_progenitors(
                                            subhalo, fields, sim,
                                            main_branch_only = False)
                                  for subhalo in subhalos],
                                 fields = fields)

    mask = (progenitor_subhalos['DescendantID'] != -1)
    tree_halos = _groupnum_sn_2to1(*locate_object.group_num(
                progenitor_subhalos['SubhaloID'][mask], sim,
                internal = True))
    descendant_pointers = _groupnum_sn_2to1(*locate_object.group_num(
                     progenitor_subhalos['DescendantID'][mask], sim,
                     internal = True))
    order = np.argsort(descendant_pointers)
    descendant_pointers = descendant_pointers[order]
    tree_halos = tree_halos[order]
    breaks = np.append(np.where(np.roll(descendant_pointers, -1) !=
                                descendant_pointers)[0],
                       len(descendant_pointers))
    halos_uniq = _groupnum_sn_1to2(descendant_pointers[breaks[:-1]])
    halos_prog = np.array([_groupnum_sn_1to2(np.unique(
                           tree_halos[breaks[i] : breaks[i + 1]]))
                           for i in range(len(breaks) - 1)], dtype = object)

    progenitor_halos_dict = {'Number': len(halos_uniq[0]),
                             'HaloGroupNum': halos_uniq[0],
                             'HaloSnapNum': halos_uniq[1],
                             'ProgenitorGroupNum': halos_prog[:, 0],
                             'ProgenitorSnapNum': halos_prog[:, 1]}

    return progenitor_halos_dict



def descendant_halos():
    pass
