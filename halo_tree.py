"""
Module that constructs halo-based merger trees from the subhalo-based trees.

"""

import numpy as np

import simulation_box
import locate_object
import load_sublink


def _groupnum_sn_1to2(groupsnapnum):
    return [groupsnapnum % 1000000000000,
            groupsnapnum // 1000000000000]


def _groupnum_sn_2to1(groupnum, snapnum):
    return snapnum * 1000000000000 + groupnum


def immediate_progenitor_halos(groupnum, snapnum, sim):
    """ Finds the immediate progenitor halos of the given halo,
    sorted by virial mass.

    """

    central_sfid = locate_object.subfind_central(groupnum, snapnum, sim)
    central_shid = locate_object.sublink_id(central_sfid, snapnum, sim)
    fields = ['SubhaloID', 'Group_M_TopHat200', 'SubhaloGrNr']
    subhalos = load_sublink.load_group_subhalos(central_shid,
                                                fields, sim)['SubhaloID']

    immediate_progenitor_subhalos = load_sublink.merge_tree_dicts(
                                 *[load_sublink.load_immediate_progenitors(
                                             subhalo, fields, sim)
                                 for subhalo in subhalos],
                                 fields = fields)

    immediate_progenitors = _groupnum_sn_2to1(*locate_object.group_num(
                                immediate_progenitor_subhalos['SubhaloID'],
                                sim))
    immediate_progenitors, idx = np.unique(immediate_progenitors,
                                           return_index = True)
    progenitor_masses = immediate_progenitor_subhalos['Group_M_TopHat200'][idx]
    order = np.argsort(-progenitor_masses)

    immediate_progenitors = _groupnum_sn_1to2(immediate_progenitors[order])
    progenitor_masses = progenitor_masses[order]

    return immediate_progenitors, progenitor_masses


def immediate_descendant_halos(groupnum, snapnum, sim):
    """ Finds the immediate descendant halos of the given halo,
    sorted by virial mass.

    """

    central_sfid = locate_object.subfind_central(groupnum, snapnum, sim)
    central_shid = locate_object.sublink_id(central_sfid, snapnum, sim)
    fields = ['SubhaloID', 'Group_M_TopHat200', 'SubhaloGrNr']
    subhalos = load_sublink.load_group_subhalos(central_shid,
                                                fields, sim)['SubhaloID']

    immediate_descendant_subhalos = load_sublink.merge_tree_dicts(
                                 *[load_sublink.load_immediate_descendant(
                                             subhalo, fields, sim)
                                 for subhalo in subhalos],
                                 fields = fields)

    immediate_descendants = _groupnum_sn_2to1(*locate_object.group_num(
                                immediate_descendant_subhalos['SubhaloID'],
                                sim))
    immediate_descendants, idx = np.unique(immediate_descendants,
                                           return_index = True)
    descendant_masses = immediate_descendant_subhalos['Group_M_TopHat200'][idx]
    order = np.argsort(-descendant_masses)

    immediate_descendants = _groupnum_sn_1to2(immediate_descendants[order])
    descendant_masses = descendant_masses[order]

    return immediate_descendants, descendant_masses


def main_merger_tree():
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
                                 *[load_sublink.load_tree_progenitors(
                                             subhalo, fields, sim,
                                             main_branch_only = False)
                                 for subhalo in subhalos],
                                 fields = fields)

    mask = (progenitor_subhalos['DescendantID'] != -1)
    tree_halos = _groupnum_sn_2to1(*locate_object.group_num(
               progenitor_subhalos['SubhaloID'][mask], sim))
    descendant_pointers = _groupnum_sn_2to1(*locate_object.group_num(
                     progenitor_subhalos['DescendantID'][mask], sim))
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
