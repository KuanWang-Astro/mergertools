"""
Module that constructs halo-based merger trees from the subhalo-based trees.

"""

import numpy as np
import h5py
import os

import load_by_file
import locate_object
import load_sublink


def progenitor_halos(groupnum, snapnum, sim):
    """ Loads the immediate progenitor halos (FOF groups) of the given halo.
    Immediate progenitor halos are defined as halos that contain subhalos that
    are immediate progenitors of the subhalos in the input halo. Note that in
    some rare cases immediate progenitors are not in the immediately previous
    snapshot.

    Parameters
    ----------
    groupnum : int
      The group number to find the progenitor halos for. Group number
      is the index of the FOF group in the group catalog. The group number
      is only unique within each snapshot and not throughout the history.

    Returns
    -------


    """

    central = locate_object.subfind_central(groupnum, snapnum, sim)
    subhalos = locat
    progenitor_subhalos =
    progenitor_halos =





def descendant_halos():
    pass

def track_halo_history():
    pass

def construct_halo_tree():
    pass
