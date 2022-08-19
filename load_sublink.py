import numpy as np

import load_data
import locate_object

def load_single_subhalo(shid, *args, **kwargs):
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
    locate_object.chunk_num(shid)
    load_data.load_data(*args, **kwargs)
    locate_object.row_in_chunk(shid, **kwargs)
    pass

def load_group_subhalos():
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
    pass

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
