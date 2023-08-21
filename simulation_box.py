"""
Module storing the 'mergertools.simulation_box.simulation_box' class, the
class used for specifying the simulation box and loading data.

"""

import numpy as np
import h5py
import os
from packaging import version
import gc

assert version.parse(h5py.__version__) >= version.parse('2.9'), \
    'h5py>=2.9.x is needed'


# require h5py>=2.9.x for simulation.hdf5.
# use simulation.hdf5 for GroupFirstSub, but still use separate chunks for
# tree, to speed up searches.

class SimulationBox:
    """ The class that stores the simulation box information and loads
    catalogs as dictionaries of numpy arrays.

    """

    def __init__(self, boxsize, resolution, dark, basepath):
        """ Initializes the class with simulation box information, and
        checks whether the box exists.

        Parameters
        ----------

        boxsize : int
          The box size in Mpc, should be one of 50, 100 and 300.

        resolution : int
          The resolution, smaller numbers correspond to higher resolutions.
          The 50Mpc box has 4 levels, and 100 and 300 each has 3.

        dark : bool
          If True, load the dark matter-only box data, if False, load the
          baryonic box data.

        basepath : str
          The base path where all data are stored.

        """

        self.res_dict = {50: [1, 2, 3, 4],
                         100: [1, 2, 3],
                         300: [1, 2, 3]}

        if boxsize not in [50, 100, 300]:
            raise ValueError('Unknown box size: {}. '.format(boxsize) +
                             'Choose from 50, 100, 300.')
        if resolution not in self.res_dict[boxsize]:
            raise ValueError('Unknown resolution for box size ' +
                             '{}: {}. Choose from {}.'.format(
                                 boxsize, resolution, self.res_dict[boxsize]))

        self.Lbox = boxsize
        self.resolution = resolution
        self.dark = dark
        self.basepath = basepath
        self.loaded = {}

        self.columns = {'SubLinkOffsets': ['SubhaloID'],
                        'SubLink': ['SnapNum', 'SubhaloID', 'SubfindID',
                                    'SubhaloGrNr', 'GroupFirstSub',
                                    'FirstProgenitorID', 'NextProgenitorID',
                                    'DescendantID', 'LastProgenitorID',
                                    'MainLeafProgenitorID', 'RootDescendantID',
                                    'FirstSubhaloInFOFGroupID',
                                    'NextSubhaloInFOFGroupID',
                                    'Group_M_TopHat200', 'GroupMass',
                                    'GroupPos', 'SubhaloMass'],
                        'Group': ['GroupFirstSub']}

    def data_path(self, catalog, filenum):
        """ Constructs the path to a data file for the specified catalog
        type and file number.

        Parameters
        ----------
        catalog : str
          The type of catalog to load. Should be one of 'SubLinkOffsets',
          'SubLink', and 'Group'. Catalog descriptions are available at
          https://www.tng-project.org/data/docs/specifications. See sec3a
          for 'SubLinkOffsets', sec4a for 'SubLink', sec2a for 'Group'.

        filenum : int
          The number of the data file to load. This number is the snapshot
          number for 'SubLinkOffsets' and 'Group', and the chunk number for
          'SubLink'.

        Returns
        -------
        fpath : str
          The file path to the data that needs to be loaded.

        """

        if catalog not in ['SubLinkOffsets', 'SubLink', 'Group']:
            raise ValueError('Unknown data catalog: {}. '.format(catalog) +
                             'Choose from SubLinkOffsets, SubLink, Group.')

        if catalog == 'SubLinkOffsets':
            num = str(filenum).zfill(3)
        else:
            num = str(filenum)

        fn_dict = {'SubLinkOffsets': 'offsets_{}.hdf5',
                   'SubLink': 'tree_extended.{}.hdf5',
                   'Group': 'simulation.hdf5'}

        dir_dict = {'SubLinkOffsets': '/postprocessing/offsets/',
                    'SubLink': '/postprocessing/trees/SubLink/',
                    'Group': '/'}

        if self.dark:
            fpath = os.path.join(
                self.basepath,
                'TNG{}-{}-Dark{}'.format(self.Lbox,
                                         self.resolution,
                                         dir_dict[catalog]),
                fn_dict[catalog].format(num))
        else:
            fpath = os.path.join(
                self.basepath,
                'TNG{}-{}{}'.format(self.Lbox,
                                    self.resolution,
                                    dir_dict[catalog]),
                fn_dict[catalog].format(num))

        if os.path.isfile(fpath):
            return fpath

        raise OSError('Invalid file path: {}'.format(fpath))

    def load_by_file(self, catalog, filenum, fields=None):
        """ Loads a data catalog into a dictionary, where keys are consistent
        with TNG column names and columns are converted to numpy arrays.

        Parameters
        ----------
        catalog : str
          The type of catalog to load. Should be one of 'SubLinkOffsets',
          'SubLink', and 'Group'. Catalog descriptions are available at
          https://www.tng-project.org/data/docs/specifications. See sec3a
          for 'SubLinkOffsets', sec4a for 'SubLink', sec2a for 'Group'.

        filenum : int
          The number of the data file to load. This number is the snapshot
          number for 'SubLinkOffsets' and 'Group', and the chunk number for
          'SubLink'.

        fields : list of str, optional
          The columns to load from the table. If not given, the default columns
          will be used.

        Returns
        -------
        None :
          Constructs a dictionary containing columns of the specified catalog
          with keys consistent with TNG column names and columns converted to
          numpy arrays. Appends result to the self.loaded dictionary
          with the input catalog and filenum as key.


        Notes
        -----
        This version is specific to my data organization on greatlakes.

        """

        if fields is None:
            fields = self.columns[catalog]

        if isinstance(fields, str):
            fields = [fields]

        if catalog + str(filenum) in self.loaded:
            existing = set(self.loaded[catalog + str(filenum)].keys())
            fields = list(set(fields) - existing)
            if not fields:  # all columns already loaded
                return
        else:
            self.loaded[catalog + str(filenum)] = {}

        path = self.data_path(catalog, filenum)

        if catalog == 'Group':
            arr_dict = {}
            with h5py.File(path, 'r') as f:
                for field in fields:
                    arr_dict[field] = np.array(f['Groups/{}/Group/'.
                                               format(filenum) + field])

        else:
            arr_dict = {}
            if catalog == 'SubLinkOffsets':
                with h5py.File(path, 'r') as f:
                    for field in fields:
                        arr_dict[field] = np.array(f['Subhalo/SubLink'][field])
            else:
                with h5py.File(path, 'r') as f:
                    for field in fields:
                        arr_dict[field] = np.array(f[field])

        self.loaded[catalog + str(filenum)] = {
            **self.loaded[catalog + str(filenum)],
            **arr_dict}

        return

    def clear_loaded(self, keep_catalogs=None):
        """ Deletes part or all of the loaded catalogs to release memory.

        Parameters
        ----------
        keep_catalogs : list of str, optional
          List of keys to the loaded categories that are to be kept. The
          rest will be deleted. Default is None, in which case all loaded
          catalogs will be deleted.

        Returns
        -------
        None

        """

        if isinstance(keep_catalogs, str):
            keep_catalogs = [keep_catalogs]

        if keep_catalogs:
            self.loaded = {k: self.loaded[k] for k in keep_catalogs}
        else:
            self.loaded = {}
        gc.collect()
        return
