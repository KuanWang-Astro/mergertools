"""
Module storing the 'mergertools.simulation_box.simulation_box' class, the
class used for specifying the simulation box and preloading data.

"""

import numpy as np
import h5py
import os

# require h5py>=2.9.x for simulation.hdf5
# use simulation.hdf5 for GroupFirstSub, but still use separate chunks for tree

class SimulationBox:
    """ The class that stores the simulation box information and preloads
    catalogs as dictionaries of numpy arrays.

    """

    def __init__(self, Lbox, resolution, dark, basepath):
        """ Initializes the class with simulation box information, and
        checks whether the box exists.

        Parameters
        ----------

        Lbox : int
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

        self.res_dict = {50:  [1, 2, 3, 4],
                         100: [1, 2, 3],
                         300: [1, 2, 3]}

        if Lbox not in [50, 100, 300]:
            raise ValueError('Unknown box size: {}. '.format(str(Lbox)) +
                             'Choose from 50, 100, 300.')
        if resolution not in self.res_dict[Lbox]:
            raise ValueError('Unknown resolution for box size ' +
                             '{}: {}. Choose from {}.'.format(
                             str(Lbox), str(resolution),
                             str(self.res_dict[Lbox])[1 : -1]))

        self.Lbox = Lbox
        self.resolution = resolution
        self.dark = dark
        self.basepath = basepath
        self.preloaded = {}

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

        Notes
        -----
        This version is specific to my data organization on greatlakes.

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
                   'Group': 'host_sub{}.npy'}

        if self.dark:
            fpath = os.path.join(
                self.basepath,
                'TNG{}-{}-Dark_{}'.format(str(self.Lbox),
                                          str(self.resolution),
                                          catalog),
                fn_dict[catalog].format(num))
        else:
            fpath = os.path.join(
                self.basepath,
                'TNG{}-{}_{}'.format(str(self.Lbox),
                                     str(self.resolution),
                                     catalog),
                fn_dict[catalog].format(num))

        if os.path.isfile(fpath):
            return fpath

        raise OSError('Invalid file path: {}'.format(fpath))

    def load_data(self, catalog, filenum, keys = None):
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

        keys : list of str, optional
          The columns to load from the table. If not specified, load all
          available columns in file.

        Returns
        -------
        None :
          Constructs a dictionary containing columns of the specified catalog
          with keys consistent with TNG column names and columns converted to
          numpy arrays. Appends result to the self.preloaded dictionary
          with the input catalog and filenum as key.


        Notes
        -----
        This version is specific to my data organization on greatlakes.

        """

        path = self.data_path(catalog, filenum)

        if isinstance(keys, str):
            keys = [keys]

        if catalog == 'Group':
            arr = np.load(path)
            arr_dict = {'GroupFirstSub': arr}

        else:
            if catalog == 'SubLinkOffsets':
                f = h5py.File(path, 'r')['Subhalo/SubLink']
            elif catalog == 'SubLink':
                f = h5py.File(path, 'r')

            arr_dict = {}
            if keys is None:
                keys = f.keys()
            if self.preloaded.__contains__(catalog + str(filenum)):
                existing = set(self.preloaded[catalog + str(filenum)].keys())
                keys = list(set(keys) - existing)
                if len(keys) == 0:
                    return
                
            for key in keys:
                arr_dict[key] = np.array(f[key])
            f.close()

        if self.preloaded.__contains__(catalog + str(filenum)):
            self.preloaded[catalog + str(filenum)] = {
                    **self.preloaded[catalog + str(filenum)],
                    **arr_dict}
        else:
            self.preloaded[catalog + str(filenum)] = arr_dict

        return
