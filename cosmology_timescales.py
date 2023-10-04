"""
Module with cosmological and dynamical timescale calculations.

"""

import numpy as np
from scipy.integrate import quad
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM

# Constants for unit conversion
const1 = 0.102271217  # 100km/s/Mpc = const1 Gyr^-1, H0 = h * const1 Gyr^-1
const2 = 3.2407793E-18  # 100km/s/Mpc = const2 s^-1, H0 = h * const2 s^-1
G = 4.5170908E-48  # grav. constant in s^-2/(Msun/Mpc^3)

# Snapshot scale factors in TNG
a_list = np.load('a_list.npy')


class LCDMCosmology:
    """ The class that stores the cosmology information and calculates
    relevant quantities.

    """

    def __init__(self, h=0.6774, omega_m=0.3089, a_list=a_list):
        """ Initializes the class with cosmology information.

        Parameters
        ----------

        h : float, optional
          The dimensionless Hubble constant, default is the TNG value.

        omega_m : float, optional
          The matter density parameter, default is the TNG value.

        a_list : array_like
          List of scale factors for the snapshots in simulation, default
          is the TNG snapshots, accessed from the TNG website.

        Returns
        -------
        None :
          Initializes the class object with input arguments.

        """

        self.h = h
        self.omega_m = omega_m
        self.omega_l = 1 - omega_m
        self.cosmo = FlatLambdaCDM(H0=h * 100, Om0=omega_m)
        self.a_list = a_list
        self.z_list = 1 / a_list - 1

    def e2(self, a):
        """ Calculates the E^2 function for the specified LambdaCDM
        cosmology at scale factor a, where E(a) = H(a) / H0.

        Parameters
        ----------
        a : float
          Scale factor.

        Returns
        -------
        E2 : float
          The E^2 function at epoch a.

        """

        return self.omega_m / (a ** 3) + self.omega_l

    def delta_c(self, a):
        """ Calculates the virial overdensity with respect to the
        critical density of the universe, for the specified LambdaCDM
        cosmology at scale factor a. See Bryan & Norman 1998 Equation 6.

        Parameters
        ----------
        a : float
          Scale factor.

        Returns
        -------
        Delta_c : float
          The virial overdensity with respect to the critical density
          of the universe at scale factor a.

        """

        x = - self.omega_l / self.e2(a)
        return 18 * np.pi * np.pi + 82 * x - 39 * x * x

    def rho_c(self, a):
        """ Calculates the critical density of the universe, for the
        specified LambdaCDM cosmology at scale factor a. In physical
        units of Msun / Mpc^3.

        Parameters
        ----------
        a : float
          Scale factor.

        Returns
        -------
        rho_c : float
          The critical density of the universe at scale factor a. In
          physical units of Msun / Mpc^3.

        """

        return (3 * self.h * self.h * const2 * const2
                * self.e2(a) / 8 / np.pi / G)

    def cosmo_age(self, a):  # in Gyr
        z = 1 / a - 1
        return self.cosmo.age(z)

    def t_dyn(self, a): # in Gyr
        return np.pi / self.h / const1 / np.sqrt(self.e2(a) * 2 * self.delta_c(a))


# change to take a instead of z as input
    def n_dyn(self, a, aref=None): # number of tdyn's at a since ai
        if aref is None:
            aref = self.a_list[0]
        z = 1 / a - 1
        zref = 1 / aref - 1
        return quad(lambda x: 1 / (1 + x) / self.t_dyn(1 / (1 + x)) / np.sqrt(self.e2(x)),
                    z, zref)[0] / self.h / const1

a_interp = np.linspace(0.02, 1, 1000)
n_interp = np.array([n_dyn(a) for a in a_interp])
ntau_a = interp1d(a_interp, n_interp)
a_ntau = interp1d(n_interp, a_interp)

z_list = 1 / a_list - 1
n_list = ntau_a(a_list)

# Major mergers

def a_before_mm(a_mm, n_back = 1):
    n_mm = ntau_a(a_mm)
    n_before = n_mm - 1
    assert n_before >= 0
    return a_ntau(n_before)

def a_after_mm(a_mm, n_forth = 1):
    n_mm = ntau_a(a_mm)
    n_after = n_mm + 1
    assert n_after <= ntau_a(1)
    return a_ntau(n_after)
