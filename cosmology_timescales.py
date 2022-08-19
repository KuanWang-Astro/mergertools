import numpy as np
import collections
from scipy import stats
import itertools
import pandas
from scipy.integrate import quad
from scipy.interpolate import interp1d
from astropy.cosmology import FlatLambdaCDM
#from sklearn.linear_model import LinearRegression



#############################
# Cosmological calculations #
#############################

h = 0.688
omega_m = 0.295
omega_L = 1 - omega_m
cosmo = FlatLambdaCDM(H0 = h * 100, Om0 = omega_m)
const1 = 0.102271217 # 100km/s/Mpc = const1 Gyr^-1, H0 = h * const1 Gyr^-1
const2 = 3.2407793E-18 # 100km/s/Mpc = const2 s^-1, H0 = h * const2 s^-1
G = 4.5170908E-48 # grav. constant in s^-2/(Msun/Mpc^3)

def E2(z):
    return omega_m * (1 + z) ** 3 + omega_L

def Delta_c(z):
    x = - omega_L / E2(z)
    return 18 * np.pi * np.pi + 82 * x - 39 * x * x

def rho_c(z): # physical in Msun / Mpc^3
    return 3 * h * h * const2 * const2 * E2(z) / 8 / np.pi / G

def cosmo_age(a): # in Gyr
    z = 1 / a - 1
    return cosmo.age(z)


#############################
###### Dynamical times ######
#############################

def t_dyn(a): # in Gyr
    z = 1 / a - 1
    return np.pi / h / const1 / np.sqrt(E2(z) * 2 * Delta_c(z))

def n_t_dyn(a, ai = 0.06): # number of tdyn's at a since ai
    z = 1 / a - 1
    zi = 1 / ai - 1
    return quad(lambda x: 1 / (1 + x) / t_dyn(1 / (1 + x)) / np.sqrt(E2(x)),
                z, zi)[0] / h / const1

a_interp = np.linspace(0.06, 1, 1000)
n_interp = np.array([n_t_dyn(a) for a in a_interp])
ntau_a = interp1d(a_interp, n_interp)
a_ntau = interp1d(n_interp, a_interp)

#############################
#### Time steps in tree #####
#############################

a_list = np.hstack((np.linspace(6, 10, 9), np.linspace(11, 100, 90))) / 100
z_list = 1 / a_list - 1
n_list = ntau_a(a_list)



#############################
####### Major mergers #######
#############################

def a_before_mm(a_mm, n_back = 1):
    n_mm = ntau_a(a_mm)
    n_before = n_mm - 1
    assert n_before >= 0
    return a_nt(n_before)

def a_after_mm(a_mm, n_forth = 1):
    n_mm = ntau_a(a_mm)
    n_after = n_mm + 1
    assert n_after <= ntau_a(1)
    return a_nt(n_after)
