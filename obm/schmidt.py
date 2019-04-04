#! /usr/bin/env python
import numpy as np
from .constants import constants

T0_Kelvin = constants.T0_Kelvin
rho_ref = constants.rho_ref
R_gasconst = constants.R_gasconst


def _diff(S, T, D0, Ea):
    """
    Reference:
    He, Ne, Kr, Xe freshwater values from Jahne et al., 1987.
    "Measurement of Diffusion Coeffients of Sparingly Soluble Gases in Water"
    J. Geophys. Res., 92(C10), 10767-10776.

    output: diff [m^2/s]

    """
    TK = T + T0_Kelvin
    diff = D0 * np.exp(Ea / (R_gasconst * 1e-3 * TK)) * 1e-5
    diff = diff * (1.0 - 0.049 * S / 35.5)  # salinity correction
    diff = diff * 1.0e-4  # convert from cm^2/s to m^2/s
    return diff


def _visc(S, T):
    """
    Compute kinematic viscosity of seawater
    visc: m^2/s
    """
    visc = 1e-4 * (17.91 - 0.5381 * T + 0.00694 * T**2. + 0.02305 * S)
    return visc / rho_ref


def N2(S, T):
    D0 = 3412.0
    Ea = -18.5
    return _visc(S, T) / _diff(S, T, D0, Ea)


def Ne(S, T):
    D0 = 1608.0
    Ea = -14.84
    return _visc(S, T) / _diff(S, T, D0, Ea)


def Ar(S, T):
    D0 = 2227.0
    Ea = -16.68
    return _visc(S, T) / _diff(S, T, D0, Ea)


def Kr(S, T):
    D0 = 6393.0
    Ea = -20.20
    return _visc(S, T) / _diff(S, T, D0, Ea)


def Xe(S, T):
    D0 = 9007.0
    Ea = -21.61
    return _visc(S, T) / _diff(S, T, D0, Ea)


def O2(S, T):
    a = 1638.0
    b = 81.83
    c = 1.483
    d = 0.008004
    return a + T * (-b + T * (c + T * (-d)))


def CO2(S, T):
    T_sq = T * T
    T_cu = T_sq * T
    a = np.array([2073.1, 125.62, 3.6276, 0.043219])
    sc_gas = a[0] - a[1] * T + a[2] * T_sq - a[3] * T_cu

    return sc_gas
