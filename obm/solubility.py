#! /usr/bin/env python
import numpy as np
from .constants import constants

T0_Kelvin = constants.T0_Kelvin
rho_ref = constants.rho_ref

salt_min = 0.1


def _garcia_gordon_polynomial(S, T,
                              A0=0., A1=0., A2=0., A3=0., A4=0., A5=0.,
                              B0=0., B1=0., B2=0., B3=0.,
                              C0=0.):

    T_scaled = np.log((298.15 - T) / (273.15 + T))
    return np.exp(A0 + A1 * T_scaled + A2 * T_scaled**2. + A3 * T_scaled**3. + A4 * T_scaled**4. + A5 * T_scaled**5. +
                  S * (B0 + B1 * T_scaled + B2 * T_scaled**2. + B3 * T_scaled**3.) + C0 * S**2.)


def _umolkg_to_mmolm3(value):
    return value * rho_ref / 1000.


def Ar(S, T, **kwargs):
    '''
    Solubility of Ar in sea water
    INPUT:  (if S and T are not singular they must have same dimensions)
    S=salinity    [PSS]
    T=temperature [degree C]

    conc=solubility of Ar [mmol/m^3]

    REFERENCE:
    Roberta Hamme and Steve Emerson, 2004.
    "The solubility of neon, nitrogen and argon in distilled water and seawater."
    Deep-Sea Research I, 51(11), p. 1517-1528.
    '''

    # constants from Table 4 of Hamme and Emerson 2004
    conc = _garcia_gordon_polynomial(S, T,
                                     A0=2.79150,
                                     A1=3.17609,
                                     A2=4.13116,
                                     A3=4.90379,
                                     B0=-6.96233e-3,
                                     B1=-7.66670e-3,
                                     B2=-1.16888e-2)
    return _umolkg_to_mmolm3(conc)


def Ne(S, T, **kwargs):
    '''
    Solubility (saturation) of neon (Ne) in sea water
    at 1-atm pressure of air including saturated water vapor

    INPUT:  (if S and T are not singular they must have same dimensions)
    S=salinity    [PSS]
    T=temperature [degree C]

    OUTPUT:
    conc=solubility of Ne  [mmol/m^3]

    REFERENCE:
    Roberta Hamme and Steve Emerson, 2004.
    "The solubility of neon, nitrogen and argon in distilled water and seawater."
    Deep-Sea Research I, 51(11), p. 1517-1528.
    '''

    conc = _garcia_gordon_polynomial(S, T,
                                     A0=2.18156,
                                     A1=1.29108,
                                     A2=2.12504,
                                     B0=-5.94737e-3,
                                     B1=-5.13896e-3)
    # convert from nmol/kg to umol/kg
    conc = conc / 1000.
    return _umolkg_to_mmolm3(conc)


def Xe(S, T, **kwargs):
    '''
    Solubility (saturation) of xeon (Xe) in sea water
    at 1-atm pressure of air including saturated water vapor

    INPUT:  (if S and T are not singular they must have same dimensions)
    S=salinity    [PSS]
    T=temperature [degree C]

    OUTPUT:
    conc=solubility of Xe [mmol/m^3]

    REFERENCE:
    R. Hamme fit to data of
    D. Wood and R. Caputi (1966) "Solubilities of Kr and Xe in fresh and sea water"
    U.S. Naval Radiological Defense Laboratory, Technical Report USNRDL-TR-988,
    San Francisco, CA, pp. 14.
    '''

    conc = _garcia_gordon_polynomial(S, T,
                                     A0=-7.48679,
                                     A1=5.08637,
                                     A2=4.22243,
                                     B0=-8.15683e-3,
                                     B1=-1.20129e-3)

    return _umolkg_to_mmolm3(conc)


def N2(S, T, **kwargs):
    '''
    Solubility (saturation) of nitrogen (N2) in sea water
    at 1-atm pressure of air including saturated water vapor

    INPUT:  (if S and T are not singular they must have same dimensions)
    S=salinity    [PSS]
    T=temperature [degree C]

    OUTPUT:
    conc=solubility of N2  [mol/m^3]

    REFERENCE:
    Roberta Hamme and Steve Emerson, 2004.
    "The solubility of neon, nitrogen and argon in distilled water and seawater."
    Deep-Sea Research I, 51(11), p. 1517-1528.
    '''

    conc = _garcia_gordon_polynomial(S, T,
                                     A0=6.42931,
                                     A1=2.92704,
                                     A2=4.32531,
                                     A3=4.69149,
                                     B0=-7.44129e-3,
                                     B1=-8.02566e-3,
                                     B2=-1.46775e-2)
    return _umolkg_to_mmolm3(conc)


def Kr(S, T, **kwargs):
    '''
    Compute solubilities of krypton at 1 atm including saturated water vapor

    INPUT:  (if S and T are not singular they must have same dimensions)
    S=salinity    [PSS]
    T=temperature [degree C]

    OUTPUT:
    conc=solubility of Ke  [mmol/m^3]

    REFERENCE:
    Ray F. Weiss and T. Kurt Kyser (1978)
    "Solubility of Krypton in Water and Seawater"
    Journal of Chemical Thermodynamics, 23(1), 69-72.
    '''
    A1 = -112.6840
    A2 = 153.5817
    A3 = 74.4690
    A4 = -10.0189
    B1 = -0.011213
    B2 = -0.001844
    B3 = 0.0011201

    TK = T + T0_Kelvin
    TKp01 = TK / 100.0

    conc = np.exp(A1 + A2 * 100.0 / TK + A3 * np.log(TKp01)
                  + A4 * TKp01 + S * (B1 + B2 * TKp01 + B3 * TKp01**2.))

    # Convert concentration from mL/kg to mmol/m^3
    conc = conc * rho_ref / 22.3511e-3 / 1000.0
    return conc


def O2(S, T, **kwargs):
    '''
    Solubility of O2 in sea water
    INPUT:
    S=salinity    [PSS]
    T=temperature [degree C]

    conc=solubility of O2 [mmol/m^3]

    REFERENCE:
    Hernan E. Garcia and Louis I. Gordon, 1992.
    "Oxygen solubility in seawater: Better fitting equations"
    Limnology and Oceanography, 37, pp. 1307-1312.
    '''

    # constants from Table 4 of Hamme and Emerson 2004
    conc = _garcia_gordon_polynomial(S, T,
                                     A0=5.80871,
                                     A1=3.20291,
                                     A2=4.17887,
                                     A3=5.10006,
                                     A4=-9.86643e-2,
                                     A5=3.80369,
                                     B0=-7.01577e-3,
                                     B1=-7.70028e-3,
                                     B2=-1.13864e-2,
                                     B3=-9.51519e-3,
                                     C0=-2.75915e-7)
    return _umolkg_to_mmolm3(conc)


def CO2(S, T, **kwargs):
    '''
    Solubility of CO2 in sea water
    INPUT:
    S=salinity    [PSS]
    T=temperature [degree C]

    conc=solubility of CO2 [mmol/m^3/ppm]
    Weiss & Price (1980, Mar. Chem., 8, 347-359;
    Eq 13 with table 6 values)
    '''

    salt_lim = np.max((S, salt_min))
    tk = T0_Kelvin + T
    tk100 = tk * 1e-2
    tk1002 = tk100 * tk100
    invtk = 1.0 / tk
    dlogtk = np.log(tk)
    invRtk = (1.0 / 83.1451) * invtk

    # compute CO2 solubility in mol.kg^{-1}.atm^{-1}
    arg = -162.8301 + 218.2968 / tk100 + \
        90.9241 * (dlogtk + np.log(1e-2)) - 1.47696 * tk1002 + \
        salt_lim * (0.025695 - 0.025225 * tk100 + 0.0049867 * tk1002)
    # return

    # convert to mmol/m^3/atm
    return np.exp(arg) * rho_ref * 1.0e3
