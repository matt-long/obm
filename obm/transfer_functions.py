import numpy as np

from .constants import constants


def michaelis_menten(X, K):
    return X / (X + K)


def Q10(T, Q_10=2., Tref=30.):
    T0_Kelvin = constants.T0_Kelvin
    return Q_10**(((T + T0_Kelvin) - (Tref + T0_Kelvin)) / 10.)


def arrhenius(T, Ea=0.5, Tref=30.):
    T0_Kelvin = constants.T0_Kelvin
    K_Boltz = constants.K_Boltz
    return np.exp(
        -Ea * (Tref - T) /
        (K_Boltz * (T + T0_Kelvin) * (Tref + T0_Kelvin))
    )
