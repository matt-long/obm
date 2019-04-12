
import pytest

import numpy as np
from obm import solubility
from obm import schmidt
from obm.constants import constants


rho_ref = constants.rho_ref


@pytest.mark.parametrize('S', [35.])
@pytest.mark.parametrize('T', [24., 10.])
def test_solubility_values(S, T):

    if S == 35. and T == 24.:
        check_values = {'CO2': 28931.794148,  # mmol/m^3/atm
                        }
    elif S == 35. and T == 10.:
        check_values = {'CO2': 44298.475749,  # mmol/m^3/atm
                        }
    else:
        raise ValueError(f'no check value for S={S}, T={T}')

    for gas, val in check_values.items():
        func = getattr(solubility, gas)
        np.testing.assert_almost_equal(func(S, T), val, decimal=4)


@pytest.mark.parametrize('S', [35.])
@pytest.mark.parametrize('T', [20.])
def test_schmidt_values(S, T):
    if S == 35. and T == 20.:
        check_values = {'Ar': 462.7199524345627,  # output, not real check
                        'Ne': 301.2313844051046,  # output, not real check
                        'N2': 637.2619556776932,  # output, not real check
                        'Kr': 683.1706763572696,  # output, not real check
                        'Xe': 864.7455001107579,
                        'O2': 530.568,
                        'CO2': 665.9879999999998,
                        }
    else:
        raise ValueError(f'no check value for S={S}, T={T}')

    for gas, val in check_values.items():
        func = getattr(schmidt, gas)
        np.testing.assert_almost_equal(func(S, T), val, decimal=4)
