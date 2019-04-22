import numpy as np
import xarray as xr

from .box_model_base import box_model
from . import co2calc
from . import schmidt
from . import solubility


def piston_velocity(fice, u10sq, temp, sc):
    # (m/s s^2/m^2) from (cm/hr s^2/m^2) in Wannikhof 1992
    xkw_coef = 0.31 / 3.6e5
    return (1. - fice) * xkw_coef * u10sq * ((sc / 660.0) ** -0.5)


class parameters(object):
    def __init__(self, **kwargs):
        self.ratio_O22C = -170. / 117.
        self.ratio_P2C = 1. / 117.
        self.ratio_N2C = 16. / 117.
        self.xkw_coef = 0.31 / 3.6e5

        for key, val in kwargs.items():
            if key not in self.__dict__:
                raise ValueError(f'unknown parameter {key}')
            self.__dict__[key] = val


class surface_mixed_layer(box_model):
    '''A box model.'''

    def __init__(self, time_step=86400., **kwargs):
        '''Initialize model.'''

        super(surface_mixed_layer, self).__init__(**kwargs)
        self.restore_timescale_years = 0.
        self.restore_DIC_concentration = 2200.

        self.parm = parameters(**kwargs)

        self.dt = time_step
        self.boxes = ['surface']
        self.tracers = ['DIC', 'O2']
        self._allocate_state()

    def _init_diags(self):
        diag_list = ['pCO2', 'stf_CO2', 'stf_O2', 'O2sat', 'restore_DIC']

        self.diag_values = {k: None for k in diag_list}
        self.diag_values.update({k: None for k in self.forcing.data_vars})

        self.diag_definitions = {}
        for v in diag_list:
            self.diag_definitions[v] = {'dims': ('time',)}

        for v in self.forcing.data_vars:
            self.diag_definitions[v] = {'dims': ('time',)}
            self.diag_definitions[v]['attrs'] = self.forcing[v].attrs

        self.diag_definitions['pCO2']['attrs'] = {'long_name': 'pCO$_2$',
                                                  'units': 'ppm'}
        self.diag_definitions['stf_CO2']['attrs'] = {'long_name': 'Air-sea CO$_2$ flux',
                                                     'units': 'mol m$^{-2}$ yr$^{-1}$'}

        self.diag_definitions['stf_O2']['attrs'] = {'long_name': 'Air-sea O$_2$ flux',
                                                    'units': 'mol m$^{-2}$ yr$^{-1}$'}

        self.diag_definitions['O2sat']['attrs'] = {'long_name': 'O$_2$ saturation',
                                                   'units': 'mol m$^{-3}$'}

        self.diag_definitions['restore_DIC']['attrs'] = {'long_name': 'DIC restoring tendency',
                                                         'units': 'mol m$^{-2}$ yr$^{-1}$'}

    def init_forcing(self, nday, dt, **kwargs):

        forcing_values = {}
        forcing_values['U10'] = kwargs.pop('U10', 7.5)
        forcing_values['SALT'] = kwargs.pop('SALT', 34.7)
        forcing_values['TEMP'] = kwargs.pop('TEMP', 15.)
        forcing_values['Patm'] = kwargs.pop('Patm', 1.)
        forcing_values['ALK'] = kwargs.pop('ALK', 2300.)
        forcing_values['ICE_FRAC'] = kwargs.pop('ICE_FRAC', 0.)
        forcing_values['h'] = kwargs.pop('h', 100.)
        forcing_values['NCP'] = kwargs.pop('NCP', 2.)
        forcing_values['XCO2atm'] = kwargs.pop('XCO2atm', 284.7)

        if kwargs:
            raise ValueError(f'unknown keyword arg(s): {kwargs}')

        attrs = {}
        attrs['U10'] = {'units': 'm/s', 'long_name': 'Wind speed'}
        attrs['SALT'] = {'units': 'psu', 'long_name': 'Salinity'}
        attrs['TEMP'] = {'units': 'deg C', 'long_name': 'Temperature'}
        attrs['Patm'] = {'units': 'atm', 'long_name': 'Atmospheric pressure'}
        attrs['ALK'] = {'units': 'mmol/m^3', 'long_name': 'Alkalinity'}
        attrs['ICE_FRAC'] = {'units': '', 'long_name': 'Ice fraction'}
        attrs['h'] = {'units': 'm', 'long_name': 'MLD'}
        attrs['NCP'] = {'units': 'mol/m^2/yr', 'long_name': 'NCP'}
        attrs['XCO2atm'] = {'units': 'ppm', 'long_name': 'CO$_2$$^{atm}$'}

        time = np.arange(0., nday + dt, dt)
        nt = len(time)

        forcing = xr.Dataset({'time': time})
        for v in forcing_values.keys():
            if np.isscalar(forcing_values[v]):
                forcing[v] = xr.DataArray(forcing_values[v] * np.ones(nt),
                                          dims=('time'),
                                          attrs=attrs[v])
            else:
                values = np.concatenate(
                    ([forcing_values[v][0]], forcing_values[v][:]))
                forcing[v] = xr.DataArray(values,
                                          dims=('time'),
                                          attrs=attrs[v])

        return forcing

    def compute_tendencies(self, t, state, return_diags=False):

        # local variables
        ind = self.ind
        DIC = state[ind['DIC']]
        O2 = state[ind['O2']]

        # local forcing variables
        forcing_t = self.interp_forcing(t)

        ALK = forcing_t['ALK'].values
        TEMP = forcing_t['TEMP'].values
        SALT = forcing_t['SALT'].values
        ICE_FRAC = forcing_t['ICE_FRAC'].values
        U10 = forcing_t['U10'].values
        Patm = forcing_t['Patm'].values
        Xco2atm = forcing_t['XCO2atm'].values
        h = forcing_t['h'].values

        # mol/m^2/yr --> mmol/m^2/s
        NCP_dic = forcing_t['NCP'].values * 1e3 / self.const.spy
        NCP_o2 = self.parm.ratio_O22C * NCP_dic

        # update carbonate system
        co2aq, hco3, co3 = co2calc.co2sys_from_dic_alk(S=SALT, T=TEMP,
                                                       DIC=DIC, ALK=ALK)

        # gas exchange
        xkw = (1. - ICE_FRAC) * self.parm.xkw_coef * U10**2

        sc_co2 = schmidt.CO2(SALT, TEMP)
        k_co2 = xkw * ((sc_co2 / 660.0) ** -0.5)  # m/s
        co2sol = solubility.CO2(SALT, TEMP)      # mmol/m^3/atm
        co2sat = co2sol * Patm * Xco2atm * 1e-6  # mmol/m^3

        sc_o2 = schmidt.O2(SALT, TEMP)
        k_o2 = xkw * ((sc_o2 / 660.0) ** -0.5)  # m/s
        o2sol = solubility.O2(SALT, TEMP)      # mmol/m^3/atm
        o2sat = o2sol * Patm                   # mmol/m^3

        stf_co2 = k_co2 * (co2sat - co2aq)     # mmol/m^2/s
        stf_o2 = k_o2 * (o2sat - O2)           # mmol/m^2/s

        # physics
        restore_DIC = 0.
        if self.restore_timescale_years != 0.:
            restore_DIC = ((self.restore_DIC_concentration - DIC) /
                           (self.restore_timescale_years * self.const.spy))  # mmol/m^3/s


        if return_diags:
            self.diag_values['pCO2'] = 1.e6 * co2aq / co2sol   # ppm
            self.diag_values['stf_CO2'] = stf_co2 * 1e-3 * \
                self.const.spy  # mmol/m^2/s --> mol/m^2/yr

            self.diag_values['stf_O2'] = stf_o2 * 1e-3 * \
                self.const.spy  # mmol/m^2/s --> mol/m^2/yr
            self.diag_values['O2sat'] = o2sat

            self.diag_values['restore_DIC'] = restore_DIC * h * \
                1e-3 * self.const.spy  # mmol/m^3/s --> mol/m^2/yr

            self.diag_values.update({k: da.values for k, da in forcing_t.items()})
            return self.diag_values

        # accumulate tendencies
        self.dcdt[ind['DIC']] = (stf_co2 - NCP_dic) / h + restore_DIC
        self.dcdt[ind['O2']] = (stf_o2 - NCP_o2) / h


        return self.dcdt
