import numpy as np
import xarray as xr
from scipy.optimize import fsolve

import matplotlib.pyplot as plt

from .constants import constants

def rk4(dfdt, dt, t, y, kwargs={}):
    '''
    4th order Runge-Kutta
    '''
    dydt1, diag1 = dfdt( t, y, **kwargs )
    dydt2, diag2 = dfdt( t + dt / 2., y + dt*dydt1 / 2., **kwargs)
    dydt3, diag3 = dfdt( t + dt / 2, y + dt*dydt2 / 2., **kwargs )
    dydt4, diag4 = dfdt( t + dt, y + dt*dydt3, **kwargs )

    y_next_t = y + dt * (dydt1 + 2. * dydt2 + 2. * dydt3 + dydt4) / 6.

    diag_t = diag1
    for key in diag1.keys():
        diag_t[key] = (diag1[key] + 2. * diag2[key] + 2. * diag3[key]
                       + diag4[key]) / 6.

    return y_next_t,diag_t


class box_model(object):
    '''A box model.'''

    const = constants

    def __init__(self, **kwargs):
        '''Initialize model.'''
        self.user_time_units = kwargs.pop('time_units', 'day')

        if 'day' in self.user_time_units:
            self.convert_model_to_user_time = 1./self.const.spd
        elif 'year' in self.user_time_units:
            self.convert_model_to_user_time = 1./self.const.spy
        else:
            raise ValueError('unknown forcing time units')

        self.forcing_t = xr.Dataset()
        self.dt = 3600.

    def _allocate_state(self, nboxes, ntracers):
        self.state = np.empty((nboxes, ntracers))
        self.dcdt = np.zeros((nboxes, ntracers))

    def reset(self):
        pass

    def _init_diags(self):
        self.diag = {}

    def compute_tendencies(self, t, state):
        raise NotImplementedError('subclass must implement')

    def interp_forcing(self, t, forcing):
        '''Interpolate forcing dataset at time = t.'''
        if forcing:
            self.forcing_t = forcing.interp(
                {'time': self.convert_model_to_user_time * t})

    def _init_output(self, time):
        user_time = time[1:] * self.convert_model_to_user_time
        nt_output = len(user_time)
        nboxes = len(self.boxes)

        time_coord = xr.DataArray(user_time, dims=('time'),
                                     attrs={'units': self.user_time_units})
        box_coord = xr.DataArray(self.boxes, dims=('box'))

        output = xr.Dataset(coords={'time': time_coord, 'box': box_coord})

        for tracer in self.tracers:
            output[tracer] = xr.DataArray(np.empty((nt_output, nboxes)),
                                          dims=('time', 'box'),
                                          attrs={'units': 'mmol/m$^3$',
                                                 'long_name': tracer},
                                          coords={'time': time_coord, 'box': box_coord})

        for var in self.forcing_t:
            output[var] = xr.DataArray(np.empty(nt_output),
                                       dims=('time'),
                                       coords={'time': time_coord})

        for key, val in self.diag.items():
            output[key] = xr.DataArray(**val)

        return output

    def _init(self, state_init, init_option=None, **kwargs):
        raise NotImplementedError('init not implemented.')

    def _fsolve_equilibrium(self, state_init, **kwargs):
        """Find cyclostationary solution."""
        dstate_out = np.zeros((len(self.tracers)))

        def wrap_model(state_in):
            out = self.run(start=kwargs['start'], stop=kwargs['stop'],
                           forcing=kwargs['forcing'], state_init=state_in)
            dstate_out = np.sum((self.state-state_in)**2, axis=0)
            return dstate_out

        return fsolve(wrap_model, state_init)

    def run(self, start, stop, forcing=xr.Dataset(), state_init=None,
            use_init_method=False):
        """Integrate the model in time.

        Parameters
        ----------
        start : numeric
           Starting value for time.

        stop : numeric
           Final time value.

        forcing : xarray.Dataset
           Forcing data defined with `time` coordinate.

        state_init : numpy.array
           Initial state with dimensions [nboxes, ntracers]

        use_init_method : bool
            If true, pass state_init to "init" method of subclass.

        Returns
        -------
        out : xarray.Dataset
           Model solution.
        """

        #-- pointers for local variable
        ind = self.ind
        dt = self.dt
        tend_func = self.compute_tendencies

        #-- time axis
        time = np.arange(start / self.convert_model_to_user_time,
                         stop / self.convert_model_to_user_time + dt, dt)
        nt = len(time)

        #-- initialize
        self.interp_forcing(time[0], forcing)
        if use_init_method:
            state_init = self._init(state_init=state_init,
                                    start=start,
                                    stop=stop,
                                    forcing=forcing)
            print(state_init)
        self.state[:] = np.array(state_init)

        self._init_diags()
        output = self._init_output(time)

        #-- begin timestepping
        for l in range(1, nt):

            #-- interpolate forcing
            self.interp_forcing(time[l], forcing)

            #-- integrated
            self.state, diag_t = rk4(tend_func, dt, time[l], self.state)

            self.reset()

            #-- save output
            for tracer in self.tracers:
                output[tracer].data[l-1, :] = self.state[:, ind[tracer]]

            for var in self.forcing_t:
                output[var].data[l-1] = self.forcing_t[var]

            for key, val in diag_t.items():
                output[key].data[l-1] = val


        return output

    def plot(self, out):
        for tracer in self.tracers:
            plt.figure()
            out[tracer].plot()
