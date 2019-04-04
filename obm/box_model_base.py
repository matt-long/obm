import numpy as np
import xarray as xr
from scipy.optimize import fsolve

import matplotlib.pyplot as plt

from .constants import constants


def rk4(dfdt, dt, t, y, kwargs={}):
    '''
    4th order Runge-Kutta
    '''
    dydt1, diag1 = dfdt(t, y, **kwargs)
    dydt2, diag2 = dfdt(t + dt / 2., y + dt * dydt1 / 2., **kwargs)
    dydt3, diag3 = dfdt(t + dt / 2, y + dt * dydt2 / 2., **kwargs)
    dydt4, diag4 = dfdt(t + dt, y + dt * dydt3, **kwargs)

    y_next_t = y + dt * (dydt1 + 2. * dydt2 + 2. * dydt3 + dydt4) / 6.

    diag_t = diag1
    for key in diag1.keys():
        diag_t[key] = (diag1[key] + 2. * diag2[key] + 2. * diag3[key]
                       + diag4[key]) / 6.

    return y_next_t, diag_t


class box_model(object):
    '''A box model.'''

    const = constants

    def __init__(self, **kwargs):
        '''Initialize model.'''
        self.user_time_units = kwargs.pop('time_units', 'day')

        if 'day' in self.user_time_units:
            self.convert_model_to_user_time = 1. / self.const.spd
        elif 'year' in self.user_time_units:
            self.convert_model_to_user_time = 1. / self.const.spy
        else:
            raise ValueError('unknown forcing time units')

        self.forcing_t = xr.Dataset()
        self.dt = 3600.

    def _allocate_state(self):
        if not hasattr(self, 'tracers'):
            raise ValueError('"tracers" attribute is unset')

        if not hasattr(self, 'boxes'):
            raise ValueError('"boxes" attribute is unset')

        nboxes = len(self.boxes)
        ntracers = len(self.tracers)

        self.ind = {tracer: int(i) for i, tracer in enumerate(self.tracers)}
        self.state = np.empty((nboxes, ntracers))
        self.dcdt = np.zeros((nboxes, ntracers))

    def reset(self):
        pass

    def _init_diags(self):
        self.diag_values = {}
        self.diag_definitions = {}

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

        for var, da in self.forcing_t.data_vars.items():
            attrs = da.attrs
            output[var] = xr.DataArray(np.empty(nt_output),
                                       dims=('time'),
                                       coords={'time': time_coord},
                                       attrs=attrs)

        for key, val in self.diag_definitions.items():
            output[key] = xr.DataArray(np.empty(nt_output), **val)

        return output

    def _init(self, state_init, init_option='input', init_file=None, **kwargs):
        """Initialize the model."""

        if init_option == 'input':
            if state_init is None:
                raise ValueError(
                    'state_init is `None`; cannot initialize model.')
            self.state[:] = np.array(state_init)

        elif init_option == 'fsolve':
            if state_init is None:
                state_init = np.ones(self.state.shape)
            self.state[:] = self._fsolve_equilibrium(state_init, **kwargs)

            if init_file is not None:
                np.save(init_file, self.state)

        elif init_option == 'file':
            if init_file is None:
                raise ValueError('must specify `init_file`')
            self.state[:] = np.load(init_file)

        else:
            raise ValueError('unknown init option')

    def _fsolve_equilibrium(self, state_init, **kwargs):
        """Find cyclostationary solution."""
        dstate_out = np.zeros((len(self.tracers)))

        def wrap_model(state_in):
            out = self.run(time_stop=kwargs['time_stop'],
                           forcing=kwargs['forcing'],
                           state_init=state_in,
                           init_option='input')
            dstate_out = np.sum((self.state - state_in)**2, axis=0)
            return dstate_out

        return fsolve(wrap_model, state_init, xtol=1e-5, maxfev=100)

    def run(self, time_stop, forcing=xr.Dataset(), state_init=None,
            init_option='input', init_file=None):
        """Integrate the model in time.

        Parameters
        ----------
        time_stop : numeric
           Final time value.

        forcing : xarray.Dataset
           Forcing data defined with `time` coordinate.

        state_init : numpy.array
           Initial state with dimensions [nboxes, ntracers]

        init_option : string, optional [default='input']
            Initialization method:
              - 'input': use the `state_init` as passed in
              - 'fsolve': use scipy.optimize.fsolve to compute cyclostationary
                          equilibrium, where `state_init` provides an initial
                          guess.

        init_file : string, optional [default=None]
            File name from which to read initial state or to which to write
            initial state following 'fsolve' spinup.

        Returns
        -------
        out : xarray.Dataset
           Model solution.
        """

        # pointers for local variable
        ind = self.ind
        dt = self.dt
        tend_func = self.compute_tendencies

        # time axis
        time = np.arange(
            0.,
            time_stop /
            self.convert_model_to_user_time +
            dt,
            dt)
        nt = len(time)

        # initialize
        self.interp_forcing(time[0], forcing)
        self._init(state_init=state_init,
                   init_option=init_option,
                   init_file=init_file,
                   time_stop=time_stop,
                   forcing=forcing)

        self._init_diags()
        output = self._init_output(time)

        # begin timestepping
        for l in range(1, nt):

            # -- interpolate forcing
            self.interp_forcing(time[l], forcing)

            # -- integrated
            self.state, diag_t = rk4(tend_func, dt, time[l], self.state)

            self.reset()

            # -- save output
            for tracer in self.tracers:
                output[tracer].data[l - 1, :] = self.state[:, ind[tracer]]

            for var in self.forcing_t:
                output[var].data[l - 1] = self.forcing_t[var]

            for key, val in diag_t.items():
                output[key].data[l - 1] = val

        return output

    def plot(self, out):
        for tracer in self.tracers:
            plt.figure()
            out[tracer].plot()
