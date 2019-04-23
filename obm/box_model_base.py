import numpy as np
import xarray as xr

from scipy.optimize import fsolve
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

from .constants import constants


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

        self.cyclic_annual_forcing = False
        self.units_tracer = 'mmol/m$^3$'
        self.forcing = None
        self.dt = 3600.

    def _allocate_state(self):
        if not hasattr(self, 'tracers'):
            raise ValueError('"tracers" attribute is unset')

        if not hasattr(self, 'boxes'):
            raise ValueError('"boxes" attribute is unset')

        self.ntracers = len(self.tracers)
        self.nboxes = len(self.boxes)

        self.ind = {}
        for i, tracer in enumerate(self.tracers):
            self.ind[tracer] = np.arange(i * self.nboxes, i * self.nboxes + self.nboxes, 1)

        self.state = np.empty((self.nboxes*self.ntracers))
        self.dcdt = np.zeros((self.nboxes*self.ntracers))

    def reset(self):
        pass

    def _init_diags(self):
        self.diag_values = {}
        self.diag_definitions = {}

    def compute_tendencies(self, t, state):
        raise NotImplementedError('subclass must implement')

    def _compute_forcing_time(self, t):
        forcing_time = t
        if self.cyclic_annual_forcing:
            forcing_time = forcing_time - 365. * np.floor(forcing_time / 365.)

        return forcing_time * self.convert_model_to_user_time

    def interp_forcing(self, t):
        '''Interpolate forcing dataset at time = t.'''

        if self.forcing is not None:
            return self.forcing.interp({'time': self._compute_forcing_time(t)})

    def _init(self, state_init, init_option, init_file, **kwargs):
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
        dstate_out = np.zeros((self.ntracers))

        # precondition
        for i in range(3):
            state_out = self.run(t_final_days=kwargs['t_final_days'],
                                 state_init=state_init,
                                 init_option='input',
                                 forcing=kwargs['forcing'],
                                 method=kwargs['method'],
                                 dt=kwargs['dt'],
                                 rtol=kwargs['rtol'],
                                 atol=kwargs['atol'],
                                 return_only_state=True)
            state_init = state_out[-1, :]

        def wrap_model(state_in):
            state_out = self.run(t_final_days=kwargs['t_final_days'],
                                 state_init=state_init,
                                 init_option='input',
                                 forcing=kwargs['forcing'],
                                 method=kwargs['method'],
                                 dt=kwargs['dt'],
                                 rtol=kwargs['rtol'],
                                 atol=kwargs['atol'],
                                 return_only_state=True)

            for i, tracer in enumerate(self.tracers):
                state_out_tracer = state_out[-1, self.ind[tracer]]
                state_in_tracer = state_in[self.ind[tracer]]
                dstate_out[i] = np.sum((state_out_tracer - state_in_tracer)**2)

            return dstate_out

        return fsolve(wrap_model, state_init, xtol=1e-7, maxfev=2000)


    def plot(self, out):
        for tracer in self.tracers:
            plt.figure()
            out[tracer].plot()

    def run(self, t_final_days, state_init, init_option='input', init_file=None,
            dt=1., forcing=None, method='Radau',
            rtol=1e-3, atol=1e-3, return_only_state=False):
        """Integrate the model in time.

        Parameters
        ----------
        t_final_days : numeric
           Final time value in days.

        state_init : numpy.array
           Initial state with dimensions [nboxes, ntracers]

        forcing : xarray.Dataset, optional
           Forcing data defined with `time` coordinate.

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
        nt = np.int(t_final_days / dt)

        # time axis
        eval_time = np.arange(0., t_final_days + dt, dt) / self.convert_model_to_user_time

        # set forcing
        if forcing is not None:
            self.forcing = forcing

        run_kwargs = dict(t_final_days=t_final_days, dt=dt, forcing=forcing,
                          method=method, rtol=rtol, atol=atol)

        state_init = self._init(state_init=state_init,
                                init_option=init_option,
                                init_file=init_file,
                                **run_kwargs)

        self._init_diags()

        # solve the model
        soln = solve_ivp(self.compute_tendencies,
                         t_span=[eval_time[0], eval_time[-1]],
                         t_eval=eval_time,
                         y0=self.state,
                         method=method,
                         rtol=rtol, atol=atol)

        if not soln.success:
            raise Exception(soln.message)

        soln_state = soln.y.T
        soln_time = soln.t * self.convert_model_to_user_time

        if return_only_state:
            return soln_state

        time_coord = xr.DataArray(soln_time, dims=('time'),
                                  attrs={'units': 'days'})
        box_coord = xr.DataArray(self.boxes, dims=('box'))

        output = xr.Dataset(coords={'time': time_coord, 'box': box_coord})

        for i, tracer in enumerate(self.tracers):
            output[tracer] = xr.DataArray(soln_state[1:, self.ind[tracer]],
                                          dims=('time', 'box'),
                                          attrs={'units': self.units_tracer,
                                                 'long_name': tracer},
                                          coords={'time': time_coord, 'box': box_coord})

        # get diagnostic quantities by re-calling `compute_tendencies`
        for key, val in self.diag_definitions.items():
            output[key] = xr.DataArray(np.empty(nt), **val)

        for l in range(1, len(soln_time)):
            t = soln_time[l] / self.convert_model_to_user_time
            diag_t = self.compute_tendencies(t, soln_state[l, :], return_diags=True)
            for key, val in diag_t.items():
                output[key].data[l] = np.array(val)

        return output
