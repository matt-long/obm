from functools import partial
import copy

import numpy as np
import xarray as xr

from scipy.integrate import solve_ivp

from . import forcing_tools


class o2co2(object):

    def __init__(self, **kwargs):

        self.boxes = ['BL', 'MT', 'UT']
        self.tracers = ['CO2', 'O2']
        self.mass_air = np.array([5, 20, 15])
        self.ntracers = len(self.tracers)
        self.nboxes = len(self.boxes)

        tau_parms = {'BL-MT': dict(mu=0.1, amp=-0.1),
                     'MT-UT': dict(mu=0.5, amp=-0.5),
                     'BL-boundary': dict(mu=0.1, amp=-0.1),
                     'MT-boundary': dict(mu=0.5, amp=-0.5),
                     'UT-boundary': dict(mu=0.5, amp=-0.5)}

        surface_flux = kwargs.pop('surface_flux')
        self.boundary_condition = np.ones(self.nboxes * self.ntracers) * 5.
        self.dMdt = np.zeros((self.nboxes * self.ntracers))

        self._init_surface_flux(surface_flux)
        self._init_exchange_matrix(tau_parms)

    def _init_surface_flux(self, surface_flux):
        self.Fsrf = {}

        Fsrf = np.empty((self.ntracers * self.nboxes), dtype=object)
        data = np.empty((self.nboxes, self.ntracers, 365))
        for i, tracer in enumerate(self.tracers):
            ind = np.arange(i * self.nboxes, i * self.nboxes + self.nboxes, 1)
            # fill missing fluxes with 0.
            for j, box in enumerate(self.boxes):
                if box not in surface_flux[tracer]:
                    data[j, i, :] = 0.
                else:
                    data[j, i, :] = np.array(surface_flux[tracer][box])
                Fsrf[ind[j]] = partial(
                    self._interp_forcing, data=data[j, i, :])

        self.Fsrf = Fsrf

    def _init_exchange_matrix(self, tau_parms):

        tau = np.empty((self.nboxes, self.nboxes + 1), dtype=object)

        boxes_j = self.boxes + ['boundary']

        # loop over boexes
        for i in range(self.nboxes):
            boxi = self.boxes[i]
            # loop over boxes + boundary_condition
            for j in range(self.nboxes + 1):
                boxj = boxes_j[j]
                key = f'{boxi}-{boxj}'

                if key not in tau_parms:
                    key = self._mirror_key(key)

                if key not in tau_parms or i == j:
                    tau[i, j] = self._zero
                else:
                    tau[i, j] = partial(
                        self._exchange_forcing, **tau_parms[key])

        self.psi = lambda t: np.array([[tau[i, j](t)
                                        for j in range(self.nboxes + 1)]
                                       for i in range(self.nboxes)])

    def _exchange_forcing(self, t, **kwargs):
        func_tau_exchange = partial(forcing_tools.harmonic,
                                    phase=-0.5, N=365, steps_per_period=365)
        return self._interp_forcing(t, data=func_tau_exchange(**kwargs))

    def _interp_forcing(self, t, data):
        return np.interp(self._cyclic_t(t), np.arange(0, 365., 1), data)

    def _cyclic_t(self, t):
        return t - 365. * np.floor(t / 365.)

    def _mirror_key(self, key):
        return '-'.join(key.split('-')[::-1])

    def _zero(self, t):
        return 0.

    def _init_diags(self):
        self.diag_values = {}
        self.diag_definitions = {}

    def compute_tendencies(self, t, state):

        self.dMdt[:] = 0.
        for i, tracer in enumerate(self.tracers):
            # index into these tracers
            ind = np.arange(i * self.nboxes, i * self.nboxes + self.nboxes, 1)

            # state and boundary conditions
            bc = self.boundary_condition[ind]
            state_bc = np.concatenate((np.tile(state[ind], (self.nboxes, 1)),
                                       bc[:, None]), axis=1)

            # transport
            flux_out = np.sum(
                self.psi(t) * np.tile(state[ind][:, None], (1, self.nboxes + 1)), axis=1)
            flux_in = np.sum(self.psi(t) * state_bc, axis=1)

            # surface flux
            Fsrf = np.array([self.Fsrf[ind[j]](t) for j in range(self.nboxes)])

            # assemble tendency
            self.dMdt[ind] = (Fsrf + flux_in - flux_out) / self.mass_air

        return self.dMdt

    def run(self, t_final_days, state_init, dt=1.,
            method='Radau', rtol=1e-3, atol=1e-6):

        nt = np.int(t_final_days / dt)

        # time axis
        eval_time = np.arange(0., t_final_days + dt, dt)

        # solve the model
        soln = solve_ivp(self.compute_tendencies, t_span=[eval_time[0], eval_time[-1]], y0=state_init,
                         method=method, t_eval=eval_time, rtol=rtol, atol=atol)

        if not soln.success:
            raise Exception(soln.message)
        soln_state = soln.y.T
        soln_time = soln.t

        time_coord = xr.DataArray(soln_time, dims=('time'),
                                  attrs={'units': 'days'})
        box_coord = xr.DataArray(self.boxes, dims=('box'))

        output = xr.Dataset(coords={'time': time_coord, 'box': box_coord})

        for i, tracer in enumerate(self.tracers):
            ind = np.arange(i * self.nboxes, i * self.nboxes + self.nboxes, 1)
            output[tracer] = xr.DataArray(soln_state[:, ind],
                                          dims=('time', 'box'),
                                          attrs={'units': 'kg/kg',
                                                 'long_name': tracer},
                                          coords={'time': time_coord, 'box': box_coord})

        return output
