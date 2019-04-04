import numpy as np

from .box_model_base import box_model
from .transfer_functions import michaelis_menten, arrhenius


class simple_npd(box_model):
    '''A box model.'''

    def __init__(self, **kwargs):
        '''Initialize model.'''

        super(simple_npd, self).__init__(**kwargs)

        parm_dict = {'mu_max': 3.5,
                     'K_N': 0.5,
                     'alpha': 0.3,
                     'g_max': 3.,
                     'gamma': 0.3,
                     'K_P': 1.5,
                     'm_P': 0.01,
                     'm_PP': 0.05,
                     'm_Z': 0.1,
                     'm_ZZ': 0.1,
                     'Ea_P': 0.5,
                     'Ea_Z': 1.,
                     'Tref_P': 15.,
                     'Tref_Z': 15.}

        tau_vert_flux = kwargs.pop('tau_bio_day', 0.5)
        h_surf = kwargs.pop('h_surf', 100.)
        state_deep = kwargs.pop('state_init', None)

        parm_dict_in = kwargs.pop('parameters', {})
        parm_dict.update(parm_dict_in)

        rates_per_day = ['mu_max', 'g_max', 'm_P', 'm_PP', 'm_Z', 'm_ZZ']
        for key, val in parm_dict.items():
            if key in rates_per_day:
                self.__dict__[key] = val / 86400.
            else:
                self.__dict__[key] = val

        #self.dt = 3600. * 2.
        self.boxes = ['surface']
        self.tracers = ['N', 'P', 'Z']
        self._allocate_state()

    def compute_tendencies(self, t, state):

        N = state[:, self.ind['N']]
        P = state[:, self.ind['P']]
        Z = state[:, self.ind['Z']]

        mu_max = self.mu_max
        K_N = self.K_N
        alpha = self.alpha
        g_max = self.g_max
        gamma = self.gamma
        K_P = self.K_P
        m_P = self.m_P
        m_PP = self.m_PP
        m_Z = self.m_Z
        m_ZZ = self.m_ZZ

        Tref_P = self.Tref_P
        Tref_Z = self.Tref_Z
        Ea_P = self.Ea_P
        Ea_Z = self.Ea_Z

        G = michaelis_menten(P, K_P)
        L_N = michaelis_menten(N, K_N)
        L_T_P = arrhenius(self.forcing_t.TEMP.values, Ea=Ea_P, Tref=Tref_P)
        L_I = 1.

        L_T_Z = arrhenius(self.forcing_t.TEMP.values, Ea=Ea_Z, Tref=Tref_Z)

        growth = mu_max * L_N * L_T_P * P
        graze = g_max * G * L_T_Z * Z
        mortality_P = m_P * P + m_PP * P ** 2.
        mortality_Z = m_Z * Z + m_ZZ * Z ** 2.

        self.dcdt[:, P] = growth - graze - mortality_P
        self.dcdt[:, Z] = gamma * graze - mortality_Z
        self.dcdt[:, N] = (-1.0) * growth + (1. - gamma) * \
            graze + mortality_P + mortality_Z

        # check conservation
        np.testing.assert_allclose(self.dcdt.sum(axis=1), 0., atol=1e-7,
                                   rtol=1e-7)

        return self.dcdt, self.diag_values
