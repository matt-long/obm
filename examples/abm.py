class abm(object):

    def __init__(self):
        mass_air = np.array([10., 50., 25.])

        tau_exchange = partial(
            obm.forcing_tools.harmonic,
            phase=-0.5,
            N=365,
            steps_per_period=365)

        time = np.arange(0, 365., 1)

        def k01(t): return np.interp(
            t -
            365. *
            np.floor(
                t /
                365.),
            time,
            tau_exchange(
                mu=0.1,
                amp=-
                0.1))

        def k12(t): return np.interp(
            t -
            365. *
            np.floor(
                t /
                365.),
            time,
            tau_exchange(
                mu=0.5,
                amp=-
                0.5))

        def k03(t): return np.interp(
            t -
            365. *
            np.floor(
                t /
                365.),
            time,
            tau_exchange(
                mu=0.1,
                amp=-
                0.1))

        def k13(t): return np.interp(
            t -
            365. *
            np.floor(
                t /
                365.),
            time,
            tau_exchange(
                mu=0.5,
                amp=-
                0.5))

        def k23(t): return np.interp(
            t -
            365. *
            np.floor(
                t /
                365.),
            time,
            tau_exchange(
                mu=0.5,
                amp=-
                0.5))

        M3 = 5

        def dM(M_t): return np.hstack(
            (np.vstack((M_t - M_t[0], M_t - M_t[1], M_t - M_t[2])), (M3 - M_t)[:, None]))

        def PSI(t): return np.array([[0., k01(t), 0., k03(t)],
                                     [k01(t), 0., k12(t), k13(t)],
                                     [0., k12(t), 0., k23(t)]])

        stf_CO2 = ds.stf_CO2.loc[(ds.thermal_forcing == 0) & (
            ds.biological_forcing == 4)].values

        def Fsrf(t): return np.interp(
            t -
            365. *
            np.floor(
                t /
                365.),
            time,
            np.squeeze(stf_CO2))
