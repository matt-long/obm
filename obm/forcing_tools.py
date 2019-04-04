import numpy as np


def red_noise(mu, sigma, phi, N):
    """Generate red noise series.

    Parameters
    ----------
    mu : float
       Mean.
    sigma : float
      Variance (standard deviation).
    phi : float
      AR(1) parameter (between 0 and 1, 0 = white, 0.99 = red).
    N : int
       Number of time steps

    Returns
    -------
    Y : float
      The red noise series.
    """

    # create series
    Y = np.zeros([N])
    Zn = 0. + np.random.randn(N)
    Y[0] = sigma * np.sqrt(1 - phi**2) * Zn[0] + mu
    for i in range(1, N):
        Y[i] = (Y[i - 1] - mu) * phi + sigma * np.sqrt(1 - phi**2) * Zn[i] + mu
    return Y


def harmonic(mu, amp, phase, N, steps_per_period):
    """Compute harmonic function

    Parameters
    ----------
    mu : float
       Mean
    amp : float or interable
       Amplitude of successive harmonics
    phase : float or interable
        phase
    N : int
       Number of time steps
    steps_per_period : int
       Number of time steps per period.

    Returns
    -------
    Y : float
      The harmonic series.
    """
    if not hasattr(amp, '__len__'):
        amp = np.array([amp])

    if not hasattr(phase, '__len__'):
        phase = np.array([phase])

    n_wave_number = len(amp)
    if len(phase) != n_wave_number:
        raise ValueError('phase and amplitude need to be of same length')

    time = 2. * np.pi * np.arange(0., N, 1) / steps_per_period

    Y = np.ones(N) * mu
    for k in range(n_wave_number):
        Y = Y + amp[k] * np.sin((k + 1) * time + np.pi * phase[k])

    return Y
