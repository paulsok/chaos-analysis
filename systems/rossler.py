import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def roessler(length=10000, x0=None, a=0.2, b=0.2, c=5.7, step=0.001,
             sample=0.1, discard=1000):
    """Generate time series using the Rössler oscillator described (1976),
       "An equation for continuous chaos".

    Parameters
    ----------
    length : int
        Length of the time series to be generated.
    x0 : array
        Initial condition.
    a : float
        Constant.
    b : float
        Constant.
    c : float
        Constant.
    step : float
        Step size of integration.
    sample : int
        Sampling step.
    discard : int
        Number of samples to discard in order to eliminate transients.

    Returns
    -------
    x : ndarray shape (length, 3)
        Array containing points in phase space.
    """
    _roessler = lambda x, t: [-(x[1] + x[2]),
                              x[0] + a * x[1],
                              b + x[2] * (x[0] - c)]

    sample = int(sample/step)

    t = np.linspace(0, (sample * (length + discard)) * step,
                    sample * (length + discard))

    if not x0:
        x0 = (-9.0, 0.0, 0.0) + 0.25 * (-1 + 2 * np.random.random(3))

    return odeint(_roessler, x0, t)[discard * sample::sample]


if __name__ == '__main__':
    time_series = roessler(length=10000, x0=None, a=0.2, b=0.2, c=5.7,
                           step=0.001, sample=0.1, discard=1000)
    plt.plot(time_series)
    plt.show()
