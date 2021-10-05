import random
import numpy as np
import matplotlib.pyplot as plt


def feritas(length=10000, level=0.2, discard=1000):
    """Simulate the nonlinear stochastic map
       described in Freitas et al (2009).

    Parameters
    ----------
    length : int
        Length of the time series.
    level : float
        The amplitude of white noise to add to the final signal.
    discard : int
        Number of samples to discard in order to eliminate transients.

    Returns
    -------
    x : array
        Array containing the time series.
    """
    x = np.zeros(length + discard)
    v = np.random.rand(length + discard)

    x[0], x[1] = random.random(), random.random()

    for i in range(2, length + discard):
        x[i] = 3 * v[i-1] + 4 * v[i-2] * (1-v[i-1])

    # add noise to the data
    _x = x.conjugate() + level * np.std(x.conjugate()) * np.random.rand(length + discard)

    return _x[discard:]


if __name__ == '__main__':
    time_series = feritas(length=10000, level=0.2, discard=1000)
    plt.plot(time_series)
    plt.show()
