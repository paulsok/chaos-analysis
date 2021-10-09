import math
import random
import numpy as np
import matplotlib.pyplot as plt


def ikeda(length=10000, level=0, mu=0.9, discard=1000):
    """ Simulate the Ikeda map described in Hammel et al (1985), "Global dynamical behavior
        of the optical field in a ring cavity", based on earlier work in Ikeda (1979),
        "Multiple-valued stationary state and its instability of the transmitted light
        by a ring cavity system".

    Parameters
    ----------
    length : int
        Length of the time series.
    level : float
        The amplitude of white noise to add to the final signal.
    mu : float
        Constant.
    discard : int
        Number of steps to discard in order to eliminate transients.

    Returns
    -------
    x : array
        Array containing the time series.
    y : array
        Array containing the time series.
    """
    x = np.zeros(length + discard)
    y = np.zeros(length + discard)

    x0 = random.random()
    y0 = random.random()
    t = 0.4 - 6 / (1 + x0 ** 2 + y0 ** 2)

    x[0] = 1 + mu * (x0 * math.cos(t) - y0 * math.sin(t))
    y[0] = mu * (x0 * math.sin(t) + y0 * math.cos(t))

    for i in range(1, length + discard):
        t = 0.4 - 6 / (1 + x[i-1] ** 2 + y[i-1] ** 2)
        x[i] = 1 + mu * (x[i-1] * math.cos(t) - y[i-1] * math.sin(t))
        y[i] = mu * (x[i-1] * math.sin(t) + y[i-1] * math.cos(t))

    # add white noise
    _x = x + level * np.std(x) * np.random.rand(length + discard)
    _y = y + level * np.std(y) * np.random.rand(length + discard)

    return _x[discard:], _y[discard:]


if __name__ == '__main__':
    x, y = ikeda(length=10000, level=0, mu=0.9, discard=1000)
    plt.plot(x)
    plt.plot(y)
    plt.show()
