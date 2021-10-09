import random
import numpy as np
import matplotlib.pyplot as plt


def henon(length=10000, level=0, a=1.25, b=0.3, discard=1000):
    """Simulate the Henon map described in Henon (1976),
       "A two-dimensional mapping with a strange attractor".

    Parameters
    ----------
    length : int
        Length of the time series.
    level : float
        The amplitude of white noise to add to the final signal.
    a : float
        Constant.
    b : float
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
    x_0 = 0.1 * random.uniform(-1, 1)
    y_0 = 0.1 * random.uniform(-1, 1)

    x = np.zeros(length + discard)
    y = np.zeros(length + discard)

    x[0] = 1 - a * x_0 ** 2 + b * y_0
    y[0] = b * x_0

    for i in range(1, length + discard):
        x[i] = 1 - a * x[i-1] ** 2 + y[i-1]
        y[i] = b * x[i-1]

    # add white noise
    _x = x + level * np.std(x) * np.random.rand(length + discard)
    _y = y + level * np.std(y) * np.random.rand(length + discard)

    return _x[discard:], _y[discard:]


if __name__ == '__main__':
    _x, _y = henon(length=10000, level=0, a=1.25, b=0.3, discard=1000)
    plt.plot(_x)
    plt.plot(_y)
    plt.show()
