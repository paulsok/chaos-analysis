import random
import numpy as np
import matplotlib.pyplot as plt


def gen_henon(length=10000, level=0,  a=1.76, b=0.1, discard=1000):
    """Simulate the generalized Henon map described in Richter (2002),
       "The generalized Henon maps: examples for higher-dimensional chaos".

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
    """
    x = np.zeros(length + discard)

    x[0], x[1] = random.random(), random.random()

    for i in range(2, length + discard - 1):
        x[i+1] = a - x[i-1] ** 2 - b * x[i-2]

    # add white noise
    _x = x + level * np.std(x) * np.random.rand(length + discard)

    return _x[discard:]


if __name__ == '__main__':
    time_series = gen_henon(length=10000, level=0,  a=1.76, b=0.1, discard=1000)
    plt.plot(time_series)
    plt.show()
