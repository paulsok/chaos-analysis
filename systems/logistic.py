import random
import numpy as np
import matplotlib.pyplot as plt


def logistic(length=10000, level=0, r=3.5, discard=1000):
    """Simulate the logistic map described in May (1976),
       "Simple mathematical models with very complicated dynamics".

    Parameters
    ----------
    length : int
        Length of the time series.
    level : float
        The amplitude of white noise to add to the final signal.
    r : float
        Constant.
    discard : int
        Number of steps to discard in order to eliminate transients.

    Returns
    -------
    x : array
        Array containing the time series.
    """
    y0 = random.random()
    x = np.zeros(length + discard)

    x[0] = r * y0 * (1 - y0)

    for i in range(1, length + discard):
        x[i] = r * x[i-1] * (1 - x[i-1])

    # add white noise
    _x = x + level * np.std(x) * np.random.rand(length + discard)

    return _x[discard:]


if __name__ == '__main__':
    time_series = logistic(length=10000, level=0, r=3.5, discard=1000)
    plt.plot(time_series)
    plt.show()
