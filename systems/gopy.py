import math
import random
import numpy as np
import matplotlib.pyplot as plt


def gopy(length=10000, level=0.2, sigma=1.5, discard=1000):
    """Simulate the GOPY system described in Grebogi et al (1984),
       "Strange attractors that are not chaotic".

    Parameters
    ----------
    length : int
        Length of the time series.
    level : float
        The amplitude of white noise to add to the final signal.
    sigma : float
        Constant.
    discard : int
        Number of samples to discard in order to eliminate transients.

    Returns
    -------
    x : array
        Array containing the time series.
    """
    w = (math.sqrt(5) - 1) / 2  # golden ratio

    x = np.zeros(length + discard)
    theta = np.zeros(length + discard)

    x[0], theta[0] = random.random(), random.random()

    for i in range(1, length + discard - 1):
        x[i+1] = 2 * sigma * math.tanh(x[i - 1]) * math.cos(2 * math.pi * theta[i - 1])
        theta[i] = (theta[i-1] + w) % 1

    # add white noise
    _x = x + level * np.std(x) * np.random.rand(length + discard)

    return _x[discard:]


if __name__ == '__main__':
    time_series = gopy(length=10000, level=.2, sigma=1.5, discard=1000)
    plt.plot(time_series)
    plt.show()
