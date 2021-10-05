import math
import random
import numpy as np
import matplotlib.pyplot as plt


def noise_sine(length=10000, level=0.2, mu=2.4, discard=1000):
    """Simulate the noise-driven sine map described in Freitas et al (2009),
       "Failure in distinguishing colored noise from chaos using the 'noise
       titration' technique".

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
    """
    x = np.zeros(length + discard)
    x[0] = random.random()

    for i in range(1, length + discard):
        q = random.random()
        if q <= 0.01:
            y = 1
        else:
            y = 0
        x[i] = mu * math.sin(x[i-1]) + y * (4 * random.random() - 2)

    # add white noise
    _x = x + level * np.std(x) * np.random.rand(length + discard)

    return _x[discard:]


if __name__ == '__main__':
    time_series = noise_sine(length=10000, level=0.2, mu=2.4, discard=1000)
    plt.plot(time_series)
    plt.show()
