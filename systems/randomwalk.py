import numpy as np
import matplotlib.pyplot as plt


def randomwalk(length=10000, level=0, discard=1000):
    """Generate time series from the random walk
       (a stochastic, non-stationary signal).

    Parameters
    ----------
    length : int
        Length of the time series.
    level : float
        The amplitude of white noise to add to the final signal.
    discard : int
        Number of steps to discard in order to eliminate transients.

    Returns
    -------
    x : array
        Array containing the time series.
    """
    x = np.zeros(length + discard)
    x[0] = np.random.random_sample()

    for i in range(1, length + discard):
        x[i] = x[i-1] + np.random.normal(0, 1, 1)

    # add white noise
    _x = x + level * np.std(x) * np.random.rand(length + discard)

    return _x[discard:]


if __name__ == '__main__':
    time_series = randomwalk(length=10000, level=0, discard=1000)
    plt.plot(time_series)
    plt.show()
