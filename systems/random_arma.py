import random
import numpy as np
import matplotlib.pyplot as plt


def random_arma(length=10000, level=0, p=1, theta=2, discard=1000):
    """Generate time series from the random ARMA model
       with Gaussian-distributed error.

    Parameters
    ----------
    length : int
        Length of the time series.
    level : float
        The amplitude of white noise to add to the final signal.
    p : float
        Constant.
    theta : float
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

    err = np.random.rand(length + discard)

    for i in range(1, length + discard):
        x[i] = p * x[i-1] + err[i] + err[i-1] * (1 + theta)

    # add white noise
    _x = x + level * np.std(x) * np.random.rand(length + discard)

    return _x[discard:]


if __name__ == '__main__':
    time_series = random_arma(length=10000, level=0, p=0.8, theta=1, discard=1000)
    plt.plot(time_series)
    plt.show()
