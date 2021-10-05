import math
import numpy as np
import matplotlib.pyplot as plt


def cyclostationary(length=10000, tau=500, t=100):
    """ Simulate the cyclostationary autoregressive process described in Timmer,
        1998, "Power of surrogate data testing with respect to nonstationarity.

    Parameters
    ----------
    length : int
        Length of the time series to be generated.
    tau : int
        Relaxation time.
    t : int
        Oscillation period.

    Returns
    -------
    x : array
        Array containing the time series.
    """
    a_1 = 2 * math.cos(2 * math.pi / t) * math.exp(-1 / tau)
    a_2 = -math.exp(-2 / tau)

    x = np.zeros(length)

    x[0], x[1] = abs(np.random.normal(0, 1, 1)), abs(np.random.normal(0, 1, 1))

    for i in range(2, length):
        x[i] = a_1 * x[i-1] + a_2 * x[i-2] + abs(np.random.normal(0, 1, 1))

    return x


if __name__ == '__main__':
    time_series = cyclostationary(length=10000, tau=500, t=100)
    plt.plot(time_series)
    plt.show()
