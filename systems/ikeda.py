import numpy as np
import matplotlib.pyplot as plt


def ikeda(length=10000, level=0.2, x0=None, alpha=6.0, beta=0.4, gamma=1.0, mu=0.9, discard=1000):
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
    x0 : array
        Initial condition.
    alpha : float
        Constant.
    beta : float
        Constant.
    gamma : float
        Constant.
    mu : float
        Constant.
    discard : int
        Number of steps to discard in order to eliminate transients.

    Returns
    -------
    x : array
        Array containing the time series.
    """
    x = np.empty((length + discard, 2))

    if not x0:
        x[0] = 0.1 * (-1 + 2 * np.random.random(2))
    else:
        x[0] = x0

    for i in range(1, length + discard):
        phi = beta - alpha / (1 + x[i - 1][0] ** 2 + x[i - 1][1] ** 2)
        x[i] = (gamma + mu * (x[i - 1][0] * np.cos(phi) - x[i - 1][1] *
                np.sin(phi)),
                mu * (x[i - 1][0] * np.sin(phi) + x[i - 1][1] * np.cos(phi)))

    return x[:, 0][discard:]


if __name__ == '__main__':
    x = ikeda(length=10000, level=0.2, x0=None, alpha=6.0, beta=0.4, gamma=1.0, mu=0.9, discard=1000)
    plt.plot(x)
    plt.show()
