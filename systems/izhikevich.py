import random
import numpy as np
import matplotlib.pyplot as plt


def izhikevich(length=10000, level=0, tau=0.25, a=0.2, b=2, c=-56, d=-16, mu=-99, discard=1000):
    """Simulate the Izhikevich spiking neuron model described in Izhikevich (2003),
       "Simple model of spiking neurons" and Izhikevich (2004), "Which model to use
       for cortical spiking neurons?".

    Parameters
    ----------
    length : int
        Length of the time series.
    level : float
        The amplitude of white noise to add to the final signal.
    tau : float
        Integration step size.
    a : float
        Constant.
    b : float
        Constant.
    c : float
        Constant.
    d : float
        Constant
    mu : float
        Constant
    discard : int
        Number of steps to discard in order to eliminate transients.

    Returns
    -------
    x : array
        Array containing the time series.
    """
    v = random.random() * 80
    u = b * v
    x, uu = [], []

    for _ in range(length + discard):
        v = v + tau * (0.04 * v ** 2 + 5 * v + 140 - u + mu)
        u = u + tau * a * (b * v - u)
        if v > 30:
            x.append(30)
            v = c
            u = u + d
        else:
            x.append(v)
        uu.append(u)

    # add white noise
    _x = x + level * np.std(x) * np.random.rand(length + discard)

    return _x[discard:]


if __name__ == '__main__':
    time_series = izhikevich(length=10000, level=0, tau=0.25, a=0.2, b=2, c=-56, d=-16, mu=-99, discard=1000)
    plt.plot(time_series)
    plt.show()
