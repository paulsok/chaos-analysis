import math
import random
import numpy as np
import matplotlib.pyplot as plt


def poincare(length=10000, level=0.2, b=1.13, tau=0.75, discard=1000):
    """The following is slightly modified from the script originally
       written by Leon Glass for Nonlinear Dynamics in Physiology
       and Medicine (2003).

    Parameters
    ----------
    length : int
        Length of the time series.
    level : float
        The amplitude of white noise to add to the final signal.
    b : float
        Constant.
    tau : float
        Constant.
    discard : int
        Number of steps to discard in order to eliminate transients.

    Returns
    -------
    beats : array
        Beats lists.
    phi : array
        Time-series of successive phases of the oscillator.
    """
    phi = np.zeros(length + discard)
    phi[0] = random.random()
    beats = np.zeros(length + discard)

    for i in range(1, length + discard):
        angle = 2 * math.pi * phi[i-1]
        r_prime = math.sqrt(1 + b ** 2 + 2 * b * math.cos(angle))
        argument = (math.cos(angle) + b) / r_prime
        phi[i] = math.acos(argument) / (2 * math.pi)

        if phi[i-1] > 0.5:
            phi[i] = 1 - phi[i]

        phi[i] = phi[i] + tau
        beats[i] = phi[i] - phi[i] % 1
        phi[i] = phi[i] % 1

    # add white noise
    _beats = beats + level * np.std(beats) * np.random.rand(length + discard)
    _phi = phi + level * np.std(phi) * np.random.rand(length + discard)

    return _beats[discard:], _phi[discard:]


if __name__ == '__main__':
    _beats, _phi = poincare(length=10000, level=0.2, b=1.13, tau=0.75, discard=1000)
    plt.plot(_beats)
    plt.plot(_phi)
    plt.show()
