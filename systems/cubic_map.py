import math
import random
import numpy as np
import matplotlib.pyplot as plt


def cubic_map(length=10000, f=0, q=0, a=1, level=0.2, regime='periodic', discard=1000):
    """Generate time series using the periodically forced cubic map described in Venkatesan
        and  Lakshmanan (2001), "Interruption of torus doubling bifurcation and genesis
        of strange nonchaotic attractors in a quasiperiodically forced map:
        Mechanisms and their characterizations".

    Parameters
    ----------
    length : int
        Length of the time series to be generated.
    f : float
        Constant.
    q : float
        Constant.
    a : float
        Constant.
    level : float
        Amplitude of white noise to add to the final signals.
    regime : str ['periodic', 'chaotic', 'Heagy-Hammel', 'S3', '2T']
        Type of the dynamics.
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

    x[0] = random.random()
    theta[0] = random.random()

    if regime == 'periodic':
        f, q, a = 0, 0, 1
    elif regime == 'chaotic':
        f, q, a = -.8, 0, 1.5
    elif regime == 'Heagy-Hammel':
        a, q, f = 1.88697, 0, 0.7
    elif regime == 'S3':
        a, q, f = 2.14, 0, .35
    elif regime == '2T':
        a, q, f = 1.1, 0, -.18
    elif regime == '1T':
        a, q, f = 0.5, 0, 0.4

    for i in range(1, length + discard):
        x[i] = q + f * math.cos(2 * math.pi * theta[i-1]) - a * x[i-1] + x[i-1] ** 3
        theta[i] = (theta[i-1] + w) % 1

    # adding a noise
    _x = x + level * np.std(x) * np.random.rand(length + discard)

    return _x[discard:]


if __name__ == '__main__':
    time_series = cubic_map(length=10000, f=0, q=0, a=1, level=0.2, regime='1T', discard=1000)
    plt.plot(time_series)
    plt.show()
