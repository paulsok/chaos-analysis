import numpy as np
import matplotlib.pyplot as plt


def granulocyte(length=10000, level=0.2, a=0.2, b=0.1, c=10, s=10, discard=1000):
    """ Simulate the circulating granulocyte levels model described in
    Mackey and Glass (1977), "Oscillation and chaos in physiological control systems".

    Parameters
    ----------
    length : int
        Length of the time series.
    level : float
        The amplitude of white noise to add to the final signal.
    a : float
        Constant.
    b : float
        Constant.
    c : float
        Constant.
    s : float
        Constant
    discard : int
        Number of steps to discard in order to eliminate transients.

    Returns
    -------
    x : array
        Array containing the time series.
    """
    x = np.zeros(length + discard)

    x_0 = 0.1 * np.random.rand(s)
    x[0] = x_0[s-1] + a * x_0[0] / (1 + x_0[0] ** c) - b * x_0[s-1]

    for i in range(1, s):
        x[i] = x[i-1] + a * x_0[i] / (1 + x_0[i] ** c) - b * x[i-1]

    for i in range(s, length+discard):
        x[i] = x[i-1] + a * x[i-s] / (1 + x[i-s] ** c) - b * x[i-1]

    # add white noise
    _x = x + level * np.std(x) * np.random.rand(length + discard)

    return _x[discard:]


if __name__ == '__main__':
    time_series = granulocyte(length=10000, level=0, a=0.2, b=0.1, c=10, s=10, discard=1000)
    plt.plot(time_series)
    plt.show()
