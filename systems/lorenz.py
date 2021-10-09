import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def lorenz(length=10000, x0=None, sigma=16, beta=4, rho=45.92,
           step=0.001, sample=0.03, discard=1000):
    """Simulate the Lorenz system described in Lorenz (1963),
       "Deterministic nonperiodic flow" using a fourth-order
       Runge Kutta method.

    Parameters
    ----------
    length : int
        Length of the time series.
    x0 : array
        Initial condition.
    sigma : float
        Constant.
    beta : float
        Constant.
    rho : float
        Constan.
    step : float
        Step size of integration.
    sample : int
        Sampling step.
    discard : int
        Number of samples to discard in order to eliminate transients.

    Returns
    -------
    x : ndarray, shape (length, 3)
        Array containing points in phase space.
    """
    _lorenz = lambda x, t: [sigma * (x[1] - x[0]), x[0] *
                            (rho - x[2]) - x[1],
                            x[0] * x[1] - beta * x[2]]

    if not x0:
        x0 = (0.0, -0.01, 9.0) + 0.25 * (-1 + 2 * np.random.random(3))

    sample = int(sample / step)

    t = np.linspace(0, (sample * (length + discard)) * step,
                    sample * (length + discard))

    return odeint(_lorenz, x0, t)[discard * sample::sample]


if __name__ == '__main__':
    time_series = lorenz(length=10000, x0=None, sigma=16, beta=4, rho=45.92,
                         step=0.001, sample=0.03, discard=1000)
    plt.plot(time_series)
    plt.show()
