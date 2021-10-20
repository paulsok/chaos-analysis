from os import listdir
from os.path import isfile
from posixpath import join
import numpy as np
from scipy.io.wavfile import write

from systems.cubic_map import cubic_map
from systems.cyclostationary import cyclostationary
from systems.freitas import freitas
from systems.gen_henon import gen_henon
from systems.gopy import gopy
from systems.granulocyte import granulocyte
from systems.henon import henon
from systems.ikeda import ikeda
from systems.izhikevich import izhikevich
from systems.logistic import logistic
from systems.lorenz import lorenz
from systems.mackey_glass import mackey_glass
from systems.noise_sine import noise_sine
from systems.random_arma import random_arma
from systems.randomwalk import randomwalk
from systems.rossler import rossler


# all scripts for dataset creation
script_path = 'systems'
files = [f for f in listdir(script_path) if isfile(join(script_path, f))]
print(files)

n = 1024  # number of points to simulate
data = np.zeros([len(files)*10-24, n+1])

# (1) generate cubic map
regimes = ['periodic', 'chaotic', 'Heagy-Hammel', 'S3', '2T', '1T']

for i, regime in enumerate(regimes):
    time_series = cubic_map(length=n, regime=regime, level=0, discard=int(n/10))
    data[i, :] = np.concatenate(([1], time_series), axis=0)

# (2) generate cyclostationary
m1 = 10  # number of different time series of the same system
t = np.linspace(10, 500, num=m1)

for i, t in zip(range(m1), t):
    time_series = cyclostationary(length=n, t=t)
    data[i+6, :] = np.concatenate(([2], time_series), axis=0)

# (3) generate freitas
level = np.linspace(0, 0.75, num=m1)

for i, level, in zip(range(m1), level):
    time_series = freitas(length=n, level=level, discard=int(n/10))
    data[i+16, :] = np.concatenate(([3], time_series), axis=0)

# (4) generate gen. henon
a = np.linspace(1.51, 0.89, num=m1)
b = np.linspace(0.05, 0.15, num=m1)

for i, a, b, in zip(range(m1), a, b):
    time_series = gen_henon(length=n, a=a, b=b, level=0, discard=int(n/10))
    data[i+26, :] = np.concatenate(([4], time_series), axis=0)

# (5) generate gopy
sigma = np.linspace(0.5, 2, num=m1)

for i, sigma, in zip(range(m1), sigma):
    time_series = gopy(length=n, sigma=1.5, level=0, discard=int(n/10))
    data[i+36, :] = np.concatenate(([5], time_series), axis=0)

# (6) generate granulocyte
a = np.linspace(0.1, 0.4, num=m1)
b = np.linspace(0.05, 0.2, num=m1)
c = np.linspace(5, 20, num=m1)
s = np.linspace(1, 20, num=m1)

for i, a, b, c, s in zip(range(m1), a, b, c, s):
    time_series = granulocyte(length=n, a=0.2, b=0.1,
                              c=10, s=10, level=0, discard=int(n/10))
    data[i+46, :] = np.concatenate(([6], time_series), axis=0)

# (7) generate henon
a = np.linspace(1, 1.5, num=m1)
b = np.linspace(0.1, 0.5, num=m1)

for i, a, b in zip(range(5), a, b):
    x,  y = henon(length=n, a=a, b=b, level=0, discard=int(n/10))
    data[i+56, :] = np.concatenate(([7], x), axis=0)
    data[i+61, :] = np.concatenate(([7], y), axis=0)

# (8) generate ikeda
mu = np.linspace(0.51, 0.89, num=5)

for i, mu in zip(range(5), mu):
    x, y = ikeda(length=n, level=0, mu=mu, discard=int(n/10))
    data[i+66, :] = np.concatenate(([8], x), axis=0)
    data[i+71, :] = np.concatenate(([8], y), axis=0)

# (9) generate izhikevich
a = np.linspace(0.21, 0.24, num=m1)
b = np.linspace(2, 2.5, num=m1)
c = np.linspace(-55, -57, num=m1)
d = np.linspace(-15, -17, num=m1)

for i, a, b, c, d in zip(range(m1), a, b, c, d):
    time_series = izhikevich(length=n, a=a, b=b, c=c,
                             d=d, level=0, discard=int(n/10))
    data[i+76, :] = np.concatenate(([9], time_series), axis=0)

# (10) generate logistic
r = np.linspace(3.5, 3.9, num=m1)

for i, r in zip(range(m1), r):
    time_series = logistic(length=n, r=r, level=0, discard=int(n/10))
    data[i+86, :] = np.concatenate(([10], time_series), axis=0)

# (11) generate lorenz
sigma = np.linspace(12, 17, num=5)
beta = np.linspace(1.5, 4.5, num=5)
rho = np.linspace(41, 49, num=5)

for i, sigma, beta, rho in zip(range(5), sigma, beta, rho):
    time_series = lorenz(length=n, sigma=sigma, beta=beta,
                         rho=rho, discard=int(n/10))
    x, y, z = time_series[:, 0], time_series[:, 1], time_series[:, 2]
    data[i+96, :] = np.concatenate(([11], x), axis=0)
    data[i+101, :] = np.concatenate(([11], y), axis=0)

# (12) generate mackey glass
a = np.linspace(0.3, 0.5, num=m1)
b = np.linspace(0.1, 0.4, num=m1)
c = np.linspace(10, 15, num=m1)

for i, a, b, c in zip(range(m1), a, b, c):
    time_series = mackey_glass(length=n, a=a, b=b, c=c, discard=int(n/10))
    data[i+106, :] = np.concatenate(([12], time_series), axis=0)

# (13) generate noise sine
mu = np.linspace(1, 2.5, num=m1)

for i, mu in zip(range(m1), mu):
    time_series = noise_sine(length=n, mu=mu, level=0, discard=int(n/10))
    data[i+116, :] = np.concatenate(([13], time_series), axis=0)

# (14) generate randomwalk arma
p = np.linspace(0.1, 0.9, num=m1)
theta = np.linspace(0.1, 0.9, num=m1)

for i, p, theta in zip(range(m1), p, theta):
    time_series = random_arma(length=n, level=0, p=p, theta=theta, discard=int(n/10))
    data[i+126, :] = np.concatenate(([14], time_series), axis=0)

# (15) generate randomwalk
for i in range(m1):
    time_series = randomwalk(length=n, level=0, discard=int(n/10))
    data[i+136, :] = np.concatenate(([15], time_series), axis=0)

# (16) generate rossler
a = np.linspace(0.1, 0.3, num=5)
b = np.linspace(0.1, 0.3, num=5)
c = np.linspace(4, 7, num=5)

for i, a, b, c in zip(range(5), a, b, c):
    time_series = rossler(length=n, a=a, b=b, c=c, discard=int(n/10))
    x, y, z = time_series[:, 0], time_series[:, 1], time_series[:, 2]
    data[i+146, :] = np.concatenate(([16], x), axis=0)
    data[i+151, :] = np.concatenate(([16], y), axis=0)

# save dataset to txt file
np.savetxt(f'data/data.txt', data, delimiter=',')
