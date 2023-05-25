import numpy as np
import matplotlib.pyplot as plt
import pathlib, sys
from src.lorenz import LorenzTransformed

app = LorenzTransformed()
initialdir = pathlib.Path.home().joinpath('data_dir')

u0 = np.array([1, 0, 0])
info = {'T': 30, 'q': 0.1, 'u0': u0}
info['scheme'] = "IE"
info['nsamples'] = 400
info['dtit'] = 0
dirname = app.getDir(info)
datadir = pathlib.Path.home().joinpath('data_dir', "changed", dirname)

# fig = plt.figure(1)
# app.plot3d(fig, {'um': datadir})

fig = plt.figure(1)
app.histogram1d(fig, {'um': datadir})

# fig = plt.figure(3)
# app.histogram2d(fig, {'um': datadir})

fig = plt.figure(2)
app.plot_energy(fig, {'um': datadir})

plt.show()


