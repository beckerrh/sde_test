import numpy as np
import pathlib, sys
import matplotlib.pyplot as plt

SCRIPT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0,SCRIPT_DIR)
from src.lorenz import LorenzTransformed


infos = {'nsamples':[1000], 'q':[0.3], 'dtit':np.arange(0,6), 'T':[15], 'scheme':["CNSE", "IE"]}

u0 = np.array([1, 0, 0])
info = {'u0': u0, 'nsamples':1000, 'q':0.3, 'T':15}

app = LorenzTransformed()
info1 = {'u0': u0, 'nsamples':1,   'dtit':2, 'q':0,   'T':60, 'scheme':"IE"}
info2 = {'u0': u0, 'nsamples':600, 'dtit':2, 'q':0.1, 'T':60, 'scheme':"IE"}
info3 = {'u0': u0, 'nsamples':600, 'dtit':2, 'q':0.5, 'T':60, 'scheme':"IE"}

datadirs = {}
datadirs["q=0"] = pathlib.Path.home().joinpath('data_dir', "deterministic", app.getDir(info1))
datadirs["q=0.1"] = pathlib.Path.home().joinpath('data_dir', "changed", app.getDir(info2))
datadirs["q=0.5"] = pathlib.Path.home().joinpath('data_dir', "changed", app.getDir(info3))

fig = plt.figure(1)
app.plot2d(fig, datadirs, name="ul")
plt.show()
fig = plt.figure(2)
app.plot2d(fig, {k:v for k,v in datadirs.items() if k in ["q=0"]}, name="um")
plt.show()
fig = plt.figure(3)
app.plot2d(fig, {k:v for k,v in datadirs.items() if k in ["q=0.1","q=0.5"]}, name="um")
plt.show()