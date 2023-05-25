import numpy as np
import pathlib, sys
import matplotlib.pyplot as plt

SCRIPT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0,SCRIPT_DIR)
from src.lorenz import LorenzTransformed


infos = {'nsamples':[1000], 'q':[0.3], 'dtit':np.arange(0,6), 'T':[15], 'scheme':["CNSE", "IE"]}

u0 = np.array([1, 0, 0])
info = {'u0': u0, 'nsamples':1000, 'q':0.3, 'T':15}

app = LorenzTransformed(rho=15)

T = float(info['T'])
for scheme in ["CNSE", "IE"]:
    info['scheme'] = scheme
    for dtit in np.arange(0,6):
        info['dtit'] = dtit
        dirname = app.getDir(info)
        datadir = pathlib.Path.home().joinpath('data_dir', 'convtest', dirname)
        if dtit:
            uold = u
            Hold = H
            h1old, h2old, h3old = h1, h2, h3
        u = np.load(datadir / "en.npy")
        H = np.load(datadir / "H.npy")
        h = np.sum(H.T, axis=(1, 2))
        h1 = h/np.sum(h)
        h = np.sum(H.T, axis=(0, 2))
        h2 = h/np.sum(h)
        h = np.sum(H.T, axis=(0, 1))
        h3 = h/np.sum(h)

        edges = [np.load(datadir / "E0.npy"), np.load(datadir / "E1.npy"), np.load(datadir / "E2.npy")]
        if dtit:
            # print(f"{np.linalg.norm(u[::2]-uold)/np.sqrt(u.shape[0])} {np.linalg.norm(Hold-H)}")
            print(f" {np.linalg.norm(h1-h1old)} {np.linalg.norm(h2-h2old)} {np.linalg.norm(h3-h3old)}")
            t = np.linspace(0, T, uold.shape[0])
            n = np.argmax(t > 5)
            # plt.plot(t[n:], (u[::2]-uold)[n:], label=f"{scheme=} {dtit=}")
    # plt.legend()
    # plt.show()
