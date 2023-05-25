import matplotlib.pyplot as plt
import numpy as np
import pathlib, sys

SCRIPT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0,SCRIPT_DIR)
from src.lorenz import LorenzTransformed

#------------------------------------------------------------------
def plot_errors(res):
    from scipy.stats import linregress
    n = list(res.values())[0].shape[0]
    x = 0.5**np.arange(n-1)
    for k,v in res.items():
        w = v[1:]-v[-1]
        print(f"{w=}")
        res = linregress(x, w)
        plt.plot(x,w, "-X", label=k)
        plt.plot(x, res.intercept + res.slope * x, 'r', label=f"r={res.slope}")
        plt.grid()
        plt.legend()
        plt.show()


#------------------------------------------------------------------
def run(infos, rho=28):
    L = LorenzTransformed(rho=rho)
    u0 = np.array([1,0,0])
    info = {'u0':u0, 'scheme':'IE'}
    schemes = infos.pop('scheme', ['IE'])
    for T in infos['T']:
        info['T'] = T
        for q in infos['q']:
            info['q'] = q
            for dtit in infos['dtit']:
                info['dtit'] = dtit
                info['istep'] = 2**dtit
                for nsamples in infos['nsamples']:
                    info['nsamples'] = nsamples
                    for scheme in schemes:
                        dt = 1e-3 / 2 ** dtit
                        nt = int(T / dt)
                        dt = T / nt
                        res = L._run_convergence(scheme, dt, nt, u0, q=q, nsamples=nsamples, bins=50, ntest=6)
                        print(f"{T=} {rho=} {q=} {dtit=} {nsamples=} {scheme=}\n\t{res=}")


#------------------------------------------------------------------
if __name__ == "__main__":
    #deterministic
    # infos = {'nsamples':[1], 'q':[0], 'dtit':np.arange(0,6), 'T':[30,60,90,120]}
    # run(infos=infos, dirname="deterministic")

    # infos = {'nsamples':[1000], 'q':[0.3], 'dtit':[5], 'T':[15,20,25,30,60,90,120,150,200,300,500], 'scheme':["CNSE", "IE"]}
    # infos = {'nsamples':[1000], 'q':[0.1,0.3], 'dtit':[0], 'T':[30], 'scheme':["IE"]}
    # run(infos=infos, rho=13)
    # run(infos=infos, dirname="q_r_test", rho=13)
    # run(infos=infos, dirname="q_r_test", rho=28)

    res={'u': np.array([70.82474768, 73.37445657, 65.84990795, 60.34809857, 50.25045615]), 'H': np.array([0.01863892, 0.01839657, 0.01666778, 0.01554702, 0.01291664]), 'E': np.array([0.33365916, 0.33457892, 0.31690979, 0.29943103, 0.26239553])}
    plot_errors(res)