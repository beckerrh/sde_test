from lorenz import LorenzTransformed
import numpy as np

#------------------------------------------------------------------
if __name__ == "__main__":
    L = LorenzTransformed()
    u0 = np.array(L.FP2)+np.array([0.1,-0.1,0.1])
    # u0 = np.array([-1,0,0])
    u0 = np.array([1,0,0])
    info = {'T': 100, 'q':1, 'u0':u0}
    nsampless = 100*np.arange(1,15,4)
    dtits = np.arange(1,4)
    schemes = ["CNSE", "IE"]
    for dtit in dtits:
        info['dtit'] = dtit
        for nsamples in nsampless:
            info['nsamples'] = nsamples
            for scheme in schemes:
                print(f"{dtit=} {nsamples=} {scheme=}")
                info['scheme'] = scheme
                res = L.run(info)
