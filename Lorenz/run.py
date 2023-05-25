import numpy as np
import pathlib, sys

SCRIPT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0,SCRIPT_DIR)
from src.lorenz import LorenzTransformed

#------------------------------------------------------------------
def run(infos, dirname, rho=28):
    L = LorenzTransformed(rho=rho)
    u0 = np.array(L.FP2)+np.array([0.1,-0.1,0.1])
    # u0 = np.array([-1,0,0])
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
                        info['scheme'] = scheme
                        res = L.run(info, dirname=dirname)
                        print(f"{T=} {q=} {dtit=} {nsamples=} {scheme=}")


#------------------------------------------------------------------
if __name__ == "__main__":
    #deterministic
    # infos = {'nsamples':[1], 'q':[0], 'dtit':np.arange(0,6), 'T':[30,60,90,120]}
    # run(infos=infos, dirname="deterministic")

    # infos = {'nsamples':[1000], 'q':[0.3], 'dtit':[5], 'T':[15,20,25,30,60,90,120,150,200,300,500], 'scheme':["CNSE", "IE"]}
    infos = {'nsamples':[1], 'q':[0], 'dtit':[5], 'T':[15,20,25,30,60,90,120,150,200,300,500], 'scheme':["CNSE", "IE"]}
    infos = {'nsamples':[500], 'q':[0,0.1,0.5], 'dtit':[3], 'T':[30], 'scheme':["IE"]}
    run(infos=infos, dirname="q_r_test", rho=25)
    # run(infos=infos, dirname="q_r_test", rho=13)
    # run(infos=infos, dirname="q_r_test", rho=28)
