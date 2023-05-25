import numpy as np
import matplotlib.pyplot as plt
import pathlib
if __name__ == "__main__":
    import lorenz_run
else:
    from . import lorenz_run

#------------------------------------------------------------------
class LorenzTransformed(lorenz_run.LorenzRun):
    def __init__(self, sigma=10, rho=28, beta=8/3):
        super().__init__(sigma, rho, beta)
        self.FP1 =  [np.sqrt(beta*(rho-1)), np.sqrt(beta*(rho-1)),rho-1]
        self.FP2 =  [-np.sqrt(beta*(rho-1)), -np.sqrt(beta*(rho-1)),rho-1]
        self.linestyles = ['solid', 'dotted', 'dashed', 'dashdot', (5, (10, 3)), (0, (3, 1, 1, 1))]
        self.BaW=False
    def change_solution(self, u): u[:,2] += self.rho+self.sigma
    def compute_histogram(self, files, bins=50, range=[(-20, 20), (-25, 25), (0, 50)]):
        H = {}
        for key, file in files.items():
            u = np.load(file)
            H[key], edges = np.histogramdd(u, bins=bins, range=range, density=True)
        return H, edges
    def get_sorted_dict(self, dirs):
        try:
            float(list(dirs.keys())[0].split('=')[1])
            dirsiter = dict(sorted(dirs.items(), key=lambda x: float(x[0].split('=')[1])))
        except:
            dirsiter = dict(sorted(dirs.items()))
        return dirsiter
    def plt_style(self, i):
        if not self.BaW: return {}
        return {'c':'black', 'linestyle':self.linestyles[i%6]}
    def histogram2d(self, fig, dirs):
        # print(f"{files=} {compute=} {dim=}")
        import matplotlib.gridspec as gridspec
        from matplotlib import cm
        cmap = cm.gist_gray_r
        nplots = len(dirs)
        gs = gridspec.GridSpec(nplots, 3, wspace=0.4, hspace=0.3)
        dirsiter = self.get_sorted_dict(dirs)
        for i,(key, datadir) in enumerate(dirsiter.items()):
            H = np.load(datadir/"H.npy")
            edges = [np.load(datadir/"E0.npy"),np.load(datadir/"E1.npy"),np.load(datadir/"E2.npy")]
            axs = [fig.add_subplot(gs[i,j]) for j in range(3)]
            if i==0:
                axs[0].set_title(f"X-Y")
                axs[1].set_title(f"X-Z")
                axs[2].set_title(f"Y-Z")
            axs[0].set_ylabel(f"{key}", fontsize = 9)
            axs[0].pcolormesh(*np.meshgrid(edges[0], edges[1]), np.sum(H.T, axis=2).T, cmap=cmap)
            axs[1].pcolormesh(*np.meshgrid(edges[0], edges[2]), np.sum(H.T, axis=1).T, cmap=cmap)
            axs[2].pcolormesh(*np.meshgrid(edges[1], edges[2]), np.sum(H.T, axis=0).T, cmap=cmap)
    def histogram1d(self, fig, dirs):
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(1, 3, wspace=0.4, hspace=0.3)
        axs = [fig.add_subplot(gs[0, j]) for j in range(3)]
        axs[0].set_title(f"X")
        axs[1].set_title(f"Y")
        axs[2].set_title(f"Z")
        dirsiter = self.get_sorted_dict(dirs)
        for i,(key, datadir) in enumerate(dirsiter.items()):
            H = np.load(datadir/"H.npy")
            h = np.sum(H.T, axis=(1, 2))
            axs[0].plot(h/np.sum(h), label=key, **(self.plt_style(i)))
            h = np.sum(H.T, axis=(0, 2))
            axs[1].plot(h/np.sum(h), label=key, **(self.plt_style(i)))
            h = np.sum(H.T, axis=(0, 1))
            axs[2].plot(h/np.sum(h), label=key, **(self.plt_style(i)))
        for ax in axs: ax.legend()
    def plot3d(self, fig, dirs, name="um"):
        ax = fig.add_subplot(projection='3d')
        plotfixed=True
        dirsiter = self.get_sorted_dict(dirs)
        for i,(key, datadir) in enumerate(dirsiter.items()):
            u = np.load(datadir/(name+".npy"))
            x,y,z = u[:,0], u[:,1], u[:,2]
            ax.plot(x, y, z, label=key, lw=0.5)
            ax.plot(x[-1], y[-1], z[-1], 'X', label=key+"(T)")
            if plotfixed:
                ax.plot(x[0], y[0], z[0], 'X', label="X_0")
                ax.plot(*self.FP1, color='k', marker="8", ls='')
                ax.plot(*self.FP2, color='k', marker="8", ls='')
                plotfixed = False
        ax.view_init(26, 130)
        ax.legend()
        # ax.set_title(key)
        plt.draw()
    def plot2d(self, fig, dirs, name="um", endpoints=True):
        ax = fig.add_subplot()
        plotfixed=True
        dirsiter = self.get_sorted_dict(dirs)
        for i,(key, datadir) in enumerate(dirsiter.items()):
            u = np.load(datadir/(name+".npy"))
            xy, z = 0.5 * (u[:, 0] + u[:, 1]), u[:, 2]
            ax.plot(xy, z, label=key, lw=0.5, **self.plt_style(i))
            ax.plot(xy[-1], z[-1], 'X', label=key+"(T)")
            if plotfixed:
                ax.plot(xy[0], z[0], 'X', label="X_0")
                x,y,z = self.FP1
                ax.plot((x+y)/2, z, color='k', marker="8", ls='')
                x,y,z = self.FP2
                ax.plot((x+y)/2, z, color='k', marker="8", ls='')
                plotfixed=False
        ax.legend()
    def plot_energy(self, fig, dirs):
        ax = fig.add_subplot()
        dirsiter = self.get_sorted_dict(dirs)
        for i,(key, datadir) in enumerate(dirsiter.items()):
            info = self.getDict(datadir)
            T = float(info['T'])
            u = np.load(datadir/"en.npy")
            t = np.linspace(0,T,u.shape[0])
            n = np.argmax(t > 5)
            ax.plot(t[n:], u[n:], label=key, **self.plt_style(i))
        ax.legend()

    def getDir(self, info):
        u0s = '_'.join([str(i) for i in info['u0']])
        return f"T={info['T']}@r={self.rho}@q={info['q']}@dtit={info['dtit']}@M={info['nsamples']}@u0={u0s}@met={info['scheme']}"
    def getDict(self, dir):
        dir = str(dir.name)
        info = {}
        for ds in dir.split('@'):
            dss = ds.split('=')
            info[dss[0]] = dss[1]
        return info

# ----------------------------------------------------------
    def _run(self, scheme, dt, nt, u0, istep=1, q=0, nsamples=1, bins=20):
        u = np.empty(shape=(nt, 3))
        sigma, rho, beta = self.sigma, self.rho, self.beta
        u_mean = np.zeros_like(u)
        E = np.zeros(nt)
        for isample in range(nsamples):
            u[0] = u0
            u[0, 2] -= rho + sigma
            noise = np.random.randn(nt)
            if scheme=="CNSE":
                self._run_cnse(u, dt, nt, q, noise)
            elif scheme=="CNSI":
                self._run_cnsi(u, dt, nt, q, noise)
            elif scheme == "IE":
                self._run_ie(u, dt, nt, q, noise)
            elif scheme == "EE":
                self._run_ee(u, dt, nt, q, noise)
            else:
                raise ValueError("unknown scheme {}".format(scheme))
            self.change_solution(u)
            H, edges = np.histogramdd(u, bins=bins, range=[(-20, 20), (-25, 25), (0, 50)], density=True)
            if not isample:
                H_mean = np.zeros_like(H)
            H_mean += H / nsamples
            u_mean += u/nsamples
            E += np.sum(u*u, axis=1)/nsamples
            print(f"{100 * (isample / nsamples):4.1f}%", end="\r")
        return u[::istep], u_mean[::istep], H_mean, edges, E

    # ----------------------------------------------------------
    def _run_convergence(self, scheme, dt0, nt0, u0, q=0, nsamples=1, bins=50, ntest=6):
        sigma, rho, beta = self.sigma, self.rho, self.beta
        nts = np.array([(nt0-1)*2**i for i in range(ntest)])+1
        dts = np.array([dt0/2**i for i in range(ntest)])
        for isample in range(nsamples):
            noise = np.random.randn(nts[-1])
            us, Hs, Es = [], [], []
            for i in range(ntest):
                dt, nt = dts[i], nts[i]
                u = np.empty(shape=(nt, 3))
                u[0] = u0
                u[0, 2] -= rho + sigma
                if scheme == "CNSE":
                    self._run_cnse(u, dt, nt, q, noise[::2**(ntest-i-1)])
                elif scheme == "CNSI":
                    self._run_cnsi(u, dt, nt, q, noise[::2**(ntest-i-1)])
                elif scheme == "IE":
                    self._run_ie(u, dt, nt, q, noise[::2**(ntest-i-1)])
                elif scheme == "EE":
                    self._run_ee(u, dt, nt, q, noise[::2**(ntest-i-1)])
                else:
                    raise ValueError("unknown scheme {}".format(scheme))
                self.change_solution(u)
                us.append(u[::2**i])
                H, edges = np.histogramdd(u, bins=bins, range=[(-20, 20), (-25, 25), (0, 50)], density=True)
                Hs.append(H)
                Es.append(np.sum(u[::2**i] * u[::2**i], axis=1) / nt0)
                print(f"{100 * (isample / nsamples):4.1f}%", end="\r")
            # err_u_om = np.array([np.linalg.norm((us[i+1]-us[i])**2)/np.sqrt(nt0) for i in range(ntest-1)])
            # err_H_om = np.array([np.linalg.norm((Hs[i + 1] - Hs[i]) ** 2) for i in range(ntest - 1)])
            # err_E_om = np.array([np.linalg.norm(Es[i + 1] - Es[i]) for i in range(ntest - 1)])
            err_u_om = np.array([np.linalg.norm((us[-1]-us[i])**2)/np.sqrt(nt0) for i in range(ntest-1)])
            err_H_om = np.array([np.linalg.norm((Hs[-1] - Hs[i]) ** 2) for i in range(ntest - 1)])
            err_E_om = np.array([np.linalg.norm(Es[-1] - Es[i]) for i in range(ntest - 1)])
            if isample==0:
                err_u = np.zeros_like(err_u_om)
                err_H = np.zeros_like(err_H_om)
                err_E = np.zeros_like(err_E_om)
            err_u += err_u_om/nsamples
            err_H += err_H_om / nsamples
            err_E += err_E_om / nsamples
        return {'u':err_u, 'H':err_H, 'E':err_E}

    # ----------------------------------------------------------
    def run(self, info, dirname):
        import shutil
        scheme, u0, q, T, dtit, nsamples, istep = info['scheme'], info['u0'], info['q'], info['T'], info['dtit'], info['nsamples'], info['istep']
        dt = 1e-3/2**dtit
        nt = int(T/dt)
        dt = T / nt
        ul, um, H, E, en = self._run(scheme, dt, nt, u0, istep=istep, q=q, nsamples=nsamples, bins=50)
        # print(f"{ul.shape=} {nt=}")
        dirname2 = self.getDir(info)
        datadir = pathlib.Path.home().joinpath( 'data_dir', dirname, dirname2)
        try:
            shutil.rmtree(datadir)
        except:
            pass
        pathlib.Path(datadir).mkdir(parents=True, exist_ok=True)
        np.save(datadir / "ul", ul)
        np.save(datadir / "um", um)
        np.save(datadir / "H", H)
        np.save(datadir / "E0", E[0])
        np.save(datadir / "E1", E[1])
        np.save(datadir / "E2", E[2])
        np.save(datadir / "en", en)

#---------------------------------------------------------------------
if __name__ == "__main__":
    lorenz = LorenzTransformed()
    import time
    t0 = time.time()
    lorenz._run(scheme="CNSE", dt=1e-3, nt=1000000, u0=[1,1,0], istep=1, q=0, nsamples=1)
    t1 = time.time()
    lorenz._run(scheme="IE", dt=1e-3, nt=1000000, u0=[1,1,0], istep=1, q=0, nsamples=1)
    t2 = time.time()
    print(f"{t1-t0:10.3e} {t2-t1:10.3e}")
