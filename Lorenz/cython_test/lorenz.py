import numpy as np
import matplotlib.pyplot as plt
import pathlib
import lorenz_run

#------------------------------------------------------------------
class LorenzTransformed(lorenz_run.LorenzRun):
    def __init__(self, sigma=10, rho=28, beta=8/3):
        super().__init__(sigma, rho, beta)
        self.FP1 =  [np.sqrt(beta*(rho-1)), np.sqrt(beta*(rho-1)),rho-1]
        self.FP2 =  [-np.sqrt(beta*(rho-1)), -np.sqrt(beta*(rho-1)),rho-1]
        # self.dnl = lambda u: np.array([[0,0,0], [-u[2],0,-u[0]], [u[1],u[0],0]])
        # self.F = lambda u: np.array([[0,0,0], [0,0,-u[0]], [0,u[0],0]])
    def change_solution(self, u): u[:,2] += self.rho+self.sigma
    def plot_histogram(self, fig, files, compute=False, dim=2, bins=20):
        # print(f"{files=} {compute=} {dim=}")
        import matplotlib.gridspec as gridspec
        from matplotlib import cm
        cmap = cm.gist_gray_r
        nplots = len(files)
        gs = gridspec.GridSpec(nplots, 3, wspace=0.2)
        i = 0
        for key, file in files.items():
            if compute:
                u = np.load(file)
                H, edges = np.histogramdd(u, bins=bins, range=[(-20, 20), (-25, 25), (0, 50)], density=True)
            else:
                H = np.load(file)
                fileE0 = pathlib.Path(str(file).replace("H", "E0"))
                fileE1 = pathlib.Path(str(file).replace("H", "E1"))
                fileE2 = pathlib.Path(str(file).replace("H", "E2"))
                edges = [np.load(fileE0),np.load(fileE1),np.load(fileE2)]
            # print(f"{H.shape=}")
            axs = [fig.add_subplot(gs[i,j]) for j in range(3)]
            if i==0:
                if dim==2:
                    axs[0].set_title(f"X-Y")
                    axs[1].set_title(f"X-Z")
                    axs[2].set_title(f"Y-Z")
                else:
                    axs[0].set_title(f"X")
                    axs[1].set_title(f"Y")
                    axs[2].set_title(f"Z")
            if dim == 2:
                axs[0].pcolormesh(*np.meshgrid(edges[0], edges[1]), np.mean(H, axis=2).T, cmap=cmap)
                axs[1].pcolormesh(*np.meshgrid(edges[0], edges[2]), np.mean(H, axis=1).T, cmap=cmap)
                axs[2].pcolormesh(*np.meshgrid(edges[1], edges[2]), np.mean(H, axis=0).T, cmap=cmap)
            else:
                axs[0].plot(np.mean(H, axis=(1,2)))
                axs[1].plot(np.mean(H, axis=(0,2)))
                axs[2].plot(np.mean(H, axis=(0,1)))
            i += 1
    def plot3d(self, fig, files):
        ax = fig.add_subplot(projection='3d')
        plotfixed=True
        for key, file in files.items():
            u = np.load(file)
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
    def plot2d(self, fig, files):
        ax = fig.add_subplot()
        plotfixed=True
        for key, file in files.items():
            u = np.load(file)
            xy, z = 0.5 * (u[:, 0] + u[:, 1]), u[:, 2]
            ax.plot(xy, z, label=key, lw=0.5)
            ax.plot(xy[-1], z[-1], 'X', label=key+"(T)")
            if plotfixed:
                ax.plot(xy[0], z[0], 'X', label="X_0")
                x,y,z = self.FP1
                ax.plot((x+y)/2, z, color='k', marker="8", ls='')
                x,y,z = self.FP2
                ax.plot((x+y)/2, z, color='k', marker="8", ls='')
                plotfixed=False
        ax.legend()

# ----------------------------------------------------------
    def _run(self, scheme, dt, nt, u0, istep=1, q=0, nsamples=1, bins=20):
        u = np.empty(shape=(nt, 3))
        sigma, rho, beta = self.sigma, self.rho, self.beta
        u[0] = u0
        u[0,2] -= rho+sigma
        u_mean = np.zeros_like(u)
        for isample in range(nsamples):
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
            H, edges = np.histogramdd(u, bins=bins, range=[(-20, 20), (-25, 25), (0, 50)], density=True)
            if not isample:
                H_mean = np.zeros_like(H)
            H_mean += H / nsamples
            u_mean += u/nsamples
            print(f"{100 * (isample / nsamples):4.1f}%")
        self.change_solution(u)
        self.change_solution(u_mean)
        return u[::istep], u_mean[::istep], H_mean, edges
    def getDir(self, info):
        u0s = '_'.join([str(i) for i in info['u0']])
        return f"T={info['T']}@r={self.rho}@q={info['q']}@dtit={info['dtit']}@M={info['nsamples']}@u0={u0s}@met={info['scheme']}"
    def run(self, info):
        import shutil
        dirname = self.getDir(info)
        datadir = pathlib.Path.home().joinpath( 'data_dir', "schemes", dirname)
        try:
            shutil.rmtree(datadir)
        except:
            pass
        pathlib.Path(datadir).mkdir(parents=True, exist_ok=True)
        scheme, u0, q, T, dtit, nsamples = info['scheme'], info['u0'], info['q'], info['T'], info['dtit'], info['nsamples']
        dt = 1e-3/4**dtit
        nt = int(T/dt)
        dt = T / nt
        ul, um, H, E = self._run(scheme, dt, nt, u0, istep=dtit, q=q, nsamples=nsamples, bins=20)
        np.save(datadir / "ul", ul)
        np.save(datadir / "um", um)
        np.save(datadir / "H", H)
        np.save(datadir / "E0", E[0])
        np.save(datadir / "E1", E[1])
        np.save(datadir / "E2", E[2])

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
