import numpy as np
import pathlib, sys
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib

SCRIPT_DIR = str(pathlib.Path(__file__).parent.parent)
sys.path.insert(0,SCRIPT_DIR)
import src.models
from functools import partial

#-----------------------------------------------------------------------------#
class Myapp(src.models.application.Application):
    def __init__(self, h=0.1, mu=1):
        super().__init__(h=h, ncomp=2)
        data = self.problemdata
        data.bdrycond.set("Dirichlet", [1000, 1002])
        # data.params.fct_glob["initial_condition"] = []
        data.params.scal_glob["mu"] = mu
        data.modes = []
        m = [partial(self.sin_sin_V0, a=2, b=1), partial(self.sin_sin_V1, a=2, b=1)]
        data.modes.append(m)
        m = [partial(self.sin_sin_V0, a=1, b=2), partial(self.sin_sin_V1, a=1, b=2)]
        data.modes.append(m)
    def defineGeometry(self, geom, h):
        p = geom.add_rectangle(xmin=0, xmax=1, ymin=0, ymax=1, z=0, mesh_size=h)
        geom.add_physical(p.surface, label="100")
        geom.add_physical(p.lines[2], label="1002")
        geom.add_physical([p.lines[0], p.lines[1], p.lines[3]], label="1000")
    def sin_sin_V0(self, x, y, z, a, b):
        # a, b = 2, 1
        return 2*b*np.pi*np.cos(b*np.pi*y)*np.sin(b*np.pi*y)*np.sin(a*np.pi*x)**2
        return -2*y*(1-y)*(1-2*y)*x**2*(1-x)**2
    def sin_sin_V1(self, x, y, z, a, b):
        # a, b = 2, 1
        return -2*a*np.pi*np.cos(a*np.pi*x)*np.sin(a*np.pi*x)*np.sin(b*np.pi*y)**2
        return 2*x*(1-x)*(1-2*x)*y**2*(1-y)**2

#-----------------------------------------------------------------------------#
class StokesSDE(src.models.stokes.Stokes):
    def __init__(self,**kwargs):
        self.wnoise = kwargs.pop("wnoise", None)
        super().__init__(**kwargs)
    def setMesh(self, mesh):
        super().setMesh(mesh)
        self.modes = []
        for m in self.application.problemdata.modes:
            self.modes.append([self.femv.interpolate(m[i]) for i in range(self.ncomp)])
    def computeRhs(self, scale, rhs, it, dt):
        scale /= np.sqrt(dt)
        for i, m in enumerate(self.modes):
            if len(self.wnoise.shape) == 2:
                for icomp in range(self.ncomp):
                    self.vectorview.add(0, icomp, rhs, self.wnoise[i, it] * scale, self.Mass.M @ m[icomp])
            else:
                for icomp in range(self.ncomp):
                    self.vectorview.add(0, icomp, rhs, self.wnoise[icomp,i,it]*scale, self.Mass.M@m[icomp])
    def computeMatrixConstant(self, u=None, coeffmass=None, theta=None):
        S = super().computeMatrix(u, coeffmass)
        S.B /= theta
        S.C /= theta
        return S
    def rhs_dynamic(self, rhs, u, Aconst, time, dt, theta, it):
        self.Mass.dot(rhs, 1 / (theta * theta * dt), u)
        v, p, lam = self.vectorview.get_parts(u)
        rhs_v, rhs_p, rhs_lam = self.vectorview.get_parts(u)
        rhs_v += (theta - 1) / theta*Aconst._dot_A(v)
        rhs_p.fill(0)
        rhs_lam.fill(0)
        # rhs += (1 / theta) * self.computeRhs()
        self.computeRhs(scale=1/theta, rhs=rhs, it=it, dt=dt)
    def dynamic(self, u0, n_inner, nframes, **kwargs):
        if u0 is None:
            u0 = np.zeros(self.vectorview.n())
        if not self.vectorview.n() == u0.size:
            raise ValueError(f"needs u0 of shape {self.vectorview.ncomps=}")
        dt = kwargs.pop('dt')
        theta = kwargs.pop('theta', 1)
        verbose = kwargs.pop('verbose', True)
        rtolsemi_imp = kwargs.pop('rtolsemi_imp', 1e-8)
        save_add = kwargs.pop('save_add')
        # sdata = kwargs.pop('sdata', src.solvers.newtondata.StoppingParamaters(maxiter=maxiternewton, rtol=rtolnewton))
        if len(kwargs):
            raise ValueError(f"unused arguments: {kwargs.keys()}")
        if not dt or dt<=0: raise NotImplementedError(f"needs constant positive 'dt")
        result = src.models.problemdata.Results(nframes)
        u = u0
        self.time = 0
        self.rhs = np.empty_like(u, dtype=float)
        if not hasattr(self, 'Mass'):
            self.Mass = self.computeMassMatrix()
        self.theta, self.dt = theta, dt
        self.coeffmass = 1 / dt / theta
        if not hasattr(self, 'Aconst'):
            Aconst = self.computeMatrixConstant(coeffmass=self.coeffmass, theta=theta)
        self.LS = self.computelinearSolver(Aconst)
        self.timeiter = 0
        info_new = src.solvers.newtondata.IterationData(rtolsemi_imp)
        normsnames =  ['v', 'p', 'l', 'f']
        result.enrolPPscalar(normsnames)
        if verbose:
            print(30*"*" + f" {theta=} "+30*"*")
            print(f"*** {'t':12s} {'it':6s} {'dt':6s} {'n_lin_av':8s} {'n_nl_av':8s} {'n_bad':6s} {'nnew':4s}")
        self.timer.add('init')
        for iframe in range(nframes):
            info_new.totaliter = 0
            info_new.totalliniter = 0
            info_new.bad_convergence_count = 0
            info_new.calls = 0
            pp = self.postProcess(u)
            if hasattr(self.application, "changepostproc"):
                self.application.changepostproc(pp['scalar'])
            result.addData(iframe, pp, time=self.time, iter=info_new.totaliter, liniter=info_new.totalliniter)
            self.save(u=u, iter=iframe, add=save_add)
            for it in range(n_inner):
                self.time += dt
                self.application.time = self.time
                self.rhs.fill(0)
                self.rhs_dynamic(self.rhs, u, Aconst, self.time, dt, theta, self.timeiter)
                self.timer.add('rhs')
                u = self.LS.solve(A=Aconst, b=self.rhs, rtol=rtolsemi_imp, verbose=True)
                self.timer.add('solve')
                vn, pn, ln = self.vectorview.get_norms(u)
                fn = self.vectorview.get_norms(self.rhs)[0]
                norms={'f':fn, 'v':vn, 'p':pn, 'l':ln}
                result.addPPscalar(norms)
                info_new.liniter.append(self.LS.niter)
                if self.LS.niter == self.LS.maxiter:
                    print(f"*** solver failure {rtolsemi_imp=} {self.LS.niter=}")
                self.timer.add('postproc')
                self.timeiter += 1
            if verbose:
                print(f"*** {self.time:9.3e} {iframe:6d} {self.dt:8.2e} {info_new.niter_lin_mean():8.2f} {info_new.niter_mean():8.2f} {info_new.bad_convergence_count:8.2f} {info_new.calls:3d}")
        iframe += 1
        pp = self.postProcess(u)
        if hasattr(self.application, "changepostproc"):
            self.application.changepostproc(pp['scalar'])
        result.addData(iframe, pp, time=self.time, iter=info_new.totaliter, liniter=info_new.totalliniter)
        self.save(u=u, iter=iframe, add=save_add)
        result.save(self.datadir)
        result.norms = norms
        return result



#-----------------------------------------------------------------------------#
class RunStokes():
    def __init__(self, **kwargs):
        if 'noinit' in kwargs: return
        print(f"{kwargs=}")
        self.h = kwargs.pop('h', 0.05)
        T, nt, nframes, n_inner = kwargs.pop('T', 1), kwargs.pop('nt', 7), kwargs.pop('nframes', 50), kwargs.pop('n_inner', 10)
        # nt, nframes, n_inner = 3, 5, 1
        self.T, self.nt, self.nframes, self.n_inner = T, nt, nframes, n_inner
        self.init()
    def init(self):
        self.n_inners = self.n_inner * np.array([2 ** k for k in range(self.nt)])
        datadir_add = f"_{float(self.h)}_{float(self.T)}_{self.nframes}_{self.n_inner}_{self.nt}"
        # print(f"init: {datadir_add=}")
        app = Myapp(self.h)
        self.model = StokesSDE(application=app, clean_data=False, stack_storage=False, datadir_add=datadir_add)
        self.nsample = 1000
        self.load()
    def init_from_directory(self, dirname):
        ds = dirname.name.split('_')
        # print(f"{ds=}")
        assert len(ds)>6
        self.h, self.T, self.nframes, self.n_inner, self.nt = float(ds[-5]), float(ds[-4]), int(ds[-3]), int(ds[-2]), int(ds[-1])
        self.init()
    def load(self):
        try:
            errallv = self.model.load(name="errallv")
            errallp = self.model.load(name="errallp")
        except:
            errallv, errallp = [], []
        try:
            errormean = self.model.load(name="errormean")
        except:
            errormean = np.zeros(shape=(self.nt - 1, self.model.vectorview.n()))
        # print(f"{errallv=} {errallp=}")
        self.errormean, self.errallv, self.errallp = errormean, errallv, errallp
    def run(self):
        nsample, T, nt, nframes, n_inners = self.nsample, self.T, self.nt, self.nframes, self.n_inners
        model, errormean, errallv, errallp = self.model, self.errormean, self.errallv, self.errallp
        nmodes = len(self.model.application.problemdata.modes)
        if not isinstance(errallp, list):
            errallv = [errallv[i] for i in range(errallv.shape[0])]
            errallp = [errallp[i] for i in range(errallp.shape[0])]
        for isample in range(nsample):
            # wnoise = np.random.randn(2, len(app.problemdata.modes), nframes*n_inners[-1])
            wnoise = np.random.randn(nmodes, nframes*n_inners[-1])
            for i in range(nt):
                n_inner = n_inners[i]
                # model.wnoise = wnoise[:,:,::2**(nt-1-i)]
                model.wnoise = wnoise[:,::2**(nt-1-i)]
                dt = T/nframes/n_inner
                kwargs_dynamic = {'n_inner': n_inner, 'nframes': nframes, 'dt': dt, 'save_add':f"_{n_inner}"}
                print(f"{i} ({nt})")
                model.dynamic(u0=None, **kwargs_dynamic)
            ufine = []
            for iframe in range(nframes): ufine.append(model.load(iter=iframe+1, add=f"_{n_inners[-1]}"))
            errv, errp = np.zeros(nt-1), np.zeros(nt-1)
            for i in range(nt-1):
                n_inner = n_inners[i]
                for iframe in range(nframes):
                    u = model.load(iter=iframe+1, add=f"_{n_inner}")
                    vf, pf, lf = model.vectorview.get_parts(ufine[iframe])
                    v, p, l = model.vectorview.get_parts(u)
                    v -= vf
                    p -= pf
                    model.save(u, name="error", iter=iframe, add=f"{i}")
                    n = (len(errallp)+1)*nframes
                    errormean[i] = (n-nframes)/n*errormean[i] + u/n
                    w = np.zeros_like(v)
                    model.Mass.dot(w, 1, v)
                    errv[i] = max(errv[i],v.dot(w))
                    errp[i] = max(errp[i], np.sum(p**2*model.mesh.dV))
            model.save(errormean, name="errormean")
            errallv.append(errv)
            errallp.append(errp)
            model.save(np.stack(errallv), name= "errallv")
            model.save(np.stack(errallp), name= "errallp")
            # self.load()
            # self.plot_rates()
    def plot_rates(self, fig, var = 'v'):
        model = self.model
        if var == 'v': errall = self.errallv
        else: errall = self.errallp
        nsamplesmax = errall.shape[0]
        x = np.log(self.n_inners[:-1])
        import matplotlib.gridspec as gridspec
        nplots = 3
        gs = gridspec.GridSpec(1, nplots, wspace=0.4, hspace=0.3)
        axs = [fig.add_subplot(gs[0, j]) for j in range(nplots)]
        nsamples = [(nsamplesmax+1)//4, (nsamplesmax+1)//2, nsamplesmax]
        for j in range(nplots):
            ns = nsamples[j]
            errvs = errall[:ns].mean(axis=0)
            res = scipy.stats.linregress(x, np.log(errvs))
            ls_s, ls_i = res.slope, res.intercept
            axs[j].set_title(f"nsamples={ns}")
            axs[j].plot(x, np.log(errvs), '-X', label=f"err_{var}")
            axs[j].plot(x, ls_i + x * ls_s, '--', label=f"{ls_i:5.2f} {ls_s:+5.2f}*x")
            axs[j].legend()
            axs[j].grid()
    def _colorbars(self, data, axp, axv, abs=False):
        pmax, pmin = -np.inf, np.inf
        vmax, vmin = -np.inf, np.inf
        for i in range(self.nt - 1):
            v, p = data[i][0], data[i][1]
            # v = data['point']['V']
            # p = data['cell']['P']
            if abs: p = np.absolute(p)
            vr = v.reshape(self.model.mesh.nnodes, self.model.ncomp)
            vnorm = np.linalg.norm(vr, axis=1)
            pmin = min(pmin, np.min(p))
            pmax = max(pmax, np.max(p))
            vmin = min(vmin, np.min(vnorm))
            vmax = max(vmax, np.max(vnorm))
        normp = matplotlib.colors.Normalize(vmin=pmin, vmax=pmax)
        clb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=normp, cmap=matplotlib.cm.jet), ax=axp)
        clb.ax.set_title('p')
        normv = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        clb = plt.colorbar(matplotlib.cm.ScalarMappable(norm=normv, cmap=matplotlib.cm.jet), ax=axv)
        clb.ax.set_title('v')
        return normp, normv, pmin, pmax, vmin, vmax

    def plot_errors(self, fig):
        model, errormean = self.model, self.errormean.reshape(self.nt-1,-1)
        x, y, tris = model.mesh.points[:,0], model.mesh.points[:,1], model.mesh.simplices
        gs = fig.add_gridspec(nrows=2, ncols=self.nt-1, hspace=0.1, wspace=0.05)
        axsv = [fig.add_subplot(gs[0, i]) for i in range(self.nt-1)]
        axsp = [fig.add_subplot(gs[1, i]) for i in range(self.nt-1)]
        datas = []
        for i in range(self.nt - 1):
            data = model.sol_to_data(errormean[i])
            datas.append([data['point']['V'],data['cell']['P']])
        normp, normv, pmin, pmax, vmin, vmax = self._colorbars(datas, axp=axsp[-1], axv=axsv[-1], abs=True)
        self.argscp = {'norm': normp, 'cmap': 'jet', 'edgecolors':'k'}
        self.argscfv = {'levels': 32, 'norm': normv, 'cmap': 'jet'}
        self.argscv = {'colors': 'k', 'levels': np.linspace(vmin, vmax, 32)}
        for i in range(self.nt - 1):
            axp, axv = axsp[i], axsv[i]
            axp.get_xaxis().set_visible(False)
            axp.get_yaxis().set_visible(False)
            axv.get_xaxis().set_visible(False)
            axv.get_yaxis().set_visible(False)
            data = model.sol_to_data(errormean[i])
            v = data['point']['V']
            p = data['cell']['P']
            vr = v.reshape(model.mesh.nnodes,model.ncomp)
            vnorm = np.linalg.norm(vr, axis=1)
            cnt = axp.tripcolor(x, y, tris, facecolors=p, **self.argscp)
            cnt = axv.tricontourf(x, y, tris, vnorm, **self.argscfv)
            # axv.tricontour(x, y, tris, vnorm, **self.argscv)

    def plot_solution(self, fig):
        from matplotlib import animation
        import tools.animdata
        gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0.1, wspace=0.05)
        axs = [fig.add_subplot(gs[i, 0]) for i in range(2)]
        for ax in axs:
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        datas = []
        for iframe in range(self.nframes):
            sol = self.model.load(iter=iframe, add=f"_{self.n_inners[-1]}")
            data = self.model.sol_to_data(sol)
            datas.append([data['point']['V'],data['cell']['P']])
        class ToAnimate():
            def __init__(self, model, axs, datas, normp, normv, vmin, vmax):
                x, y, tris = model.mesh.points[:, 0], model.mesh.points[:, 1], model.mesh.simplices
                self.model, self.axs, self.datas = model, axs, datas
                self.argscp = {'norm': normp, 'cmap': 'jet', 'edgecolors': 'k'}
                self.argscfv = {'levels': 32, 'norm': normv, 'cmap': 'jet'}
                self.argscv = {'colors': 'k', 'levels': np.linspace(vmin, vmax, 32)}
                v, p = datas[0][0], datas[0][1]
                vr = v.reshape(model.mesh.nnodes, model.ncomp)
                vnorm = np.linalg.norm(vr, axis=1)
                self.trip_p = axs[1].tripcolor(x, y, tris, facecolors=p, **self.argscp)
                self.trip_v = axs[0].tricontourf(x, y, tris, vnorm, **self.argscfv)
                self.qv = axs[0].quiver(x, y, vr[:, 0], vr[:, 1], units='xy')
                self.title = axs[0].set_title(f"Iter --")
            def __call__(self, iter):
                # self.title.set_text(f"Iter {iter}")
                axs[0].set_title(f"Iter {iter}")
                model = self.model
                x, y, tris = model.mesh.points[:, 0], model.mesh.points[:, 1], model.mesh.simplices
                v, p = self.datas[iter][0], self.datas[iter][1]
                vr = v.reshape(self.model.mesh.nnodes, self.model.ncomp)
                vnorm = np.linalg.norm(vr, axis=1)
                # self.axs[0].collections = []
                # self.axs[1].collections = []
                self.trip_p.set_array(p)
                # self.trip_v.set_array(vnorm)
                for c in self.trip_v.collections: c.remove()
                # cntp = self.axs[1].tripcolor(x, y, tris, facecolors=p, **self.argscp)
                self.trip_v = self.axs[0].tricontourf(x, y, tris, vnorm, **self.argscfv)
                self.qv = axs[0].quiver(x, y, vr[:, 0], vr[:, 1], units='xy')
                return self.trip_v, self.trip_p, self.qv,

        # anim = tools.animdata.AnimData(fig=fig, axs=axs, data=datas, pltfct=self.plot_anim, blit=False)
        normp, normv, pmin, pmax, vmin, vmax = self._colorbars(datas, axp=axs[1], axv=axs[0], abs=True)
        toanim = ToAnimate(self.model, axs, datas, normp, normv, vmin, vmax)
        self.paused = False
        self.animation = animation.FuncAnimation(fig, toanim, frames=len(datas), blit=False)
        fig.canvas.mpl_connect('button_press_event', self.toggle_pause)
    def toggle_pause(self, *args, **kwargs):
        if self.paused: self.animation.resume()
        else: self.animation.pause()
        self.paused = not self.paused

    def plot_normL2(self, fig):
        model = self.model
        M = model.computeMassMatrix()
        gs = fig.add_gridspec(nrows=2, ncols=1, hspace=0.1, wspace=0.05)
        axs = [fig.add_subplot(gs[i, 0]) for i in range(2)]
        vl2, pl2 = [], []
        for iframe in range(self.nframes):
            sol = model.load(iter=iframe, add=f"_{self.n_inners[-1]}")
            v, p, l = model.vectorview.get_parts(sol)
            w = np.zeros_like(v)
            M.dot(w, 1, v)
            vl2.append(v.dot(w))
            pl2.append(np.sum(p ** 2 * model.mesh.dV))
        axs[0].plot(vl2)
        axs[0].set_title("|v|")
        axs[1].plot(pl2)
        axs[1].set_title("|p|")


#-----------------------------------------------------------------------------#
if __name__ == "__main__":
    RS = RunStokes(h=0.04)
    RS.run()
# plot(h=h)