# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""
import os, shutil, pathlib
import numpy as np
import scipy.sparse.linalg as splinalg
from scipy.optimize import newton_krylov

import src.tools.analyticalfunction
import src.tools.timer
import src.tools.iterationcounter
import src.models.problemdata
from src.tools.analyticalfunction import AnalyticalFunction
import src.solvers

# import warnings
# warnings.filterwarnings("error")

#=================================================================#
class Model(object):
    def __format__(self, spec):
        if spec=='-':
            repr = f"fem={self.fem}"
            repr += f"\tlinearsolver={self.linearsolver}"
            return repr
        return self.__repr__()
    def __repr__(self):
        if hasattr(self, 'mesh'):
            repr = f"mesh={self.mesh}"
        else:
            repr = "no mesh\n"
        repr += f"problemdata={self.problemdata}"
        repr += f"\nlinearsolver={self.linearsolver}"
        repr += f"\ndisc_params={self.disc_params}"
        repr += f"\n{self.timer}"
        return repr
    def __init__(self, **kwargs):
        self.stack_storage = kwargs.pop("stack_storage", True)
        self.mode = kwargs.pop('mode', 'linear')
        self.verbose = kwargs.pop('verbose', 0)
        self.timer = src.tools.timer.Timer(verbose=self.verbose)
        self.application = kwargs.pop('application', None)
        if self.application is None:
            raise ValueError(f"Model needs application (since 22/04/23)")
        self.problemdata = self.application.problemdata
        self.ncomp = self.problemdata.ncomp
        if not hasattr(self,'linearsolver'):
            self.linearsolver = kwargs.pop('linearsolver', 'spsolve')
        if self.application.has_exact_solution:
            self._generatePDforES = True
        else:
            self._generatePDforES = False
        # femparams = kwargs.pop('femparams', {})
        # self.createFem(femparams)
        self.disc_params = kwargs.pop('disc_params', {})
        self.createFem()
        self.setMesh(self.application.createMesh())
        print(f"{self.__class__.__name__} {self.mesh=} {self.linearsolver=}")
        if not hasattr(self,'scale_ls'):
            self.scale_ls = kwargs.pop('scale_ls', True)
        if 'newton_stopping_parameters' in kwargs:
            self.newton_stopping_parameters = kwargs.pop('newton_stopping_parameters')
        else:
            maxiter = kwargs.pop('newton_maxiter', 10)
            rtol = kwargs.pop('newton_rtol', 1e-6)
            self.newton_stopping_parameters = src.solvers.newtondata.StoppingParamaters(maxiter=maxiter, rtol=rtol)
        if isinstance(self.linearsolver, str):
            self.newton_stopping_parameters.addname = self.linearsolver
        else:
            self.newton_stopping_parameters.addname = self.linearsolver['method']
        datadir_def =  pathlib.Path.home().joinpath( 'data_dir', f"_{self.__class__.__name__}"+f"_{self.application.__class__.__name__}")
        self.datadir = kwargs.pop('datadir', datadir_def)
        if kwargs.pop("clean_data",True):
            try: shutil.rmtree(self.datadir)
            except: pass
        pathlib.Path(self.datadir).mkdir(parents=True, exist_ok=True)
        with open(self.datadir / "model", "w") as file:
            file.write(str(self))
        # check for unused arguments
        if len(kwargs.keys()):
            raise ValueError(f"*** unused arguments {kwargs=}")
    def createFem(self):
        raise NotImplementedError(f"createFem has to be overwritten")
    def setMesh(self, mesh):
        self.timer.reset_all()
        self.problemdata.check(mesh)
        self.mesh = mesh
        if self.verbose: print(f"{self.mesh=}")
        self._setMeshCalled = True
        if hasattr(self,'_generatePDforES') and self._generatePDforES:
            self.generatePoblemDataForAnalyticalSolution()
            self._generatePDforES = False
    def solve(self): return self.static(method=self.mode)
    # def setParameter(self, paramname, param):
    #     assert 0
    def dirichletfct(self):
        if self.ncomp > 1:
            # def _solexactdir(x, y, z):
            #     return [self.problemdata.solexact[icomp](x, y, z) for icomp in range(self.ncomp)]
            # return _solexactdir
            from functools import partial
            solexact = self.problemdata.solexact
            def _solexactdir(x, y, z, icomp):
                return solexact[icomp](x, y, z)
            return [partial(_solexactdir, icomp=icomp) for icomp in range(self.ncomp)]
        else:
            return self.problemdata.solexact
            def _solexactdir(x, y, z):
                return self.problemdata.solexact(x, y, z)
        return _solexactdir
    def generatePoblemDataForAnalyticalSolution(self):
        bdrycond = self.problemdata.bdrycond
        self.problemdata.solexact = self.defineAnalyticalSolution(exactsolution=self.application.exactsolution, random=self.application.random_exactsolution)
        # print("self.problemdata.solexact", self.problemdata.solexact)
        solexact = self.problemdata.solexact
        self.problemdata.params.fct_glob['rhs'] = self.defineRhsAnalyticalSolution(solexact)
        for color in self.mesh.bdrylabels:
            if color in bdrycond.type and bdrycond.type[color] in ["Dirichlet","dirichlet"]:
                bdrycond.fct[color] = self.dirichletfct()
            else:
                if color in bdrycond.type:
                    cmd = "self.define{}AnalyticalSolution(self.problemdata,{})".format(bdrycond.type[color], color)
                    # print(f"cmd={cmd}")
                    bdrycond.fct[color] = eval(cmd)
                else:
                    bdrycond.fct[color] = self.defineBdryFctAnalyticalSolution(color, solexact)
    def defineAnalyticalSolution(self, exactsolution, random=True):
        dim = self.mesh.dimension
        # print(f"defineAnalyticalSolution: {dim=} {self.ncomp=}")
        return src.tools.analyticalfunction.analyticalSolution(exactsolution, dim, self.ncomp, random)
    def compute_cell_vector_from_params(self, name, params):
        if name in params.fct_glob:
            fct = np.vectorize(params.fct_glob[name])
            arr = np.empty(self.mesh.ncells)
            for color, cells in self.mesh.cellsoflabel.items():
                xc, yc, zc = self.mesh.pointsc[cells].T
                arr[cells] = fct(color, xc, yc, zc)
        elif name in params.scal_glob:
            arr = np.full(self.mesh.ncells, params.scal_glob[name])
        elif name in params.scal_cells:
            arr = np.empty(self.mesh.ncells)
            for color in params.scal_cells[name]:
                arr[self.mesh.cellsoflabel[color]] = params.scal_cells[name][color]
        else:
            msg = f"{name} should be given in 'fct_glob' or 'scal_glob' or 'scal_cells' (problemdata.params)"
            raise ValueError(msg)
        return arr
    def initsolution(self, b):
        if isinstance(b,tuple):
            raise NotImplementedError()
            return [np.copy(bi) for bi in b]
        return np.copy(b)
    def computelinearSolver(self, A):
        if isinstance(self.linearsolver,str):
            args = {'method': self.linearsolver}
        else:
            args = self.linearsolver.copy()
        if args['method'] != 'spsolve':
            if self.scale_ls and hasattr(A,'scale_A'):
                A.scale_A()
                args['scale'] = self.scale_ls
            args['matrix'] = A
            if hasattr(A,'matvec'):
                args['matvec'] = A.matvec
                args['n'] = A.nall
            else:
                args['matvec'] = lambda x: np.matmul(A,x)
                args['n'] = A.shape[0]
        return src.solvers.linalg.getLinearSolver(args=args)
    def static(self, **kwargs):
        method = kwargs.pop('method','newton')
        # raise ValueError(f"{method=}")
        u = kwargs.pop('u',None)
        if 'maxiter' in kwargs: self.newton_stopping_parameters.maxiter = kwargs.pop('maxiter')
        if 'rtol' in kwargs: self.newton_stopping_parameters.rtol = kwargs.pop('rtol')
        # self.timer.reset_all()
        result = src.models.problemdata.Results()
        if not self._setMeshCalled: self.setMesh(self.mesh)
        self.timer.add('setMesh')
        self.b = self.computeRhs()
        # raise ValueError(f"{self.b.shape=}")
        if u is None:
            u = self.initsolution(self.b)
        self.timer.add('rhs')
        if method == 'linear':
            try:
                self.A = self.computeMatrix()
                self.timer.add('matrix')
                self.LS = self.computelinearSolver(self.A)
                self.timer.add('solver')
                u = self.LS.solve(A=self.A, b=self.b, x0=u)
                niterlin = self.LS.niter
            except Warning:
                raise ValueError(f"matrix is singular {self.A.shape=} {self.A.diagonal()=}")
            self.timer.add('solve')
            iter={'lin':niterlin}
        else:
            if method == 'newton':
                u, info = src.solvers.newton.newton(u, f=self.computeDefect, computedx=self.computeDx,
                                                         verbose=True, sdata=self.newton_stopping_parameters)
                iter={'nonlin':info.iter, 'lin':np.mean(info.liniter)}
                result.newtoninfo = info
                if not info.success:
                    print(f"*** {info.failure=}")
            elif method == 'newtonkrylov':
                counter = src.tools.iterationcounter.IterationCounterWithRes(name=method, disp=1, callback_type='x,Fx')
                n = u.shape[0]
                class NewtonPrec(splinalg.LinearOperator):
                    def __init__(self, n, model, u):
                        super().__init__(shape=(n,n), dtype=float)
                        self.model = model
                        if not hasattr(self.model,'A'):
                            self.model.A = self.model.computeMatrix(u=u)
                            self.model.LS = self.model.computelinearSolver(self.model.A)
                    def _matvec(self, b):
                        A, LS = self.model.A, self.model.LS
                        du = LS.solve(A=A, b=b, maxiter=1)
                        niterlin = LS.niter
                        # print(f"{__class__.__name__} matvec {np.linalg.norm(b)=} {niterlin=}")
                        return du
                    def update(self, u, b):
                        # print(f"{__class__.__name__} update {np.linalg.norm(u)=}  {np.linalg.norm(b)=}")
                        self.model.A = self.model.computeMatrix(u=u)
                        self.model.LS = self.model.computelinearSolver(self.model.A)

                u = newton_krylov(F=self.computeDefect, xin=u, method='lgmres',
                                  maxiter=self.newton_stopping_parameters.maxiter,
                                  f_rtol=self.newton_stopping_parameters.rtol,
                                  inner_maxiter=3, inner_M=NewtonPrec(n, self, u), callback=counter)
                iter = {'lin': -1, 'nlin': counter.niter}
            else:
                raise ValueError(f"unknwon {method=}")
        pp = self.postProcess(u)
        if hasattr(self.application, "changepostproc"):
            self.application.changepostproc(pp['scalar'])
        self.timer.add('postp')
        result.setData(pp, timer=self.timer, iter=iter)
        self.save(u=u)
        return result, u
    def computeDefect(self, u):
        return self.computeForm(u)-self.b
    def computeForm(self, u):
        return self.A@u
    def initialCondition(self, interpolate=True):
        #TODO: higher order interpolation
        if not 'initial_condition' in self.problemdata.params.fct_glob:
            raise ValueError(f"missing 'initial_condition' in {self.problemdata.params.fct_glob=}")
        if not self._setMeshCalled: self.setMesh(self.mesh)
        ic = AnalyticalFunction(self.problemdata.params.fct_glob['initial_condition'])
        fp1 = self.fem.interpolate(ic)
        if interpolate:
            return fp1
        self.Mass = self.fem.computeMassMatrix()
        b = np.zeros(self.fem.nunknowns())
        self.fem.massDot(b, fp1)
        u, niter = self.solvelinear(self.Mass, b, u=fp1)
        return u
    def defect_dynamic(self, rhs, u):
        return self.computeForm(u)-rhs + self.Mass.dot(u)/(self.theta * self.dt)
    def computeMatrixConstant(self, coeffmass):
        return self.computeMatrix(coeffmass=coeffmass)
    def rhs_dynamic(self, rhs, u, Aimp, time, dt, theta):
        rhs += 1 / (theta * theta * dt) * self.Mass.dot(u)
        rhs += (theta-1)/theta * Aimp.dot(u)
        # rhs2 = self.computeRhs()
        rhs += (1 / theta) * self.computeRhs()
    def computeDx(self, b, u, info):
        computeMatrix = False
        rtol = 1e-5
        if (not hasattr(self, 'timeiter')) and (info.iter==0 or info.bad_convergence):
            computeMatrix=True
        if hasattr(self, 'timeiter') and self.timeiter==0 and info.iter==0:
            computeMatrix=True
        elif hasattr(self, 'timeiter') and info.bad_convergence:
            computeMatrix = True
        if hasattr(info,'rhor'):
            rtol = min(0.01, info.rhor)
            rtol = max(rtol, info.tol_missing)
        if computeMatrix:
            if not hasattr(self, 'timeiter'):
                coeffmass=None
            else:
                coeffmass=self.coeffmass
            self.A = self.computeMatrix(u=u, coeffmass=coeffmass)
            if hasattr(self,'LS'):
                # self.LS = self.computelinearSolver(self.A)
                self.A.scale_A()
                self.LS.update(self.A)
            else:
                self.LS = self.computelinearSolver(self.A)
        try:
            du = self.LS.solve(A=self.A, b=b, rtol=rtol)
            niter = self.LS.niter
            if niter==self.LS.maxiter:
                return du, niter, False
        except Warning:
            raise ValueError(f"matrix is singular {self.A.shape=} {self.A.diagonal()=}")
        self.timer.add('solve_linear')
        return du, niter, True
    def dynamic(self, u0, t_span, nframes, **kwargs):
        # TODO: passing time
        """
        u_t + A(u) = f, u(t_0) = u_0
        M(u^{n+1}-u^n)/dt + A(theta u^{n+1}+(1-theta)u^n) = theta f^{n+1}+(1-theta)f^n
        :param u0: initial condition
        :param t_span: time interval bounds (tuple)
        :param nframes: number of frames to store
        :param dt: time-step (fixed for the moment!)
        :param mode: (only linear for the moment!)
        :param callback: if given function called for each frame with argumntes t, u
        :param method: CN or BE for Crank-Nicolson (a=1/2) or backward Euler (a=1)
        :return: results with data per frame
        """
        from functools import partial

        if t_span[0]>=t_span[1]: raise ValueError(f"something wrong in {t_span=}")
        import math
        callback = kwargs.pop('callback', None)
        dt = kwargs.pop('dt', (t_span[1]-t_span[0])/(10*nframes))
        dtmin = kwargs.pop('dtmin', 1e-10*dt)
        theta = kwargs.pop('theta', 0.8)
        verbose = kwargs.pop('verbose', True)
        maxiternewton = kwargs.pop('maxiternewton', 10)
        rtolnewton = kwargs.pop('rtolnewton', 1e-3)
        sdata = kwargs.pop('sdata', src.solvers.newtondata.StoppingParamaters(maxiter=maxiternewton, rtol=rtolnewton))
        output_vtu = kwargs.pop('output_vtu', False)
        if len(kwargs):
            raise ValueError(f"unused arguments: {kwargs.keys()}")

        if not dt or dt<=0: raise NotImplementedError(f"needs constant positive 'dt")
        result = src.models.problemdata.Results(nframes)
        # self.timer.add('init')
        # self.timer.add('matrix')
        u = u0
        self.time = t_span[0]
        # rhs=None
        self.rhs = np.empty_like(u, dtype=float)
        if isinstance(self.linearsolver, str):
            sdata.addname = self.linearsolver
        else:
            sdata.addname = self.linearsolver['method']
        if not hasattr(self, 'Mass'):
            self.Mass = self.computeMassMatrix()
        self.coeffmass = 1 / dt / theta
        if not hasattr(self, 'Aconst'):
            Aconst = self.computeMatrixConstant(coeffmass=self.coeffmass)
            if self.linearsolver=="pyamg":
                self.pyamgml = self.build_pyamg(Aconst)
        self.theta, self.dt = theta, dt
        times = np.linspace(t_span[0], t_span[1], nframes+1)
        info_new = None
        count_smallres = 0
        self.timeiter = 0
        for iframe in range(nframes):
            if verbose: print(f"*** {self.time=} {iframe=} {theta=} {self.dt=}")
            while self.time<times[iframe+1]:
                self.rhs.fill(0)
                self.rhs_dynamic(self.rhs, u, Aconst, self.time, dt, theta)
                # print(f"{np.linalg.norm(self.rhs)=} {np.linalg.norm(u)=}")
                self.timer.add('rhs')
                self.time += dt
                self.uold = u.copy()
                u, info_new = src.solvers.newton.newton(u, f=partial(self.defect_dynamic, self.rhs),
                                                            computedx=self.computeDx,
                                                            verbose=True, sdata=sdata, iterdata=info_new)
                self.timer.add('newton')
                if not info_new.success and info_new.failure in ["residual too small","correction too small"]:
                    count_smallres += 1
                    if count_smallres == 3:
                        print("got stationary solution")
                        return result
                elif not info_new.success:
                    u = self.uold
                    self.time -= dt
                    dtold = dt
                    dt *= 0.5
                    if dt <= dtmin:
                        raise ValueError(f"giving up {dt=}")
                    self.dt = dt
                    coeffmassold = self.coeffmass
                    self.coeffmass = 1 / dt / theta
                    Aconst = self.computeMatrixConstant(coeffmass=self.coeffmass, coeffmassold=coeffmassold)
                    self.A = self.computeMatrix(u=u, coeffmass=self.coeffmass)
                    self.LS = self.update(self.A)
                    # self.LS = self.computelinearSolver(self.A)
                    print(f"*** {info_new.failure=} {dtold=} {dt=}")
                    info_new.success = True
                else:
                    count_smallres = 0
                # self.timer.add('solve')
                self.timeiter += 1
            info_new.totaliter = 0
            info_new.totalliniter = 0
            pp = self.postProcess(u)
            if hasattr(self.application, "changepostproc"):
                self.application.changepostproc(pp['scalar'])
            result.addData(iframe, pp, time=self.time, iter=info_new.totaliter, liniter=info_new.totalliniter)
            if callback: callback(self.time, u)
            # save data
            self.save(u=u, iter=iframe)
            if output_vtu:
                data = self.sol_to_data(u, single_vector=False)
                filename = "sol" + f"_{iframe:05d}" + ".vtu"
                self.mesh.write(self.datadir/filename, data=data)
        result.save(self.datadir)
        return result

    def dynamic_linear(self, u0, t_span, nframes, dt=None, callback=None, method='CN', verbose=1):
        # TODO: passing time
        """
        u_t + A u = f, u(t_0) = u_0
        M(u^{n+1}-u^n)/dt + a Au^{n+1} + (1-a) A u^n = f
        (M/dt+aA) u^{n+1} =  f + (M/dt -(1-a)A)u^n
                          =  f + 1/a (M/dt) u^n - (1-a)/a (M/dt+aA)u^n
        (M/(a*dt)+A) u^{n+1} =  (1/a)*f + (M/(a*dt)-(1-a)/a A)u^n
                             =  (1/a)*f + 1/(a*a*dt) M u^n  - (1-a)/a*(M/(a*dt)+A)u^n
        :param u0: initial condition
        :param t_span: time interval bounds (tuple)
        :param nframes: number of frames to store
        :param dt: time-step (fixed for the moment!)
        :param mode: (only linear for the moment!)
        :param callback: if given function called for each frame with argumntes t, u
        :param method: CN or BE for Crank-Nicolson (a=1/2) or backward Euler (a=1)
        :return: results with data per frame
        """
        if not dt or dt<=0: raise NotImplementedError(f"needs constant positive 'dt")
        if t_span[0]>=t_span[1]: raise ValueError(f"something wrong in {t_span=}")
        if method not in ['BE','CN']: raise ValueError(f"unknown method {method=}")
        if method == 'BE': a = 1
        else: a = 0.5
        import math
        nitertotal = math.ceil((t_span[1]-t_span[0])/dt)
        if nframes > nitertotal:
            raise ValueError(f"Maximum valiue for nframes is {nitertotal=}")
        niter = nitertotal//nframes
        result = src.models.problemdata.Results(nframes)
        self.timer.add('init')
        if not hasattr(self, 'Mass'):
            self.Mass = self.fem.computeMassMatrix()
        if not hasattr(self, 'Aconst'):
            Aconst = self.computeMatrix(coeffmass=1 / dt / a)
            if self.linearsolver=="pyamg":
                self.pyamgml = self.build_pyamg(Aconst)
        self.timer.add('matrix')
        u = u0
        self.time = t_span[0]
        # rhs=None
        rhs = np.empty_like(u, dtype=float)
        # will be create by computeRhs()
        niterslinsol = np.zeros(niter, dtype=int)
        expl = (a-1)/a
        for iframe in range(nframes):
            if verbose: print(f"*** {self.time=} {iframe=} {niter=} {nframes=} {a=}")
            for iter in range(niter):
                self.time += dt
                rhs.fill(0)
                rhs += 1/(a*a*dt)*self.Mass.dot(u)
                rhs += expl*Aconst.dot(u)
                # print(f"@1@{np.min(u)=} {np.max(u)=} {np.min(rhs)=} {np.max(rhs)=}")
                rhs2 = self.computeRhs()
                rhs += (1/a)*rhs2
                # print(f"@2@{np.min(u)=} {np.max(u)=} {np.min(rhs)=} {np.max(rhs)=}")
                self.timer.add('rhs')
                # u, niterslinsol[iter] = self.solvelinear(self.ml, rhs, u=u, verbose=0)
                #TODO organiser solveur linéaire
                u, niterslinsol[iter] = self.solvelinear(Aconst, b=rhs, u=u)
                # print(f"{niterslinsol=} {np.linalg.norm(u)=}")
                # u, res = self.solve_pyamg(self.pyamgml, rhs, u=u, maxiter = 100)
                # u, niterslinsol[iter] = u, len(res)
                # print(f"@3@{np.min(u)=} {np.max(u)=} {np.min(rhs)=} {np.max(rhs)=}")
                self.timer.add('solve')
            result.addData(iframe, self.postProcess(u), time=self.time, iter=niterslinsol.mean())
            if callback: callback(self.time, u)
        return result

    def save(self, u, iter=None):
        solname = "sol"
        if iter is not None: solname += f"_{iter:05d}"
        # filename = os.path.join(self.dirname, solname)
        np.save(self.datadir/solname, u)
    def get_postprocs_dynamic(self):
        # filename = os.path.join(self.dirname, "time.npy")
        data = {'time': np.load(self.datadir/"time.npy"), 'postproc':{}}
        from pathlib import Path
        p = Path(self.datadir)
        for q in p.glob('postproc*.npy'):
            pname = '_'.join(str(q.parts[-1]).split('.')[0].split('_')[1:])
            # print(f"{pname=} {q=}")
            data['postproc'][pname] = np.load(q)
        return data
    def sol_to_vtu(self, **kwargs):
        niter = kwargs.pop('niter', None)
        suffix = kwargs.pop('suffix', '')
        solnamebase = "sol" + suffix
        if niter is None:
            u = kwargs.pop('u', None)
            if u is None:
                # filename = os.path.join(self.dirname, solnamebase + ".npy")
                u = np.load(self.datadir / solnamebase + ".npy")
            data = self.sol_to_data(u, single_vector=False)
            # filename = os.path.join(self.dirname, solnamebase + ".vtu")
            # self.mesh.write(filename, data=data)
            self.mesh.write(self.datadir / solnamebase + ".vtu", data=data)
            return
        for iter in range(niter):
            solname = solnamebase + f"_{iter:05d}"
            # filename = os.path.join(self.dirname, solname + ".npy")
            u = np.load(self.datadir / solnamebase + ".npy")
            data = self.sol_to_data(u, single_vector=False)
            # filename = os.path.join(self.dirname, solname + ".vtu")
            self.mesh.write(self.datadir/solname + ".vtu", data=data)

    def plot(self, **kwargs):
        u = kwargs.pop('u', None)
        fig = kwargs.pop('fig', None)
        gs = kwargs.pop('gs', None)
        if u is None:
            solname = "sol"
            iter = kwargs.pop('iter', None)
            if iter is not None: solname += f"_{iter:05d}"
            solname += ".npy"
            # filename = os.path.join(self.dirname, solname)
            u = np.load(self.datadir/solname)
        data = self.sol_to_data(u)
        # raise ValueError(f"{data=} {solname=}")
        import matplotlib.pyplot as plt
        if fig is None:
            if gs is not None:
                raise ValueError(f"got gs but no fig")
            fig = plt.figure(constrained_layout=True)
            appname = kwargs.pop('title', self.application.__class__.__name__)
            fig.suptitle(f"{appname}")
        if gs is None:
            gs = fig.add_gridspec(1, 1)[0,0]
        if self.mesh.dimension==2:
            self._plot2d(data=data, fig=fig, gs=gs, **kwargs)
        else:
            import pyvista
            tets = self.mesh.simplices
            ntets = tets.shape[0]
            celltypes = pyvista.CellType.TETRA * np.ones(ntets, dtype=int)
            cells = np.insert(tets, 0, 4, axis=1).ravel()
            mesh = pyvista.UnstructuredGrid(cells, celltypes, self.mesh.points)
            self._plot3d(mesh=mesh, data=data, fig=fig, gs=gs, **kwargs)

# ------------------------------------- #

if __name__ == '__main__':
    raise ValueError("unit test to be written")


    # def solvelinear(self, A, b, u=None, verbose=0, disp=0):
    #     if spsp.issparse(A):
    #         if len(b.shape)!=1 or len(A.shape)!=2 or b.shape[0] != A.shape[0]:
    #             raise ValueError(f"{A.shape=} {b.shape=}")
    #     if self.linearsolver is None: solver = self.linearsolver
    #     if not hasattr(self, 'info'): self.info={}
    #     if self.linearsolver not in self.linearsolvers: solver = "spsolve"
    #     if self.linearsolver == 'spsolve':
    #         return splinalg.spsolve(A, b), 1
    #         return splinalg.spsolve(A, b, permc_spec='COLAMD'), 1
    #     elif self.linearsolver in ['gmres','lgmres','bicgstab','cg','gcrotmk']:
    #         if self.linearsolver == 'cg':
    #             def gaussSeidel(A):
    #                 dd = A.diagonal()
    #                 D = spsp.dia_matrix(A.shape)
    #                 D.setdiag(dd)
    #                 L = spsp.tril(A, -1)
    #                 U = spsp.triu(A, 1)
    #                 return splinalg.factorized(D + L)
    #             M2 = gaussSeidel(A)
    #         else:
    #             # defaults: drop_tol=0.0001, fill_factor=10
    #             M2 = splinalg.spilu(A.tocsc(), drop_tol=0.1, fill_factor=3)
    #         M = splinalg.LinearOperator(A.shape, lambda x: M2.solve(x))
    #         counter = src.tools.iterationcounter.IterationCounter(name=self.linearsolver, disp=disp)
    #         args=""
    #         cmd = "splinalg.{}(A, b, M=M, tol=1e-14, callback=counter {})".format(self.linearsolver,args)
    #         u, info = eval(cmd)
    #         # print(f"{u=}")
    #         return u, counter.niter
    #     elif self.linearsolver == 'pyamg':
    #         if not hasattr(self, 'pyamgml'):
    #             self.pyamgml = self.build_pyamg(A)
    #         maxiter = 100
    #         u, res = self.solve_pyamg(self.pyamgml, b, u, maxiter)
    #         if len(res) >= maxiter: raise ValueError(f"***no convergence {res=}")
    #         if(verbose): print('niter ({}) {:4d} ({:7.1e})'.format(solver, len(res),res[-1]/res[0]))
    #         return u, len(res)
    #     else:
    #         raise NotImplementedError("unknown solve '{}'".format(self.linearsolver))
