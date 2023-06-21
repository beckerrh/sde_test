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
        print(f"{self.__class__.__name__} {self.stack_storage=} {self.singleA=} {self.mesh=} {self.linearsolver=}")
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
        datadir_def_name = f"{self.__class__.__name__}"+f"_{self.application.__class__.__name__}"
        if 'datadir_add' in kwargs:
            datadir_def_name += kwargs.pop('datadir_add')
        datadir_def =  pathlib.Path.home().joinpath( 'data_dir', datadir_def_name)
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
        if hasattr(self, 'LS'):
            del self.LS
        if hasattr(self,'_generatePDforES') and self._generatePDforES:
            self.generatePoblemDataForAnalyticalSolution()
            self._generatePDforES = False
    def solve(self):
        if self.mode=='dynamic':
            return self.dynamic()
        return self.static(method=self.mode)
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
        if hasattr(self, 'time'):
            self.problemdata.params.fct_glob['initial_condition'] = self.defineInitialConditionAnalyticalSolution(solexact)
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
        # print(f"{type(b)=}")
        if isinstance(b,tuple):
            raise NotImplementedError()
            return [np.copy(bi) for bi in b]
        return b.copy()
    def computelinearSolver(self, A):
        if isinstance(self.linearsolver,str):
            args = {'method': self.linearsolver}
        else:
            args = self.linearsolver.copy()
        if args['method'] != 'spsolve':
            if self.scale_ls:
                A.scale_matrix()
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
        self.timer.reset_all()
        result = src.models.problemdata.Results()
        if not self._setMeshCalled: self.setMesh(self.mesh)
        self.timer.add('setMesh')
        self.b = kwargs.pop('b',self.computeRhs())
        # self.b = self.computeRhs()
        # raise ValueError(f"{self.b.shape=}")
        if u is None:
            u = self.initsolution(self.b)
        self.timer.add('rhs')
        if method == 'linear':
            try:
                if not hasattr(self,'A'):
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
        if not hasattr(self, 'Mass'):
            self.Mass = self.fem.computeMassMatrix()
        b = np.zeros(self.fem.nunknowns())
        self.fem.massDot(b, fp1)
        u, niter = self.solvelinear(self.Mass, b, u=fp1)
        return u
    def defect_dynamic(self, rhs, u):
        return self.computeForm(u)-rhs + self.Mass.dot(u)/(self.theta * self.dt)
    def computeMatrixConstant(self, coeffmass):
        return self.computeMatrix(coeffmass=coeffmass)
    def rhs_dynamic(self, rhs, u, Aconst, time, dt, theta, semi_implicit):
        rhs += 1 / (theta * theta * dt) * self.Mass.dot(u)
        rhs += (theta-1)/theta * Aconst.dot(u)
        # rhs2 = self.computeRhs()
        rhs += (1 / theta) * self.computeRhs()
    def newMatrix(self, u):
        if not hasattr(self, 'timeiter'):
            coeffmass = None
        else:
            coeffmass = self.coeffmass
        self.A = self.computeMatrix(u=u, coeffmass=coeffmass)
        self.A.scale_matrix()
        if hasattr(self, 'LS'):
            # self.LS = self.computelinearSolver(self.A)
            self.LS.update(self.A)
        else:
            self.LS = self.computelinearSolver(self.A)
    def computeDx(self, b, u, info):
        computeMatrix = False
        if (not hasattr(self, 'timeiter')) and (info.iter==0 or info.bad_convergence):
            computeMatrix=True
        if hasattr(self, 'timeiter') and self.timeiter==0 and info.iter==0:
            computeMatrix=True
        elif hasattr(self, 'timeiter') and info.bad_convergence:
            computeMatrix = True
        if computeMatrix:
            self.newMatrix(u)
        rtol = 1e-5
        if hasattr(info,'rhor'):
            rtol = min(0.01, info.rhor)
            rtol = max(rtol, info.tol_missing)
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
        if not self.vectorview.n() == u0.size:
            raise ValueError(f"needs u0 of shape {self.vectorview.ncomps=}")
        if t_span[0]>=t_span[1]: raise ValueError(f"something wrong in {t_span=}")
        import math
        callback = kwargs.pop('callback', None)
        dt = kwargs.pop('dt', (t_span[1]-t_span[0])/(10*nframes))
        dtmin = kwargs.pop('dtmin', 1e-10*dt)
        theta = kwargs.pop('theta', 0.8)
        verbose = kwargs.pop('verbose', True)
        newton_verbose = kwargs.pop('newton_verbose', True)
        maxiternewton = kwargs.pop('maxiternewton', 4)
        rtolnewton = kwargs.pop('rtolnewton', 1e-3)
        rtolsemi_imp = kwargs.pop('rtolsemi_imp', 1e-8)
        sdata = kwargs.pop('sdata', src.solvers.newtondata.StoppingParamaters(maxiter=maxiternewton, rtol=rtolnewton))
        output_vtu = kwargs.pop('output_vtu', False)
        semi_implicit = kwargs.pop('semi_implicit', False)
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
        count_smallres = 0
        self.timeiter = 0
        info_new = src.solvers.newtondata.IterationData(rtolsemi_imp)
        if verbose:
            print(30*"*" + f" {theta=} "+30*"*")
            print(f"*** {'t':12s} {'it':6s} {'dt':6s} {'n_lin_av':8s} {'n_nl_av':8s} {'n_bad':6s} {'nnew':4s}")
        for iframe in range(nframes):
            info_new.totaliter = 0
            info_new.totalliniter = 0
            info_new.bad_convergence_count = 0
            info_new.calls = 0
            pp = self.postProcess(u)
            if hasattr(self.application, "changepostproc"):
                self.application.changepostproc(pp['scalar'])
            result.addData(iframe, pp, time=self.time, iter=info_new.totaliter, liniter=info_new.totalliniter)
            if callback: callback(self.time, u)
            self.save(u=u, iter=iframe)
            if output_vtu:
                data = self.sol_to_data(u, single_vector=False)
                filename = "sol" + f"_{iframe:05d}" + ".vtu"
                self.mesh.write(self.datadir/filename, data=data)
            while self.time<times[iframe+1]:
                self.rhs.fill(0)
                self.time += dt
                self.application.time = self.time
                self.rhs_dynamic(self.rhs, u, Aconst, self.time, dt, theta, semi_implicit)
                # print(f"{np.linalg.norm(self.rhs)=} {np.linalg.norm(u)=}")
                # self.timer.add('rhs')
                self.uold = u.copy()
                solver_success = True
                if semi_implicit:
                   self.newMatrix(u)
                   u = self.LS.solve(A=self.A, b=self.rhs, rtol=rtolsemi_imp, verbose=True)
                   info_new.liniter.append(self.LS.niter)
                   if self.LS.niter == self.LS.maxiter:
                       print(f"*** solver failure {rtolsemi_imp=} {self.LS.niter=}")
                       solver_success = False
                else:
                    u, info_new = src.solvers.newton.newton(u, f=partial(self.defect_dynamic, self.rhs),
                                                            computedx=self.computeDx,
                                                            verbose=newton_verbose, sdata=sdata, iterdata=info_new)
                    # self.timer.add('newton')
                    if not info_new.success and info_new.failure in ["residual too small","correction too small"]:
                        count_smallres += 1
                        if count_smallres == 3:
                            print("got stationary solution")
                            return result
                    elif not info_new.success:
                        solver_success = False
                    else:
                        count_smallres = 0
                # self.timer.add('solve')
                if not solver_success:
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
                    self.A.scale_matrix()
                    self.LS.update(self.A)
                    print(f"*** {info_new.failure=} {dtold=} {dt=}")
                    info_new.success = True

                # self.timer.add('solve')
                self.timeiter += 1
            if verbose:
                print(f"*** {self.time:9.3e} {iframe:6d} {self.dt:8.2e} {info_new.niter_lin_mean():8.2f} {info_new.niter_mean():8.2f} {info_new.bad_convergence_count:8.2f} {info_new.calls:3d}")
            # info_new.totaliter = 0
            # info_new.totalliniter = 0
            # pp = self.postProcess(u)
            # if hasattr(self.application, "changepostproc"):
            #     self.application.changepostproc(pp['scalar'])
            # result.addData(iframe, pp, time=self.time, iter=info_new.totaliter, liniter=info_new.totalliniter)
            # if callback: callback(self.time, u)
            # self.save(u=u, iter=iframe)
            # if output_vtu:
            #     data = self.sol_to_data(u, single_vector=False)
            #     filename = "sol" + f"_{iframe:05d}" + ".vtu"
            #     self.mesh.write(self.datadir/filename, data=data)
        iframe += 1
        pp = self.postProcess(u)
        if hasattr(self.application, "changepostproc"):
            self.application.changepostproc(pp['scalar'])
        result.addData(iframe, pp, time=self.time, iter=info_new.totaliter, liniter=info_new.totalliniter)
        if callback: callback(self.time, u)
        self.save(u=u, iter=iframe)
        if output_vtu:
            data = self.sol_to_data(u, single_vector=False)
            filename = "sol" + f"_{iframe:05d}" + ".vtu"
            self.mesh.write(self.datadir / filename, data=data)
        result.save(self.datadir)
        return result

    def dynamic_linear(self, u0, t_span, nframes, dt=None, callback=None, method='CN', verbose=1):
        # TODO: passing time
        """
        u_t + A u = f, u(t_0) = u_0
        M(u^{n+1}-u^n)/dt + a Au^{n+1} + (1-a) A u^n = (f^{n+1}-f^n)/2
        (M/dt+aA) u^{n+1} =  f + (M/dt -(1-a)A)u^n
                          =  f + 1/a (M/dt) u^n - (1-a)/a (M/dt+aA)u^n
        C := (M/dt+aA)
        C u^{n+1} =  (1/a)*f + (M/(a*dt)-(1-a)/a A)u^n
                  =  (1/a)*f + 1/(a*a*dt) M u^n  + (a-1)/a* C * u^n
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
    def save(self, u, iter=None, datadir=None, name= "sol", add=''):
        if datadir is None: datadir=self.datadir
        if add: name += add
        if iter is not None: name += f"_{iter:05d}"
        np.save(datadir/name, u)
    def load(self, iter=None, datadir=None, name= "sol", add=''):
        if add: name += add
        if iter is not None: name += f"_{iter:05d}"
        if datadir is None: datadir=self.datadir
        name += ".npy"
        return np.load(datadir/name)
    def get_t(self, datadir=None):
        if datadir is None: datadir=self.datadir
        times = np.load(self.datadir / "time.npy")
        return times

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
                u = np.load(self.datadir / (solnamebase + ".npy"))
            data = self.sol_to_data(u, single_vector=False)
            # filename = os.path.join(self.dirname, solnamebase + ".vtu")
            # self.mesh.write(filename, data=data)
            self.mesh.write(self.datadir / (solnamebase + ".vtu"), data=data)
            return
        for iter in range(niter):
            solname = solnamebase + f"_{iter:05d}"
            # filename = os.path.join(self.dirname, solname + ".npy")
            u = np.load(self.datadir / (solnamebase + ".npy"))
            data = self.sol_to_data(u, single_vector=False)
            # filename = os.path.join(self.dirname, solname + ".vtu")
            self.mesh.write(self.datadir/(solname + ".vtu"), data=data)

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
        from matplotlib.figure import figaspect
        if fig is None:
            if gs is not None:
                raise ValueError(f"got gs but no fig")
            fig = plt.figure(constrained_layout=True, figsize=figaspect(2))
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
