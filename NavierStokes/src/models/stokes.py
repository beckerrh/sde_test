import copy

import numpy as np
import scipy.sparse as sparse
from src import fems
from src.models.model import Model
from src.tools.analyticalfunction import analyticalSolution
from src.linalg import linalg, saddle_point, vectorview
from functools import partial

#=================================================================#
class Stokes(Model):
    """
    """
    def __init__(self, **kwargs):
        if not hasattr(self,'linearsolver_def'):
            self.linearsolver_def = {'method': 'scipy_lgmres', 'maxiter': 100, 'prec': 'Chorin', 'disp':0, 'rtol':1e-6}
        if not hasattr(self,'linearsolver'):
            self.linearsolver = kwargs.pop('linearsolver', self.linearsolver_def)
        self.singleA = kwargs.pop('singleA', True)
        super().__init__(**kwargs)
        # print(f"{self=}")
    def createFem(self):
        self.dirichletmethod = self.disc_params.get('dirichletmethod','nitsche')
        if self.dirichletmethod=='nitsche':
            self.nitscheparam = self.disc_params.get('nitscheparam', 10)
        self.femv = fems.cr1.CR1()
        self.femp = fems.d0.D0()
    def new_params(self):
        self.mucell = self.compute_cell_vector_from_params('mu', self.problemdata.params)
    def setMesh(self, mesh):
        super().setMesh(mesh)
        self._checkProblemData()
        if not self.ncomp==self.mesh.dimension: raise ValueError(f"{self.mesh.dimension=} {self.ncomp=}")
        self.femv.setMesh(self.mesh)
        self.femp.setMesh(self.mesh)
        self.pmean = not ('Neumann' in self.problemdata.bdrycond.type.values() or 'Pressure' in self.problemdata.bdrycond.type.values())
        if self.dirichletmethod=='strong':
            assert 'Navier' not in self.problemdata.bdrycond.type.values()
            assert 'Pressure' not in self.problemdata.bdrycond.type.values()
            colorsdirichlet = self.problemdata.bdrycond.colorsOfType("Dirichlet")
            colorsflux = self.problemdata.postproc.colorsOfType("bdry_nflux")
            self.bdrydata = self.femv.prepareBoundary(colorsdirichlet, colorsflux)
        if self.singleA:
            assert 'Navier' not in self.problemdata.bdrycond.type.values()
            assert 'Pressure' not in self.problemdata.bdrycond.type.values()
        if self.pmean:
            ncomps = [self.mesh.dimension, 1, 1]
            ns = [self.mesh.nfaces, self.mesh.ncells, 1]
        else:
            ncomps = [self.mesh.dimension, 1]
            ns = [self.mesh.nfaces,self.mesh.ncells]
        self.vectorview = vectorview.VectorView(ncomps=ncomps, ns=ns, stack_storage=self.stack_storage)
        self.new_params()
    def _checkProblemData(self):
        # TODO checkProblemData() incomplete
        for col, fct in self.problemdata.bdrycond.fct.items():
            type = self.problemdata.bdrycond.type[col]
            if type == "Dirichlet":
                if len(fct) != self.mesh.dimension: raise ValueError(f"*** {type=} {len(fct)=} {self.mesh.dimension=}")
    def defineAnalyticalSolution(self, exactsolution, random=True):
        dim = self.mesh.dimension
        # print(f"defineAnalyticalSolution: {dim=} {self.ncomp=}")
        if exactsolution=="Linear":
            exactsolution = ["Linear", "Constant"]
        elif exactsolution=="Quadratic":
            exactsolution = ["Quadratic", "Linear"]
        v = analyticalSolution(exactsolution[0], dim, dim, random)
        p = analyticalSolution(exactsolution[1], dim, 1, random)
        return v,p
    def dirichletfct(self):
        solexact = self.problemdata.solexact
        v,p = solexact
        def _solexactdirp(x, y, z, nx, ny, nz):
            return p(x, y, z)
        from functools import partial
        def _solexactdirv(x, y, z, icomp):
            return v[icomp](x, y, z)
        return [partial(_solexactdirv, icomp=icomp) for icomp in range(self.ncomp)]
    def defineInitialConditionAnalyticalSolution(self, solexact):
        v,p = solexact
        def _fcticv(x, y, z):
            rhsv = np.empty(shape=(self.ncomp, *x.shape))
            for i in range(self.ncomp):
                rhsv[i] = v[i](x, y, z, 0)
            return rhsv
        return _fcticv

    def defineRhsAnalyticalSolution(self, solexact):
        v,p = solexact
        mu = self.problemdata.params.scal_glob['mu']
        def _fctrhsv(x, y, z):
            rhsv = np.zeros(shape=(self.ncomp, *x.shape))
            if v[0].has_time:
                for i in range(self.ncomp):
                    rhsv[i] += v[i].t(x, y, z, self.time)
            for i in range(self.ncomp):
                for j in range(self.ncomp):
                    rhsv[i] -= mu * v[i].dd(j, j, x, y, z)
                rhsv[i] += p.d(i, x, y, z)
            # print(f"{rhsv=}")
            return rhsv
        def _fctrhsp(x, y, z):
            rhsp = np.zeros(x.shape)
            for i in range(self.ncomp):
                rhsp += v[i].d(i, x, y, z)
            return rhsp
        return _fctrhsv, _fctrhsp
    def defineNeumannAnalyticalSolution(self, problemdata, color):
        solexact = problemdata.solexact
        mu = self.problemdata.params.scal_glob['mu']
        def _fctneumannv(x, y, z, nx, ny, nz, icomp):
            v, p = solexact
            rhsv = np.zeros(shape=x.shape)
            normals = nx, ny, nz
            # for i in range(self.ncomp):
            for j in range(self.ncomp):
                rhsv += mu  * v[icomp].d(j, x, y, z) * normals[j]
            rhsv -= p(x, y, z) * normals[icomp]
            return rhsv
        return [partial(_fctneumannv, icomp=icomp) for icomp in range(self.ncomp)]
    def defineNavierAnalyticalSolution(self, problemdata, color):
        solexact = problemdata.solexact
        mu = self.problemdata.params.scal_glob['mu']
        lambdaR = self.problemdata.params.scal_glob['navier']
        def _fctnaviervn(x, y, z, nx, ny, nz):
            v, p = solexact
            rhs = np.zeros(shape=x.shape)
            normals = nx, ny, nz
            # print(f"{x.shape=} {nx.shape=} {normals[0].shape=}")
            for i in range(self.ncomp):
                rhs += v[i](x, y, z) * normals[i]
            return rhs
        def _fctnaviertangent(x, y, z, nx, ny, nz, icomp):
            v, p = solexact
            rhs = np.zeros(shape=x.shape)
            # h = np.zeros(shape=(self.ncomp, x.shape[0]))
            normals = nx, ny, nz
            rhs = lambdaR*v[icomp](x, y, z)
            for j in range(self.ncomp):
                rhs += mu*v[icomp].d(j, x, y, z) * normals[j]
            return rhs
        return {'vn':_fctnaviervn, 'g':[partial(_fctnaviertangent, icomp=icomp) for icomp in range(self.ncomp)]}
    def definePressureAnalyticalSolution(self, problemdata, color):
        solexact = problemdata.solexact
        mu = self.problemdata.params.scal_glob['mu']
        lambdaR = self.problemdata.params.scal_glob['navier']
        def _fctpressure(x, y, z, nx, ny, nz):
            v, p = solexact
            # rhs = np.zeros(shape=x.shape)
            normals = nx, ny, nz
            # print(f"{x.shape=} {nx.shape=} {normals[0].shape=}")
            rhs = 1.0*p(x,y,z)
            for i in range(self.ncomp):
                for j in range(self.ncomp):
                    rhs -= mu*v[j].d(i, x, y, z) * normals[i]* normals[j]
            return rhs
        def _fctpressurevtang(x, y, z, nx, ny, nz, icomp):
            v, p = solexact
            return v[icomp](x,y,z)
        return {'p':_fctpressure, 'v':[partial(_fctpressurevtang, icomp=icomp) for icomp in range(self.ncomp)]}
    def initialCondition(self, interpolate=True):
        #TODO: higher order interpolation
        if not 'initial_condition' in self.problemdata.params.fct_glob:
            raise ValueError(f"missing 'initial_condition' in {self.problemdata.params.fct_glob=}")
        if not self._setMeshCalled: self.setMesh(self.mesh)
        u0 = np.zeros(self.vectorview.n())
        assert interpolate
        for icomp in range(self.ncomp):
            ic = self.problemdata.params.fct_glob['initial_condition'][icomp]
            ui = self.femv.interpolate(ic)
            self.vectorview.set(0, icomp, u0, ui)
            # print(f"{icomp=} {ui=}")
        return u0
    def sol_to_data(self, u, single_vector=True):
        data = {'point':{}, 'cell':{}, 'global':{}}
        if single_vector:
            vnode = np.empty(self.mesh.nnodes*self.ncomp)
            for icomp in range(self.ncomp):
                vnode[icomp::self.ncomp] = self.femv.tonode(self.vectorview.get(0,icomp,u))
            data['point']['V'] = vnode
        else:
            for icomp in range(self.ncomp):
                data['point'][f'V_{icomp:1d}'] = self.femv.tonode(self.vectorview.get(0,icomp,u))
        data['cell']['P'] = self.vectorview.get_part(1,u)
        return data
    def postProcess(self, u):
        p = self.vectorview.get_part(1,u)
        data = {'scalar':{}}
        if self.problemdata.solexact:
            errall, ecall = [], []
            for icomp in range(self.ncomp):
                err, ec = self.femv.computeErrorL2(self.problemdata.solexact[0][icomp], self.vectorview.get(0,icomp,u))
                errall.append(err)
                ecall.append(ec)
            data['scalar']['error_V_L2'] = np.sum(errall)
            err, e = self.femp.computeErrorL2(self.problemdata.solexact[1], p)
            data['scalar']['error_P_L2'] = err
        if self.problemdata.postproc:
            types = ["bdry_pmean", "bdry_vmean", "bdry_nflux"]
            for name, type in self.problemdata.postproc.type.items():
                colors = self.problemdata.postproc.colors(name)
                if type == types[0]:
                    data['scalar'][name] = self.femp.computeBdryMean(p, colors)
                elif type == types[1]:
                    for icomp in range(self.ncomp):
                        data['scalar'][name + "_" + f"{icomp}"] = self.femv.computeBdryMean(self.vectorview.get(0,icomp,u), colors)
                elif type == types[2]:
                    if self.dirichletmethod=='strong':
                        pp = self.computeBdryNormalFluxStrong(u, colors)
                    else:
                        pp = self.computeBdryNormalFluxNitsche(u, colors)
                    assert pp.ndim == 2
                    for i, color in enumerate(colors):
                        data['scalar'][name + "_" + f"{color}"] = pp[:, i]
                else:
                    raise ValueError(f"unknown postprocess type '{type}' for key '{name}'\nknown types={types=}")
        return data
    def computelinearSolver(self, A):
        if self.linearsolver == 'spsolve':
            args = {'method':'spsolve'}
        else:
            args = copy.deepcopy(self.linearsolver)
        if args['method'] != 'spsolve':
            if self.scale_ls:
                A.scale_matrix()
                args['scale'] = self.scale_ls
            # args['counter'] = 'sys'
            args['matvec'] = A.matvec
            # args['n'] = A.nall
            args['n'] = self.vectorview.n()
            prec = args.pop('prec', 'full')
            solver_v = args.pop('solver_v', None)
            solver_p = args.pop('solver_p', None)
            if prec == 'BS':
                alpha = args.pop('alpha', 10)
                P = saddle_point.BraessSarazin(A, alpha=alpha)
            elif prec == 'Chorin':
                P = saddle_point.Chorin(A, solver_v=solver_v, solver_p=solver_p)
            else:
                P = saddle_point.SaddlePointPreconditioner(A, solver_v=solver_v, solver_p=solver_p, method=prec)
            args['preconditioner'] = P
        return linalg.getLinearSolver(**args)
    def computeRhs(self, b=None, u=None, coeffmass=None):
        b = np.zeros(self.vectorview.n())
        bp = self.vectorview.get_part(1, b)
        if 'rhs' in self.problemdata.params.fct_glob:
            rhsv, rhsp = self.problemdata.params.fct_glob['rhs']
            if rhsv:
                rhsall = self.femv.interpolate(rhsv)
                for icomp in range(self.ncomp):
                    self.femv.massDot(self.vectorview.get(0, icomp, b), rhsall[icomp])
            if rhsp: self.femp.computeRhsCells(bp, rhsp)
        colorsdir = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        colorsneu = self.problemdata.bdrycond.colorsOfType("Neumann")
        colorsnav = self.problemdata.bdrycond.colorsOfType("Navier")
        colorsp = self.problemdata.bdrycond.colorsOfType("Pressure")
        for color in colorsneu:
            if not color in self.problemdata.bdrycond.fct or not self.problemdata.bdrycond.fct[color]: continue
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = np.linalg.norm(normalsS,axis=1)
            xf, yf, zf = self.mesh.pointsf[faces].T
            nx, ny, nz = normalsS.T / dS
            for icomp in range(self.ncomp):
                bS = dS * self.problemdata.bdrycond.fct[color][icomp](xf, yf, zf, nx, ny, nz)
                self.vectorview.get(0, icomp, b)[faces] += bS

        if self.dirichletmethod == 'strong':
            self.vectorBoundaryStrong(b, self.problemdata.bdrycond.fct, self.bdrydata)
        else:
            bdryfct = self.problemdata.bdrycond.fct
            ncomp = self.ncomp
            faces = self.mesh.bdryFaces(colorsdir)
            cells = self.mesh.cellsOfFaces[faces, 0]
            normalsS = self.mesh.normals[faces][:, :ncomp]
            for icomp in range(ncomp):
                fdict = {col: bdryfct[col][icomp] for col in colorsdir if col in bdryfct.keys()}
                vdir = self.femv.interpolateBoundary(colorsdir, fdict)
                np.add.at(bp, cells, -np.einsum('n,n->n', vdir[faces], normalsS[:,icomp]))
                self.femv.computeRhsNitscheDiffusion(self.nitscheparam, self.vectorview.get(0, icomp, b), self.mucell, colorsdir,
                                                        udir = vdir, bdrycondfct = None)

            colors = set(bdryfct.keys()).intersection(colorsnav)
            if len(colors):
                if not isinstance(bdryfct[next(iter(colors))], dict):
                    msg = """
                    For Navier b.c. please give a dictionary {vn:fct_scal, g:fvt_vec} with fct_scal scalar and fvt_vec a list of dim functions
                    """
                    raise ValueError(msg + f"\ngiven: {bdryfct[next(iter(colors))]=}")
                vnfct, gfct = {}, {}
                for col in colors:
                    if 'vn' in bdryfct[col].keys():
                        if not callable(bdryfct[col]['vn']):
                            raise ValueError(f"'vn' must be a function. Given:{bdryfct[col]['vn']=}")
                        vnfct[col] = bdryfct[col]['vn']
                    if 'g' in bdryfct[col].keys():
                        if not isinstance(bdryfct[col]['g'], list) or len(bdryfct[col]['g']) != self.ncomp:
                            raise ValueError(
                                f"'g' must be a list of functions with {self.ncomp} elements. Given:{bdryfct[col]['g']=}")
                        gfct[col] = bdryfct[col]['g']
                if len(vnfct):
                    vn = self.femv.interpolateBoundary(colorsnav, vnfct, lumped=False)
                    self.computeRhsBdryNitscheNavierNormal(b, colorsnav, self.mucell, vn)
                if len(gfct):
                    gt = np.vstack([self.femv.interpolateBoundary(colors, {col: gfct[col][icomp] for col in colors if
                                                                            col in gfct.keys()}, lumped=False) for icomp
                                      in range(self.ncomp)]).T

                    self.computeRhsBdryNitscheNavierTangent(b, colorsnav, self.mucell, gt)
            # Pressure condition
            colors = set(bdryfct.keys()).intersection(colorsp)
            if len(colors):
                if not isinstance(bdryfct[next(iter(colors))], dict):
                    msg = """
                    For Pressure b.c. please give a dictionary {p:fct_scal, v:fvt_vec} with fct_scal scalar and fvt_vec a list of dim functions
                    """
                    raise ValueError(msg + f"\ngiven: {bdryfct[next(iter(colors))]=}")
                pfct, vfct = {}, {}
                for col in colors:
                    if 'p' in bdryfct[col].keys():
                        if not callable(bdryfct[col]['p']):
                            raise ValueError(f"'vn' must be a function. Given:{bdryfct[col]['p']=}")
                        pfct[col] = bdryfct[col]['p']
                    if 'v' in bdryfct[col].keys():
                        if not isinstance(bdryfct[col]['v'], list) or len(bdryfct[col]['v']) != self.ncomp:
                            raise ValueError(
                                f"'v' must be a list of functions with {self.ncomp} elements. Given:{bdryfct[col]['v']=}")
                        vfct[col] = bdryfct[col]['v']
                if len(pfct):
                    p = self.femv.fem.interpolateBoundary(colorsp, pfct, lumped=False)
                    self.computeRhsBdryNitschePressureNormal(b, colorsp, self.mucell, p)
                if len(vfct):
                    v = self.femv.interpolateBoundary(colorsp, vfct)
                    self.computeRhsBdryNitschePressureTangent(b, colorsp, self.mucell, v)

        if not self.pmean: return b
        if self.problemdata.solexact is not None:
            p = self.problemdata.solexact[1]
            bmean = self.femp.computeMean(p)
        else: bmean=0
        b[-1] = bmean
        return b
    def computeForm(self, u, coeffmass=None):
        d = np.zeros_like(u)
        dp, p = self.vectorview.get_part(1,d), self.vectorview.get_part(1,u)
        ncomp, dV, cellgrads, foc = self.ncomp, self.mesh.dV, self.femv.cellgrads, self.mesh.facesOfCells
        for icomp in range(ncomp):
            r = np.einsum('n,nil,njl,nj->ni', dV*self.mucell, cellgrads, cellgrads, self.vectorview.get(0, icomp, u)[foc])
            np.add.at(self.vectorview.get(0, icomp, d
), foc, r)
        for icomp in range(ncomp):
            r = np.einsum('n,ni->ni', -dV*p, cellgrads[:,:,icomp])
            np.add.at(self.vectorview.get(0, icomp, d), foc, r)
            dp += np.einsum('n,ni,ni->n', dV, cellgrads[:, :, icomp], self.vectorview.get(0, icomp, u)[foc])
        if coeffmass:
            for icomp in range(ncomp):
                self.femv.computeFormMass(self.vectorview.get(0, icomp, d), self.vectorview.get(0, icomp, u), coeffmass)
        colorsdir = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        colorsnav = self.problemdata.bdrycond.colorsOfType("Navier")
        if self.dirichletmethod == 'strong':
            facesdirall, ncomp = self.bdrydata.facesdirall, self.ncomp
            for icomp in range(ncomp):
                self.vectorview.get(0, icomp, d)[facesdirall] = self.vectorview.get(0, icomp, u)[facesdirall]
        else:
            self.computeFormBdryNitscheDirichlet(d, u, colorsdir, self.mucell)
            self.computeFormBdryNitscheNavier(d, u, colorsnav, self.mucell)
        if self.pmean:
            dlam, lam = self.vectorview.get_part(2, d), self.vectorview.get_part(2, u)
            self.computeFormMeanPressure(dp, dlam, p, lam)
        # if not np.allclose(d,d2):
        #     raise ValueError(f"{d=}\n{d2=}")
        return d
    def computeMassMatrix(self):
        return linalg.MassMatrixIncompressible(self, self.femv.computeMassMatrix())
    def rhs_dynamic(self, rhs, u, Aconst, time, dt, theta, semi_implicit):
        self.Mass.dot(rhs, 1 / (theta * theta * dt), u)
        rhs += (theta - 1) / theta * Aconst.dot(u)
        rhs += (1 / theta) * self.computeRhs()
    def defect_dynamic(self, f, u):
        y = self.computeForm(u)-f
        self.Mass.dot(y, 1 / (self.theta * self.dt), u)
        return y
   #     return u, niter
    def computeMatrix(self, u=None, coeffmass=None):
        if coeffmass is None and 'alpha' in self.problemdata.params.scal_glob.keys():
            coeffmass = self.problemdata.params.scal_glob['alpha']
        # print(f"computeMatrix {coeffmass=}")
        A = self.femv.computeMatrixDiffusion(self.mucell, coeffmass)
        if not self.singleA:
            A = linalg.matrix2systemdiagonal(A, self.ncomp, self.stack_storage)
        nfaces, ncells, ncomp, dV = self.mesh.nfaces, self.mesh.ncells, self.ncomp, self.mesh.dV
        nloc, cellgrads, foc = self.femv.nloc, self.femv.cellgrads, self.mesh.facesOfCells
        rowsB = np.repeat(np.arange(ncells), ncomp * nloc)
        colsB = self.vectorview.col_indices(0, foc, ncells, nloc, nfaces)
        # if self.stack_storage:
        #     colsB = np.repeat(foc, ncomp).reshape(ncells * nloc, ncomp) + nfaces*np.arange(ncomp)
        # else:
        #     colsB = ncomp*np.repeat(foc, ncomp).reshape(ncells * nloc, ncomp) + np.arange(ncomp)
        mat = np.einsum('nkl,n->nkl', cellgrads[:, :, :ncomp], dV)
        B = sparse.coo_matrix((mat.ravel(), (rowsB, colsB.ravel())),shape=(ncells, nfaces * ncomp)).tocsr()
        colorsdir = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        colorsnav = self.problemdata.bdrycond.colorsOfType("Navier")
        colorsp = self.problemdata.bdrycond.colorsOfType("Pressure")
        if len(colorsp):
            raise NotImplementedError(f"Pressure boundary condition wrong (in newton)")
        if self.dirichletmethod == 'strong':
            A, B = self.matrixBoundaryStrong(A, B, self.bdrydata, self.singleA)
        else:
            A, B = self.computeMatrixBdryNitscheDirichlet(A, B, colorsdir, self.mucell, self.singleA)
            # print(f"{id(A)=} {id(B)=}")
            if len(colorsnav):
                lam = self.problemdata.params.scal_glob.get('navier', 0)
                A, B = self.computeMatrixBdryNitscheNavier(A, B, colorsnav, self.mucell, lam)
            if len(colorsp):
                A, B = self.computeMatrixBdryNitschePressure(A, B, colorsp, self.mucell)
            # print(f"{id(A)=} {id(B)=}")
        if not self.pmean:
            return saddle_point.SaddlePointSystem(self.vectorview, [A, B], singleA=self.singleA)
        ncells = self.mesh.ncells
        rows = np.zeros(ncells, dtype=int)
        cols = np.arange(0, ncells)
        C = sparse.coo_matrix((self.mesh.dV, (rows, cols)), shape=(1, ncells)).tocsr()
        return saddle_point.SaddlePointSystem(self.vectorview, [A, B, C], singleA=self.singleA)
    def computeFormMeanPressure(self,dp, dlam, p, lam):
        dlam += self.mesh.dV.dot(p)
        dp -= lam*self.mesh.dV
    def computeBdryNormalFluxNitsche(self, u, colors):
        ncomp, bdryfct = self.ncomp, self.problemdata.bdrycond.fct
        flux = np.zeros(shape=(ncomp,len(colors)))
        p = self.vectorview.get_part(1,u)
        for col in colors:
            if not col in bdryfct.keys(): continue
            if not isinstance(bdryfct[col], list):
                raise ValueError(f"don't know how to handle {type(next(iter(bdryfct.values())))=}")
        for icomp in range(ncomp):
            bdf = {col: bdryfct[col][icomp] for col in colors if col in bdryfct.keys()}
            vdir = self.femv.interpolateBoundary(colors, bdf, lumped=False)
            flux[icomp] = self.femv.computeBdryNormalFluxNitsche(self.nitscheparam, self.vectorview.get(0,icomp,u), colors, vdir, self.mucell)
            for i,color in enumerate(colors):
                faces = self.mesh.bdrylabels[color]
                cells = self.mesh.cellsOfFaces[faces,0]
                normalsS = self.mesh.normals[faces][:,:ncomp]
                flux[icomp,i] -= p[cells].dot(normalsS[:,icomp])
        return flux
    def computeFormBdryNitscheDirichlet(self, d, u, colors, mu):
        ncomp, dim  = self.ncomp, self.mesh.dimension
        faces = self.mesh.bdryFaces(colors)
        cells = self.mesh.cellsOfFaces[faces, 0]
        normalsS = self.mesh.normals[faces][:, :self.ncomp]
        dp, p = self.vectorview.get_part(1,d), self.vectorview.get_part(1,u)
        for icomp in range(ncomp):
            dv, v = self.vectorview.get(0, icomp, d), self.vectorview.get(0, icomp,u)
            self.femv.computeFormNitscheDiffusion(self.nitscheparam, dv, v, mu, colors)
            # pression
            r = np.einsum('f,f->f', p[cells], normalsS[:,icomp])
            np.add.at(self.vectorview.get(0, icomp, d), faces, r)
            r = np.einsum('f,f->f', normalsS[:,icomp], self.vectorview.get(0, icomp,u)[faces])
            np.add.at(dp, cells, -r)

    def computeFormBdryNitscheNavier(self, d, u, colors, mu):
        if not len(colors): return
        ncomp, dim  = self.ncomp, self.mesh.dimension
        faces = self.mesh.bdryFaces(colors)
        cells = self.mesh.cellsOfFaces[faces, 0]
        normalsS = self.mesh.normals[faces][:, :self.ncomp]
        for icomp in range(ncomp):
            dv, v = self.vectorview.get(0, icomp, d), self.vectorview.get(0, icomp,u)
            self.femv.computeFormNitscheDiffusion(self.nitscheparam, dv, v, mu, colors)
            r = np.einsum('f,f->f', self.vectorview.get_part(1,u)[cells], normalsS[:,icomp])
            np.add.at(self.vectorview.get(0, icomp, d), faces, r)
            r = np.einsum('f,f->f', normalsS[:,icomp], self.vectorview.get(0, icomp,u)[faces])
            np.add.at(self.vectorview.get_part(1,d), cells, -r)
        raise NotImplementedError()

    def computeMatrixBdryNitscheDirichlet(self, A, B, colors, mucell, singleA):
        nfaces, ncells, ncomp, dim  = self.mesh.nfaces, self.mesh.ncells, self.ncomp, self.mesh.dimension
        if singleA:
            A += self.femv.computeMatrixNitscheDiffusion(self.nitscheparam, mucell, colors)
        else:
            A0 = self.femv.computeMatrixNitscheDiffusion(self.nitscheparam, mucell, colors)
            A += linalg.matrix2systemdiagonal(A0, ncomp, self.stack_storage)
        #grad-div
        faces = self.mesh.bdryFaces(colors)
        cells = self.mesh.cellsOfFaces[faces, 0]
        normalsS = self.mesh.normals[faces][:, :self.ncomp]
        if self.stack_storage:
            indfaces = np.repeat(faces, ncomp)
            for icomp in range(ncomp):
                indfaces[icomp::ncomp] += icomp*nfaces
        else:
            indfaces = np.repeat(ncomp * faces, ncomp)
            for icomp in range(ncomp): indfaces[icomp::ncomp] += icomp
        cols = indfaces.ravel()
        rows = cells.repeat(ncomp).ravel()
        mat = normalsS.ravel()
        B -= sparse.coo_matrix((mat, (rows, cols)), shape=(ncells, ncomp*nfaces))
        return A,B

    def computeRhsNitscheDiffusionNormal(self, b, diffcoff, colors, udir, ncomp):
        faces = self.mesh.bdryFaces(colors)
        normalsS = self.mesh.normals[faces][:, :ncomp]
        dS = np.linalg.norm(normalsS, axis=1)
        normals = normalsS / dS[:, np.newaxis]
        if udir.shape[0] == self.mesh.nfaces * ncomp:
            for icomp in range(ncomp):
                for jcomp in range(ncomp):
                    self.femv.computeRhsNitscheDiffusion(self.nitscheparam, b[icomp::ncomp], diffcoff, colors=colors, udir=udir[jcomp::ncomp],
                                                        bdrycondfct=None, coeff=normals[:, icomp] * normals[:, jcomp])
        else:
            assert udir.shape[0] == self.mesh.nfaces
            for icomp in range(ncomp):
                self.femv.computeRhsNitscheDiffusion(self.nitscheparam, b[icomp::ncomp], diffcoff, colors=colors, udir=udir, bdrycondfct=None,
                                                    coeff=normals[:, icomp])

    def massDotBoundary(self, b, f, colors, ncomp, coeff=1):
        for icomp in range(ncomp):
            self.femv.massDotBoundary(b[icomp::ncomp], f[icomp::ncomp], colors=colors, coeff=coeff)

    def massDotBoundaryNormal(self, b, f, colors, ncomp, coeff=1):
        faces = self.mesh.bdryFaces(colors)
        normalsS = self.mesh.normals[faces][:, :ncomp]
        dS = np.linalg.norm(normalsS, axis=1)
        normals = normalsS / dS[:, np.newaxis]
        if f.shape[0] == self.mesh.nfaces * ncomp:
            for icomp in range(ncomp):
                for jcomp in range(ncomp):
                    self.femv.massDotBoundary(b[icomp::ncomp], f[jcomp::ncomp], colors=colors,
                                             coeff=normals[:, icomp] * normals[:, jcomp] * coeff)
        else:
            assert f.shape[0] == self.mesh.nfaces
            for icomp in range(ncomp):
                self.femv.massDotBoundary(b[icomp::ncomp], f, colors=colors, coeff=normals[:, icomp] * coeff)

    def computeRhsBdryNitscheNavierNormal(self, b, colors, mucell, vn):
        bv, bp = self.vectorview.get_part(0, b), self.vectorview.get_part(1, b)
        ncomp, dim = self.ncomp, self.mesh.dimension
        faces = self.mesh.bdryFaces(colors)
        cells = self.mesh.cellsOfFaces[faces, 0]
        normalsS = self.mesh.normals[faces][:, :ncomp]
        dS = np.linalg.norm(normalsS, axis=1)
        # normals = normalsS/dS[:,np.newaxis]
        # foc = self.mesh.facesOfCells[cells]
        np.add.at(bp, cells, -dS * vn[faces])
        self.computeRhsNitscheDiffusionNormal(bv, mucell, colors, vn, ncomp)

    def computeRhsBdryNitscheNavierTangent(self, b, colors, mucell, gt):
        bv, bp = self.vectorview.get_part(0, b), self.vectorview.get_part(1, b)
        ncomp, dim = self.ncomp, self.mesh.dimension
        self.massDotBoundary(bv, gt.ravel(), colors=colors, ncomp=ncomp, coeff=1)
        self.massDotBoundaryNormal(bv, -gt.ravel(), colors=colors, ncomp=ncomp, coeff=1)

    def computeRhsBdryNitschePressureNormal(self, b, colors, mucell, p):
        bv, bp = self.vectorview.get_part(0, b), self.vectorview.get_part(1, b)
        self.massDotBoundaryNormal(bv, -p, colors=colors, ncomp=self.ncomp, coeff=1)

    def computeRhsBdryNitschePressureTangent(self, b, colors, mucell, v):
        bv, bp = self.vectorview.get_part(0, b), self.vectorview.get_part(1, b)
        ncomp, dim = self.ncomp, self.mesh.dimension
        self.computeRhsNitscheDiffusion(bv, mucell, colors, v, ncomp)
        self.computeRhsNitscheDiffusionNormal(bv, mucell, colors, -v.ravel(), ncomp)

    def matrix2system(self, A, ncomp, i, j):
        A = A.tocoo()
        data, row, col, shape = A.data, A.row, A.col, A.shape
        n = shape[0]
        assert n == shape[1]
        row2 = ncomp * row + i
        col2 = ncomp * col + j
        return sparse.coo_matrix((data, (row2, col2)), shape=(ncomp * n, ncomp * n)).tocsr()

    def computeMatrixBdryNitscheNavier(self, A, B, colors, mucell, lambdaR):
        nfaces, ncells, ncomp, dim = self.mesh.nfaces, self.mesh.ncells, self.ncomp, self.mesh.dimension
        faces = self.mesh.bdryFaces(colors)
        cells = self.mesh.cellsOfFaces[faces, 0]
        normalsS = self.mesh.normals[faces][:, :dim]
        assert self.stack_storage == False
        # grad-div
        indfaces = np.repeat(ncomp * faces, ncomp)
        for icomp in range(ncomp): indfaces[icomp::ncomp] += icomp
        cols = indfaces.ravel()
        rows = cells.repeat(ncomp).ravel()
        B -= sparse.coo_matrix((normalsS.ravel(), (rows, cols)), shape=(ncells, ncomp * nfaces))
        # vitesses
        # A += self.femv.computeMatrixNitscheDiffusionNormal(mucell, colors, ncomp)
        nfaces, ncells, dim = self.mesh.nfaces, self.mesh.ncells, self.mesh.dimension
        assert dim == ncomp
        faces = self.mesh.bdryFaces(colors)
        # cells = self.mesh.cellsOfFaces[faces, 0]
        normalsS = self.mesh.normals[faces][:, :dim]
        dS = np.linalg.norm(normalsS, axis=1)
        for i in range(ncomp):
            for j in range(i, ncomp):
                Aij = self.femv.computeMatrixNitscheDiffusion(nitsche_param=self.nitscheparam, diffcoff=mucell, colors=colors,
                                                             coeff=normalsS[:, i] * normalsS[:, j] / dS ** 2)
                A += self.matrix2system(Aij, ncomp, i, j)
                if i != j: A += self.matrix2system(Aij, ncomp, j, i)

        A0 = self.femv.computeBdryMassMatrix(colors, lambdaR)
        A += linalg.matrix2systemdiagonal(A0, ncomp, self.stack_storage)
        for i in range(ncomp):
            for j in range(i, ncomp):
                Aij = self.femv.computeBdryMassMatrix(colors, lambdaR * normalsS[:, i] * normalsS[:, j] / dS ** 2)
                A -= self.matrix2system(Aij, ncomp, i, j)
                if i != j: A -= self.matrix2system(Aij, ncomp, j, i)
        # A += self.femv.computeMassMatrixBoundary(colors, ncomp,coeff=lambdaR) \
        #      - self.femv.computeMassMatrixBoundaryNormal(colors,ncomp,coeff=lambdaR)
        return A, B

    def computeMatrixBdryNitschePressure(self, A, B, colors, mucell):
        # vitesses
        A0 = self.fem.computeMatrixNitscheDiffusion(nitsche_param=self.nitscheparam, diffcoff=mucell, colors=colors)
        A += linalg.matrix2systemdiagonal(A0, self.ncomp)
        nfaces, ncells, dim = self.mesh.nfaces, self.mesh.ncells, self.mesh.dimension
        faces = self.mesh.bdryFaces(colors)
        # cells = self.mesh.cellsOfFaces[faces, 0]
        normalsS = self.mesh.normals[faces][:, :dim]
        dS = np.linalg.norm(normalsS, axis=1)
        for i in range(self.ncomp):
            for j in range(i, self.ncomp):
                Aij = self.fem.computeMatrixNitscheDiffusion(nitsche_param=self.nitscheparam, diffcoff=mucell, colors=colors,
                                                             coeff=normalsS[:, i] * normalsS[:, j] / dS ** 2)
                A += self.matrix2system(Aij, self.ncomp, i, j)
                if i != j: A += self.matrix2system(Aij, self.ncomp, j, i)
        # A += self.femv.computeMatrixNitscheDiffusion(mucell, colors, self.ncomp)
        # A -= self.femv.computeMatrixNitscheDiffusionNormal(mucell, colors, self.ncomp)
        return A, B

    def computeFormDivBdry(self, dp, v, colorsdir):
        ncomp, dim  = self.ncomp, self.mesh.dimension
        faces = self.mesh.bdryFaces(colorsdir)
        cells = self.mesh.cellsOfFaces[faces, 0]
        normalsS = self.mesh.normals[faces][:, :self.ncomp]
        for icomp in range(ncomp):
            r = np.einsum('f,f->f', normalsS[:,icomp], self.vectorview.get(0, icomp,u)[faces])
            np.add.at(dp, cells, -r)

# ------------------------------------------------------------------------
# -------------------------       Strong BC       -------------------------
# ------------------------------------------------------------------------
    def vectorBoundaryStrong(self, b, bdryfctv, bdrydata):
        facesdirall, facesinner, colorsdir, facesdirflux = bdrydata.facesdirall, bdrydata.facesinner, bdrydata.colorsdir, bdrydata.facesdirflux
        nfaces, ncells, ncomp  = self.mesh.nfaces, self.mesh.ncells, self.ncomp
        x, y, z = self.mesh.pointsf.T
        for color in colorsdir:
            if not color in bdryfctv: continue
            faces = self.mesh.bdrylabels[color]
            for icomp in range(ncomp):
                dirichlets = bdryfctv[color][icomp](x[faces], y[faces], z[faces])
                self.vectorview.get(0,icomp,b)[faces] = dirichlets
        if self.mode == "linear":
            self.computeFormDivBdry(self.vectorview.get_part(1,b), self.vectorview.get_part(0,b), colorsdir)
        return b
    def matrixBoundaryStrong(self, A, B, bdrydata, singleA):
        facesdirall, facesinner, colorsdir, facesdirflux = bdrydata.facesdirall, bdrydata.facesinner, bdrydata.colorsdir, bdrydata.facesdirflux
        nfaces, ncomp, nf = self.mesh.nfaces, self.ncomp, len(facesdirall)

        if singleA:
            help = np.ones(nfaces)
            help[facesdirall] = 0
            help = sparse.dia_matrix((help, 0), shape=(nfaces, nfaces))
            A = help.dot(A)
            help = np.zeros(nfaces)
            help[facesdirall] = 1.0
            help = sparse.dia_matrix((help, 0), shape=(nfaces, nfaces))
            A += help

        if self.stack_storage:
            inddir = np.tile(facesdirall, ncomp)
            for icomp in range(ncomp):
                inddir[icomp*nf:(icomp+1)*nf] += icomp*nfaces
        else:
            inddir = np.repeat(ncomp * facesdirall, ncomp)
            for icomp in range(ncomp): inddir[icomp::ncomp] += icomp
        help = np.ones((ncomp * nfaces))
        help[inddir] = 0
        help = sparse.dia_matrix((help, 0), shape=(ncomp * nfaces, ncomp * nfaces))
        B = B.dot(help)
        if singleA:
            return A,B
        A = help.dot(A)
        help = np.zeros((ncomp * nfaces))
        help[inddir] = 1.0
        help = sparse.dia_matrix((help, 0), shape=(ncomp * nfaces, ncomp * nfaces))
        A += help
        return A,B
    def computeBdryNormalFluxStrong(self, u, colors):
        v, p = self.vectorview.get_part(0,u), self.vectorview.get_part(1,u)
        return np.zeros(shape=(self.ncomp,len(colors)))
        nfaces, ncells, ncomp, bdrydata  = self.mesh.nfaces, self.mesh.ncells, self.ncomp, self.bdrydata
        # flux, omega = np.zeros(shape=(ncomp,len(colors))), np.zeros(len(colors))
        # for i,color in enumerate(colors):
        #     faces = self.mesh.bdrylabels[color]
        #     normalsS = self.mesh.normals[faces]
        #     dS = np.linalg.norm(normalsS, axis=1)
        #     omega[i] = np.sum(dS)
        #     As = bdrydata.Asaved[color]
        #     Bs = bdrydata.Bsaved[color]
        #     res = bdrydata.bsaved[color] - As * v + Bs.T * p
        #     for icomp in range(ncomp):
        #         flux[icomp, i] = np.sum(res[icomp::ncomp])
        #     # print(f"{flux=}")
        #     #TODO flux Stokes Dirichlet strong wrong
        # return flux
#------------------------------------------------------------------------
#-------------------------       Plotting       -------------------------
# ------------------------------------------------------------------------

    def _plot2d(self, data, fig, gs, title='', alpha=0.5):
        import matplotlib.gridspec as gridspec
        inner = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=1, subplot_spec=gs, wspace=0.3, hspace=0.3)
        v = data['point']['V']
        p = data['cell']['P']
        ax = fig.add_subplot(inner[0])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        self.plot_v(ax, v)
        ax = fig.add_subplot(inner[1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        self.plot_p(ax, p)
    def _plot3d(self, mesh, data, fig, gs, **kwargs):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        inner = gridspec.GridSpecFromSubplotSpec(nrows=2, ncols=2, subplot_spec=gs, wspace=0.3, hspace=0.3)
        import pyvista
        alpha = kwargs.pop('alpha', 0.6)
        p = data['cell']['P']
        v = data['point']['V']
        vr = v.reshape(self.mesh.nnodes, self.ncomp)
        vnorm = np.linalg.norm(vr, axis=1)
        mesh["V"] = vr
        mesh["vn"] = vnorm
        mesh.cell_data['P'] = p
        plotter = pyvista.Plotter(off_screen=kwargs.pop('off_screen',True))
        plotter.renderer.SetBackground(255, 255, 255)
        plotter.add_mesh(mesh, opacity=alpha, color='gray', show_edges=True)
        plotter.show(title=kwargs.pop('title', self.__class__.__name__))
        ax = fig.add_subplot(inner[0])
        ax.imshow(plotter.image)
        ax.set_xticks([])
        ax.set_yticks([])
        scalar_bar_args = {'title': 'p', 'color':'black'}
        plotter = pyvista.Plotter(off_screen=kwargs.pop('off_screen',True))
        plotter.renderer.SetBackground(255, 255, 255)
        plotter.add_mesh(mesh, opacity=alpha, color='gray', scalars='P', scalar_bar_args=scalar_bar_args)
        plotter.show(title=kwargs.pop('title', self.__class__.__name__))
        ax = fig.add_subplot(inner[1])
        ax.imshow(plotter.image)
        ax.set_xticks([])
        ax.set_yticks([])
        plotter = pyvista.Plotter(off_screen=kwargs.pop('off_screen',True))
        plotter.renderer.SetBackground(255, 255, 255)
        glyphs = mesh.glyph(orient="V", scale="vn", factor=10)
        # mesh.set_active_vectors("vectors")
        plotter.add_mesh(glyphs, show_scalar_bar=False, lighting=False, cmap='coolwarm')
        # plotter.show(title=kwargs.pop('title', self.__class__.__name__))
        # mesh.arrows.plot(off_screen=kwargs.pop('off_screen',True))
        plotter.show(title=kwargs.pop('title', self.__class__.__name__))
        ax = fig.add_subplot(inner[2])
        ax.imshow(plotter.image)
        ax.set_xticks([])
        ax.set_yticks([])

    def plot_p(self, ax, p=None, uname='p', title='', alpha=0.5):
        import matplotlib.pyplot as plt
        assert self.mesh.dimension==2
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        x, y, tris = self.mesh.points[:,0], self.mesh.points[:,1], self.mesh.simplices
        if title: ax.set_title(title)
        # ax.triplot(x, y, tris, color='gray', lw=1, alpha=alpha)
        assert len(p)==self.mesh.ncells
        cnt = ax.tripcolor(x, y, tris, facecolors=p, edgecolors='k', cmap='jet')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='3%', pad=0.4)
        clb = plt.colorbar(cnt, cax=cax, orientation='vertical')
        clb.ax.set_title(uname)
        return cnt
    def plot_v(self, ax, v=None, uname='v', title='', alpha=0.5):
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        x, y, tris = self.mesh.points[:,0], self.mesh.points[:,1], self.mesh.simplices
        if title: ax.set_title(title)
        ax.triplot(x, y, tris, color='gray', lw=1, alpha=alpha)
        vr = v.reshape(self.mesh.nnodes,self.ncomp)
        vnorm = np.linalg.norm(vr, axis=1)
        cnt = ax.tricontourf(x, y, tris, vnorm, levels=16, cmap='jet')
        divider = make_axes_locatable(ax)
        # cax = divider.append_axes('bottom', size='5%', pad=0.2)
        cax = divider.append_axes('right', size='3%', pad=0.4)
        clb = plt.colorbar(cnt, cax=cax, orientation='vertical')
        # clb.ax.tick_params(axis="both", labelsize=12)
        clb.ax.set_title(uname)
        qv = ax.quiver(x, y, vr[:,0], vr[:,1], units='xy')
        return cnt, qv

#=================================================================#
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from src.meshes import plotmesh, animdata
    from src.examples.incompflow import schaeferTurek2d
    mesh, data = schaeferTurek2d(h=0.4, mu=1)
    stokes = Stokes(mesh=mesh, problemdata=data, femparams={'dirichletmethod':'strong'}, linearsolver='spsolve')
    print(f"{stokes=}")
    # results = stokes.static(mode="linear")
    results = stokes.static(mode="newton")
    print(f"{results.data['scalar']=}")
    fig = plt.figure(1)
    gs = fig.add_gridspec(2, 1)
    plotmesh.meshWithBoundaries(stokes.mesh, gs=gs[0,0], fig=fig)
    plotmesh.meshWithDataNew(stokes.mesh, data=results.data, alpha=0.5, gs=gs[1,0], fig=fig)
    plt.show()
