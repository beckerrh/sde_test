import numpy as np
from src.models.stokes import Stokes
from src import fems, meshes, solvers
from src.linalg import linalg

class NavierStokes(Stokes):
    def __format__(self, spec):
        if spec=='-':
            repr = super().__format__(spec)
            repr += f"\tconvmethod={self.convmethod}"
            return repr
        return self.__repr__()
    def __init__(self, **kwargs):
        self.linearsolver_def = {'method': 'scipy_lgmres', 'maxiter': 10, 'prec': 'Chorin', 'disp':0, 'rtol':1e-3}
        self.mode='nonlinear'
        self.convdata = fems.data.ConvectionData()
        # self.convmethod = kwargs.pop('convmethod', 'lps')
        self.convmethod = kwargs.get('convmethod', 'lps')
        self.lpsparam = kwargs.pop('lpsparam', 0.)
        self.newtontol = kwargs.pop('newtontol', 0)
        if not 'linearsolver' in kwargs: kwargs['linearsolver'] = self.linearsolver_def
        super().__init__(**kwargs)
        self.newmatrix = 0
    def new_params(self):
        super().new_params()
        self.Astokes = super().computeMatrix()
    def solve(self):
        sdata = solvers.newtondata.StoppingParamaters(maxiter=200, steptype='bt', nbase=1, rtol=self.newtontol)
        return self.static(mode='newton',sdata=sdata)
    def computeForm(self, u):
        d = self.Astokes.matvec(u)
        v = self.vectorview.get_part(0,u)
        dv = self.vectorview.get_part(0,d)
        # v = self._split(u)[0]
        # dv = self._split(d)[0]
        self.computeFormConvection(dv, v)
        self.timer.add('form')
        return d
    def rhs_dynamic(self, rhs, u, Aconst, time, dt, theta, semi_implicit):
        super().rhs_dynamic(rhs, u, Aconst, time, dt, theta, semi_implicit)
        if semi_implicit:
            self.computeFormConvection(rhs, 0.5*u)


    def computeMatrix(self, u=None, coeffmass=None):
        # X = self.Astokes.copy()
        X = super().computeMatrix(u=u, coeffmass=coeffmass)
        # v = self._split(u)[0]
        v = self.vectorview.get_part(0,u)
        theta = 1
        if hasattr(self,'uold'): theta = 0.5
        X.A += theta*self.computeMatrixConvection(v)
        self.timer.add('matrix')
        return X
    def defect_dynamic(self, f, u):
        y = super().computeForm(u)-f
        self.Mass.dot(y, 1 / (self.theta * self.dt), u)
        self.computeFormConvection(y, 0.5*(u+self.uold))
        self.timer.add('defect_dynamic')
        return y
    def computeMatrixConstant(self, coeffmass, coeffmassold=0):
        self.Astokes.A  =  self.Mass.addToStokes(coeffmass-coeffmassold, self.Astokes.A)
        return self.Astokes
        return super().computeMatrix(u, coeffmass)
    def _compute_conv_data(self, v):
        rt = fems.rt0.RT0(mesh=self.mesh)
        self.convdata.betart = rt.interpolateCR1(v, self.stack_storage)
        self.convdata.beta = rt.toCell(self.convdata.betart)
    def computeFormConvection(self, du, u):
        dim = self.mesh.dimension
        self._compute_conv_data(self.vectorview.get_part(0,u))
        colorsdirichlet = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        for icomp in range(dim):
            dv, v = self.vectorview.get(0,icomp,du), self.vectorview.get(0,icomp,u)
            fdict = {col: self.problemdata.bdrycond.fct[col][icomp] for col in colorsdirichlet if col in self.problemdata.bdrycond.fct.keys()}
            vdir = self.femv.interpolateBoundary(colorsdirichlet, fdict)
            self.femv.massDotBoundary(dv, vdir, colors=colorsdirichlet, coeff=np.minimum(self.convdata.betart, 0))
            self.femv.computeFormTransportCellWise(dv, v, self.convdata, type='centered')
            self.femv.computeFormJump(dv, v, self.convdata.betart)
    def computeMatrixConvection(self, v):
        if not hasattr(self.convdata,'beta'): self._compute_conv_data(v)
        A = self.femv.computeMatrixTransportCellWise(self.convdata, type='centered')
        A += self.femv.computeMatrixJump(self.convdata.betart)
        if self.singleA:
            return A
        return linalg.matrix2systemdiagonal(A, self.ncomp).tocsr()
    def computeBdryNormalFluxNitsche(self, u, colors):
        flux = super().computeBdryNormalFluxNitsche(u,colors)
        if self.convdata.betart is None : return flux
        ncomp, bdryfct = self.ncomp, self.problemdata.bdrycond.fct
        for icomp in range(ncomp):
            fdict = {col: bdryfct[col][icomp] for col in colors if col in bdryfct.keys()}
            vdir = self.femv.interpolateBoundary(colors, fdict)
            v = self.vectorview.get(0, icomp, u)
            for i,color in enumerate(colors):
                flux[icomp, i] -= self.femv.massDotBoundary(b=None, f=v - vdir, colors=[color],
                                                        coeff=np.minimum(self.convdata.betart, 0))
        return flux
