import numpy as np
from src.models.stokes import Stokes
from src import fems, meshes, solvers
from src.solvers import linalg

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
        v = self._split(u)[0]
        dv = self._split(d)[0]
        self.computeFormConvection(dv, v)
        self.timer.add('form')
        return d
    def computeMatrix(self, u=None, coeffmass=None):
        # X = self.Astokes.copy()
        X = super().computeMatrix(u=u, coeffmass=coeffmass)
        v = self._split(u)[0]
        theta = 1
        if hasattr(self,'uold'): theta = 0.5
        X.A += theta*self.computeMatrixConvection(v)
        self.timer.add('matrix')
        return X
    # def rhs_dynamic(self, rhs, u, Aimp, time, dt, theta):
    #     self.Mass.dot(rhs, 1 / (theta * theta * dt), u)
    #     rhs += (theta - 1) / theta * Aimp.dot(u)
    #     rhs2 = self.computeRhs()
    #     rhs += (1 / theta) * rhs2
    def defect_dynamic(self, f, u):
        y = super().computeForm(u)-f
        self.Mass.dot(y, 1 / (self.theta * self.dt), u)
        v = self._split(u)[0]
        vold = self._split(self.uold)[0]
        dv = self._split(y)[0]
        self.computeFormConvection(dv, 0.5*(v+vold))
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
    def computeFormConvection(self, dv, v):
        dim = self.mesh.dimension
        self._compute_conv_data(v)
        colorsdirichlet = self.problemdata.bdrycond.colorsOfType("Dirichlet")
        # vdir = self.femv.interpolateBoundary(colorsdirichlet, self.problemdata.bdrycond.fct).ravel()
        # self.femv.massDotBoundary(dv, vdir, colors=colorsdirichlet, ncomp=self.ncomp, coeff=np.minimum(self.convdata.betart, 0))
        for icomp in range(dim):
            fdict = {col: self.problemdata.bdrycond.fct[col][icomp] for col in colorsdirichlet if col in self.problemdata.bdrycond.fct.keys()}
            vdir = self.femv.fem.interpolateBoundary(colorsdirichlet, fdict)
            self.femv.fem.massDotBoundary(self._getv(dv, icomp), vdir, colors=colorsdirichlet, coeff=np.minimum(self.convdata.betart, 0))
            self.femv.fem.computeFormTransportCellWise(self._getv(dv, icomp), self._getv(v, icomp), self.convdata, type='centered')
            self.femv.fem.computeFormJump(self._getv(dv, icomp), self._getv(v, icomp), self.convdata.betart)
    def computeMatrixConvection(self, v):
        if not hasattr(self.convdata,'beta'): self._compute_conv_data(v)
        A = self.femv.fem.computeMatrixTransportCellWise(self.convdata, type='centered')
        A += self.femv.fem.computeMatrixJump(self.convdata.betart)
        if self.singleA:
            return A
        return linalg.matrix2systemdiagonal(A, self.ncomp).tocsr()
    def computeBdryNormalFluxNitsche(self, v, p, colors):
        flux = super().computeBdryNormalFluxNitsche(v,p,colors)
        if self.convdata.betart is None : return flux
        ncomp, bdryfct = self.ncomp, self.problemdata.bdrycond.fct
        # vdir = self.femv.interpolateBoundary(colors, bdryfct).ravel()
        for icomp in range(ncomp):
            fdict = {col: bdryfct[col][icomp] for col in colors if col in bdryfct.keys()}
            vdir = self.femv.fem.interpolateBoundary(colors, fdict)
            for i,color in enumerate(colors):
                # flux[icomp,i] -= self.femv.fem.massDotBoundary(b=None, f=v[icomp::ncomp]-vdir[icomp::ncomp], colors=[color], coeff=np.minimum(self.convdata.betart, 0))
                flux[icomp, i] -= self.femv.fem.massDotBoundary(b=None, f=self._getv(v, icomp) - vdir[icomp], colors=[color],
                                                        coeff=np.minimum(self.convdata.betart, 0))
        return flux
