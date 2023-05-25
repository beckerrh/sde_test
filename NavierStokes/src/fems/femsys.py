# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse

#=================================================================#
class Femsys():
    def __repr__(self):
        repr = f"{self.__class__.__name__}:{self.fem.__class__.__name__}"
        return repr
    def __init__(self, fem, ncomp, stack_storage, mesh=None):
        self.ncomp = ncomp
        self.stack_storage = stack_storage
        self.fem = fem
        if mesh: self.setMesh(mesh)
    def _get(self, v, icomp):
        if self.stack_storage: return v[icomp*self.mesh.nfaces: (icomp+1)*self.mesh.nfaces]
        # if self.stack_storage: return v.reshape(self.ncomp,-1)[icomp]
        return v[icomp::self.ncomp]
    def nlocal(self): return self.fem.nlocal()
    def nunknowns(self): return self.fem.nunknowns()
    def dofspercell(self): return self.fem.dofspercell()
    def setMesh(self, mesh):
        self.mesh = mesh
        self.fem.setMesh(mesh)
        # ncomp, nloc, ncells = self.ncomp, self.fem.nloc, self.mesh.ncells
        # dofs = self.fem.dofspercell()
        # nlocncomp = ncomp * nloc
        # self.rowssys = np.repeat(ncomp * dofs, ncomp).reshape(ncells * nloc, ncomp) + np.arange(ncomp, dtype=np.uint32)
        # self.rowssys = self.rowssys.reshape(ncells, nlocncomp).repeat(nlocncomp).reshape(ncells, nlocncomp, nlocncomp)
        # self.colssys = self.rowssys.swapaxes(1, 2)
        # self.colssys = self.colssys.reshape(-1)
        # self.rowssys = self.rowssys.reshape(-1)
    # def prepareBoundary(self, colorsdirichlet, colorsflux=[]):
    #     return self.fem._prepareBoundary(colorsdirichlet, colorsflux)
    def computeRhsCells(self, b, rhs):
        rhsall = self.fem.interpolate(rhs)
        for icomp in range(self.ncomp):
            # print(f"{i=} {rhsall[i]=}")
            # self.fem.massDot(b[icomp::self.ncomp], rhsall[i])
            self.fem.massDot(self._get(b, icomp), rhsall[icomp])
        return b
    # def interpolate(self, rhs): return self.fem.interpolate(rhs).ravel()
    def massDot(self, b, f, coeff=1):
        if b.shape != f.shape:
            raise ValueError(f"{b.shape=} {f.shape=}")
        for icomp in range(self.ncomp):
            # self.fem.massDot(b[icomp::self.ncomp], f[icomp::self.ncomp], coeff=coeff)
            self.fem.massDot(self._get(b, icomp), self._get(f, icomp), coeff=coeff)
        return b
    # def vectorBoundaryStrongZero(self, du, bdrydata):
    #     for i in range(self.ncomp): self.fem.vectorBoundaryStrongZero(du[i::self.ncomp], bdrydata)
    #     return du
    def computeErrorL2(self, solex, uh):
        eall, ecall = [], []
        for icomp in range(self.ncomp):
            # e, ec = self.fem.computeErrorL2(solex[icomp], uh[icomp::self.ncomp])
            e, ec = self.fem.computeErrorL2(solex[icomp], self._get(uh, icomp))
            eall.append(e)
            ecall.append(ec)
        return eall, ecall
    def computeBdryMean(self, u, colors):
        all = []
        for icomp in range(self.ncomp):
            # a = self.fem.computeBdryMean(u[icomp::self.ncomp], colors)
            a = self.fem.computeBdryMean(self._get(u, icomp), colors)
            all.append(a)
        return all

# ------------------------------
    def test(self):
        import scipy.sparse.linalg as splinalg
        colors = self.mesh.bdrylabels.keys()
        bdrydata = self.prepareBoundary(colorsdirichlet=colors)
        A = self.computeMatrixElasticity(mucell=1, lamcell=10)
        A, bdrydata = self.matrixBoundaryStrong(A, bdrydata=bdrydata)
        b = np.zeros(self.nunknowns() * self.ncomp)
        rhs = lambda x, y, z: np.ones(shape=(self.ncomp, x.shape))
        # self.computeRhsCells(b, rhs)
        fp1 = self.interpolate(rhs)
        self.massDot(b, fp1, coeff=1)
        b = self.vectorBoundaryStrongZero(b, bdrydata)
        return splinalg.spsolve(A, b)


# ------------------------------------- #

if __name__ == '__main__':
    raise ValueError(f"pas de test")
