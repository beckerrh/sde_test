# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
from src.fems import femsys, cr1
from src.tools import barycentric
import src.solvers.linalg

#=================================================================#
class CR1sys(femsys.Femsys):
    def __init__(self, ncomp, stack_storage, kwargs={}):
        super().__init__(cr1.CR1(kwargs=kwargs), ncomp, stack_storage)
    def setMesh(self, mesh):
        super().setMesh(mesh)
    def tonode(self, u):
        ncomp, nnodes = self.ncomp, self.mesh.nnodes
        unodes = np.zeros(ncomp*nnodes)
        if self.stack_storage:
            for icomp in range(ncomp):
                unodes.reshape(-1,self.ncomp).T[icomp] = self.fem.tonode(self._get(u, icomp))
        else:
            for icomp in range(ncomp):
                unodes[icomp::ncomp] = self.fem.tonode(self._get(u, icomp))
            # unodes[icomp::ncomp] = self.fem.tonode(u[icomp::ncomp])
        return unodes
    def interpolateBoundary(self, colors, f, lumped=False):
        # fs={col:f[col] for col in colors if col in f.keys()}
        if len(colors) == 0 or len(f) == 0: return
        # print(f"{f=}")
        for col in colors:
            if not col in f.keys(): continue
            if not isinstance(f[col], list):
                raise ValueError(f"don't know how to handle {type(next(iter(f.values())))=}")
        a = np.vstack([self.fem.interpolateBoundary(colors,
                                                    {col: f[col][icomp] for col in colors if col in f.keys()},
                                                    lumped=lumped) for icomp in range(self.ncomp)])
        #before a.T
        return a
    def matrixBoundaryStrong(self, A, bdrydata):
        # raise NotImplemented()
        facesdirflux, facesinner, facesdirall, colorsdir = bdrydata.facesdirflux, bdrydata.facesinner, bdrydata.facesdirall, bdrydata.colorsdir
        x, y, z = self.mesh.pointsf.T
        nfaces, ncomp = self.mesh.nfaces, self.ncomp
        for color, faces in facesdirflux.items():
            ind = np.repeat(ncomp * faces, ncomp)
            for icomp in range(ncomp): ind[icomp::ncomp] += icomp
            nb = faces.shape[0]
            help = sparse.dok_matrix((ncomp *nb, ncomp * nfaces))
            for icomp in range(ncomp):
                for i in range(nb): help[icomp + ncomp * i, icomp + ncomp * faces[i]] = 1
            bdrydata.Asaved[color] = help.dot(A)
        indin = np.repeat(ncomp * facesinner, ncomp)
        for icomp in range(ncomp): indin[icomp::ncomp] += icomp
        inddir = np.repeat(ncomp * facesdirall, ncomp)
        for icomp in range(ncomp): inddir[icomp::ncomp] += icomp
        bdrydata.A_inner_dir = A[indin, :][:, inddir]
        help = np.ones((ncomp * nfaces))
        help[inddir] = 0
        help = sparse.dia_matrix((help, 0), shape=(ncomp * nfaces, ncomp * nfaces))
        A = help.dot(A.dot(help))
        # A = help.dot(A)
        help = np.zeros((ncomp * nfaces))
        help[inddir] = 1.0
        help = sparse.dia_matrix((help, 0), shape=(ncomp * nfaces, ncomp * nfaces))
        A += help
        return A
    # def formBoundary(self, b, bdrydata, method):
    #     raise NotImplemented()
    #     facesdirall, ncomp = bdrydata.facesdirall, self.ncomp
    #     inddir = np.repeat(ncomp * facesdirall, ncomp)
    #     for icomp in range(ncomp): inddir[icomp::ncomp] += icomp
    #     b[inddir] = 0
    def vectorBoundaryStrongEqual(self, b, u, bdrydata):
        # raise NotImplemented()
        facesdirall, ncomp = bdrydata.facesdirall, self.ncomp
        inddir = np.repeat(ncomp * facesdirall, ncomp)
        for icomp in range(ncomp): inddir[icomp::ncomp] += icomp
        b[inddir] = u[inddir]
    def vectorBoundaryStrongZero(self, b, bdrydata):
        # raise NotImplemented()
        facesdirall, ncomp = bdrydata.facesdirall, self.ncomp
        inddir = np.repeat(ncomp * facesdirall, ncomp)
        for icomp in range(ncomp): inddir[icomp::ncomp] += icomp
        b[inddir] = 0
    def vectorBoundaryStrong(self, b, bdryfct, bdrydata):
        if not hasattr(bdrydata,'A_inner_dir'):
            raise Warning(f"#### vectorBoundaryStrong has to be called after matrix is constructed!!!!")
            return
        facesdirflux, facesinner, facesdirall, colorsdir = bdrydata.facesdirflux, bdrydata.facesinner, bdrydata.facesdirall, bdrydata.colorsdir
        x, y, z = self.mesh.pointsf.T
        nfaces, ncomp = self.mesh.nfaces, self.ncomp
        for color, faces in facesdirflux.items():
            ind = np.repeat(ncomp * faces, ncomp)
            for icomp in range(ncomp): ind[icomp::ncomp] += icomp
            bdrydata.bsaved[color] = b[ind]
        indin = np.repeat(ncomp * facesinner, ncomp)
        for icomp in range(ncomp): indin[icomp::ncomp] += icomp
        inddir = np.repeat(ncomp * facesdirall, ncomp)
        for icomp in range(ncomp): inddir[icomp::ncomp] += icomp
        help = np.zeros_like(b)
        for color in colorsdir:
            faces = self.mesh.bdrylabels[color]
            if color in bdryfct:
               for icomp in range(ncomp):
                    dirichlets = bdryfct[color][icomp](x[faces], y[faces], z[faces])
                    help[icomp + ncomp * faces] = dirichlets
        b[indin] -= bdrydata.A_inner_dir * help[inddir]
        b[inddir] = help[inddir]
        return b
    def computeRhsBoundary(self, b, colors, bdryfct):
        for color in colors:
            if not color in bdryfct or not bdryfct[color]: continue
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS,axis=1)
            xf, yf, zf = self.mesh.pointsf[faces].T
            nx, ny, nz = normalsS.T / dS
            for icomp in range(self.ncomp):
                bS = dS * bdryfct[color][icomp](xf, yf, zf, nx, ny, nz)
                if self.stack_storage:
                    indices = self.mesh.nfaces*icomp + faces
                else:
                    indices = icomp + self.ncomp * faces
                b[indices] += bS
        return b
    def computeMatrixDivergence(self):
        nfaces, ncells, ncomp, dV = self.mesh.nfaces, self.mesh.ncells, self.ncomp, self.mesh.dV
        nloc, cellgrads, foc = self.fem.nloc, self.fem.cellgrads, self.mesh.facesOfCells
        rowsB = np.repeat(np.arange(ncells), ncomp * nloc)
        if self.stack_storage:
            colsB = np.repeat(foc, ncomp).reshape(ncells * nloc, ncomp) + nfaces*np.arange(ncomp)
        else:
            colsB = ncomp*np.repeat(foc, ncomp).reshape(ncells * nloc, ncomp) + np.arange(ncomp)
        mat = np.einsum('nkl,n->nkl', cellgrads[:, :, :ncomp], dV)
        B = sparse.coo_matrix((mat.ravel(), (rowsB, colsB.ravel())),shape=(ncells, nfaces * ncomp)).tocsr()
        return B
    def computeFormDivGrad(self, dv, dp, v, p):
        ncomp, dV, cellgrads, foc = self.ncomp, self.mesh.dV, self.fem.cellgrads, self.mesh.facesOfCells
        for icomp in range(ncomp):
            r = np.einsum('n,ni->ni', -dV*p, cellgrads[:,:,icomp])
            np.add.at(self._get(dv, icomp), foc, r)
            dp += np.einsum('n,ni,ni->n', dV, cellgrads[:, :, icomp], self._get(v, icomp)[foc])
    def computeMatrixLaplace(self, mucell, coeffM=None):
        # assert None
        return src.solvers.linalg.matrix2systemdiagonal(self.fem.computeMatrixDiffusion(mucell, coeffM), self.ncomp, self.stack_storage).tocsr()
    def computeFormLaplace(self, mu, dv, v):
        ncomp, dV, cellgrads, foc = self.ncomp, self.mesh.dV, self.fem.cellgrads, self.mesh.facesOfCells
        for icomp in range(ncomp):
            # r = np.einsum('n,nil,njl,nj->ni', dV*mu, cellgrads, cellgrads, v[icomp::ncomp][foc])
            # np.add.at(dv[icomp::ncomp], foc, r)
            r = np.einsum('n,nil,njl,nj->ni', dV*mu, cellgrads, cellgrads, self._get(v,icomp)[foc])
            np.add.at(self._get(dv,icomp), foc, r)
    def computeFormMass(self, du, u, coeff):
        ncomp = self.ncomp
        for icomp in range(ncomp):
            # self.fem.computeFormMass(du[icomp::ncomp], u[icomp::ncomp], coeff)
            self.fem.computeFormMass(self._get(du,icomp), self._get(u,icomp), coeff)
    def computeRhsNitscheDiffusion(self, b, diffcoff, colors, udir, ncomp):
        for icomp in range(ncomp):
            # self.fem.computeRhsNitscheDiffusion(b[icomp::ncomp], diffcoff, colors, udir=udir[:,icomp], bdrycondfct=None)
            self.fem.computeRhsNitscheDiffusion(self._get(b,icomp), diffcoff, colors, udir=udir[:,icomp], bdrycondfct=None)
            # self.fem.computeRhsNitscheDiffusion(self._get(b, icomp), diffcoff, colors, self._get(udir, icomp), bdrycondfct=None)
    # def computeRhsNitscheDiffusionNormal(self, b, diffcoff, colors, udir, ncomp):
    #     faces = self.mesh.bdryFaces(colors)
    #     normalsS = self.mesh.normals[faces][:,:ncomp]
    #     dS = np.linalg.norm(normalsS, axis=1)
    #     normals = normalsS/dS[:,np.newaxis]
    #     if udir.shape[0] == self.mesh.nfaces*ncomp:
    #         for icomp in range(ncomp):
    #             for jcomp in range(ncomp):
    #                 # self.fem.computeRhsNitscheDiffusion(b[icomp::ncomp], diffcoff, colors, udir=udir[jcomp::ncomp], bdrycondfct=None, coeff=normals[:,icomp]*normals[:,jcomp])
    #                 self.fem.computeRhsNitscheDiffusion(self._get(b,icomp), diffcoff, colors, udir=self._get(udir,jcomp), bdrycondfct=None, coeff=normals[:,icomp]*normals[:,jcomp])
    #     else:
    #         assert udir.shape[0] == self.mesh.nfaces
    #         for icomp in range(ncomp):
    #             # self.fem.computeRhsNitscheDiffusion(b[icomp::ncomp], diffcoff, colors, udir=udir, bdrycondfct=None, coeff=normals[:,icomp])
    #             self.fem.computeRhsNitscheDiffusion(self._get(b,icomp), diffcoff, colors, udir=udir, bdrycondfct=None,
    #                                     coeff=normals[:, icomp])
    # def massDotBoundary(self, b, f, colors, ncomp, coeff=1):
    #     for icomp in range(ncomp):
    #         # self.fem.massDotBoundary(b[icomp::ncomp], f[icomp::ncomp], colors=colors, coeff=coeff)
    #         self.fem.massDotBoundary(self._get(b,icomp), self._get(f,icomp), colors=colors, coeff=coeff)
    # def massDotBoundaryNormal(self, b, f, colors, ncomp, coeff=1):
    #     faces = self.mesh.bdryFaces(colors)
    #     normalsS = self.mesh.normals[faces][:,:ncomp]
    #     dS = np.linalg.norm(normalsS, axis=1)
    #     normals = normalsS/dS[:,np.newaxis]
    #     if f.shape[0] == self.mesh.nfaces*ncomp:
    #         for icomp in range(ncomp):
    #             for jcomp in range(ncomp):
    #                 # self.fem.massDotBoundary(b[icomp::ncomp], f[jcomp::ncomp], colors=colors, coeff=normals[:,icomp]*normals[:,jcomp]*coeff)
    #                 self.fem.massDotBoundary(self._get(b,icomp), self._get(f,jcomp), colors=colors, coeff=normals[:,icomp]*normals[:,jcomp]*coeff)
    #     else:
    #         assert f.shape[0] == self.mesh.nfaces
    #         for icomp in range(ncomp):
    #             # self.fem.massDotBoundary(b[icomp::ncomp], f, colors=colors, coeff=normals[:,icomp]*coeff)
    #             self.fem.massDotBoundary(self._get(b,icomp), f, colors=colors, coeff=normals[:, icomp] * coeff)
    # def computeMassMatrixBoundary(self, colors, ncomp, coeff):
    #     assert None
    #     A = self.fem.computeBdryMassMatrix(colors, coeff)
    #     return src.solvers.linalg.matrix2systemdiagonal(A, ncomp, self.stack_storage)
    # def computeMassMatrixBoundaryNormal(self, colors, ncomp, coeff):
    #     nfaces, ncells, dim  = self.mesh.nfaces, self.mesh.ncells, self.mesh.dimension
    #     assert dim == ncomp
    #     faces = self.mesh.bdryFaces(colors)
    #     normalsS = self.mesh.normals[faces][:, :dim]
    #     dS = np.linalg.norm(normalsS, axis=1)
    #     A = sparse.coo_matrix((ncomp*nfaces, ncomp*nfaces))
    #     for i in range(ncomp):
    #         for j in range(i,ncomp):
    #             Aij = self.fem.computeBdryMassMatrix(colors, coeff * normalsS[:, i] * normalsS[:, j] / dS ** 2)
    #             A += self.matrix2system(Aij, ncomp, i, j)
    #             if i!=j: A += self.matrix2system(Aij, ncomp, j, i)
    #     return A
    def computeMatrixNitscheDiffusion(self, nitscheparam, diffcoff, colors, ncomp):
        A = self.fem.computeMatrixNitscheDiffusion(nitscheparam, diffcoff, colors)
        return src.solvers.linalg.matrix2systemdiagonal(A, ncomp, self.stack_storage)
    # def computeMatrixNitscheDiffusionNormal(self, diffcoff, colors, ncomp):
    #     nfaces, ncells, dim  = self.mesh.nfaces, self.mesh.ncells, self.mesh.dimension
    #     assert dim == ncomp
    #     faces = self.mesh.bdryFaces(colors)
    #     # cells = self.mesh.cellsOfFaces[faces, 0]
    #     normalsS = self.mesh.normals[faces][:, :dim]
    #     dS = np.linalg.norm(normalsS, axis=1)
    #     A = sparse.coo_matrix((ncomp*nfaces, ncomp*nfaces))
    #     for i in range(ncomp):
    #         for j in range(i,ncomp):
    #             Aij = self.fem.computeMatrixNitscheDiffusion(diffcoff, colors, coeff=normalsS[:,i]*normalsS[:,j]/dS**2)
    #             A += self.matrix2system(Aij, ncomp, i, j)
    #             if i!=j: A += self.matrix2system(Aij, ncomp, j, i)
    #     return A
    # def computeFormNitscheDiffusion(self, du, u, diffcoff, colorsdir, ncomp):
    #     for icomp in range(ncomp):
    #         # self.fem.computeFormNitscheDiffusion(du[icomp::ncomp], u[icomp::ncomp], diffcoff, colorsdir)
    #         self.fem.computeFormNitscheDiffusion(self._get(du,icomp), self._get(u,icomp), diffcoff, colorsdir)
    # def computeMatrixElasticity(self, mucell, lamcell):
    #     raise NotImplemented()
    #     nfaces, ncells, ncomp, dV = self.mesh.nfaces, self.mesh.ncells, self.ncomp, self.mesh.dV
    #     nloc, rows, cols, cellgrads = self.fem.nloc, self.rowssys, self.colssys, self.fem.cellgrads
    #     mat = np.zeros(shape=rows.shape, dtype=float).reshape(ncells, ncomp * nloc, ncomp * nloc)
    #     for i in range(ncomp):
    #         for j in range(self.ncomp):
    #             mat[:, i::ncomp, j::ncomp] += (np.einsum('nk,nl->nkl', cellgrads[:, :, i], cellgrads[:, :, j]).T * dV * lamcell).T
    #             mat[:, i::ncomp, j::ncomp] += (np.einsum('nk,nl->nkl', cellgrads[:, :, j], cellgrads[:, :, i]).T * dV * mucell).T
    #             mat[:, i::ncomp, i::ncomp] += (np.einsum('nk,nl->nkl', cellgrads[:, :, j], cellgrads[:, :, j]).T * dV * mucell).T
    #     A = sparse.coo_matrix((mat.ravel(), (rows, cols)), shape=(ncomp*nfaces, ncomp*nfaces)).tocsr()
    #     A += self.computeMatrixKorn(mucell)
    #     return A
    # def computeMatrixDivDiv(self, mucell, coeff=1):
    #     raise NotImplemented()
    #     nfaces, ncells, ncomp, dV = self.mesh.nfaces, self.mesh.ncells, self.ncomp, self.mesh.dV
    #     nloc, rows, cols, cellgrads = self.fem.nloc, self.rowssys, self.colssys, self.fem.cellgrads
    #     mat = np.zeros(shape=rows.shape, dtype=float).reshape(ncells, ncomp * nloc, ncomp * nloc)
    #     if not isinstance(coeff,(int,float)):
    #         assert coeff.shape == (self.mesh.ncells)
    #     for i in range(ncomp):
    #         for j in range(self.ncomp):
    #             mat[:, i::ncomp, j::ncomp] += np.einsum('n,nk,nl->nkl', coeff*dV, cellgrads[:, :, i], cellgrads[:, :, j])
    #     A = sparse.coo_matrix((mat.ravel(), (rows, cols)), shape=(ncomp*nfaces, ncomp*nfaces)).tocsr()
    #     return A
    # def computeMatrixKorn(self, mucell):
    #     raise NotImplemented()
    #     ncomp = self.ncomp
    #     dimension, dV, ndofs = self.mesh.dimension, self.mesh.dV, self.nunknowns()
    #     nloc, dofspercell, nall = self.nlocal(), self.dofspercell(), ncomp*ndofs
    #     ci0 = self.mesh.cellsOfInteriorFaces[:,0]
    #     ci1 = self.mesh.cellsOfInteriorFaces[:,1]
    #     assert np.all(ci1>=0)
    #     normalsS = self.mesh.normals[self.mesh.innerfaces]
    #     dS = linalg.norm(normalsS, axis=1)
    #     faces = self.mesh.faces[self.mesh.innerfaces]
    #     # fi0, fi1 = self.mesh.facesOfCellsNotOnInnerFaces(faces, ci0, ci1)
    #     fi0, fi1 = self.mesh.facesOfCellsNotOnInnerFaces(ci0, ci1)
    #     massloc = barycentric.crbdryothers(dimension)
    #     if isinstance(mucell,(int,float)):
    #         scale = mucell*dS/(dV[ci0]+ dV[ci1])
    #     else:
    #         scale = (mucell[ci0] + mucell[ci1]) * dS / (dV[ci0] + dV[ci1])
    #     scale *= 8
    #     A = sparse.coo_matrix((nall, nall))
    #     mat = np.einsum('n,kl->nkl', dS*scale, massloc).reshape(-1)
    #     for icomp in range(ncomp):
    #         d0 = ncomp*fi0+icomp
    #         d1 = ncomp*fi1+icomp
    #         rows0 = d0.repeat(nloc-1)
    #         cols0 = np.tile(d0,nloc-1).ravel()
    #         rows1 = d1.repeat(nloc-1)
    #         cols1 = np.tile(d1,nloc-1).ravel()
    #         A += sparse.coo_matrix((mat, (rows0, cols0)), shape=(nall, nall))
    #         A += sparse.coo_matrix((-mat, (rows0, cols1)), shape=(nall, nall))
    #         A += sparse.coo_matrix((-mat, (rows1, cols0)), shape=(nall, nall))
    #         A += sparse.coo_matrix((mat, (rows1, cols1)), shape=(nall, nall))
    #     return A
    # def computeMatrixHdivPenaly(self, coef=1):
    #     raise NotImplemented()
    #     ncomp = self.ncomp
    #     dim, dV, ndofs, nloc = self.mesh.dimension, self.mesh.dV, self.nunknowns(), self.nlocal()
    #     nall = ncomp*ndofs
    #     ci0 = self.mesh.cellsOfInteriorFaces[:,0]
    #     ci1 = self.mesh.cellsOfInteriorFaces[:,1]
    #     assert np.all(ci1>=0)
    #     normalsS = self.mesh.normals[self.mesh.innerfaces,:dim]
    #     dS = linalg.norm(normalsS, axis=1)
    #     fi0, fi1 = self.mesh.facesOfCellsNotOnInnerFaces(ci0, ci1)
    #     massloc = barycentric.crbdryothers(dim)
    #     if isinstance(coef,(int,float)):
    #         scale = coef/(dV[ci0]+ dV[ci1])
    #     else:
    #         assert coef.shape == (self.mesh.ncells)
    #         scale = (coef[ci0] + coef[ci1]) / (dV[ci0] + dV[ci1])
    #     A = sparse.coo_matrix((nall, nall))
    #     for icomp in range(ncomp):
    #         d0i = ncomp*fi0+icomp
    #         d1i = ncomp*fi1+icomp
    #         rows0i = d0i.repeat(nloc-1)
    #         rows1i = d1i.repeat(nloc-1)
    #         for jcomp in range(ncomp):
    #             d0j = ncomp*fi0+jcomp
    #             d1j = ncomp*fi1+jcomp
    #             cols0j = np.tile(d0j,nloc-1).ravel()
    #             cols1j = np.tile(d1j,nloc-1).ravel()
    #             mat = np.einsum('n,kl->nkl', normalsS[:,icomp]*normalsS[:,jcomp]*scale, massloc).ravel()
    #             # print(f"{mat=}")
    #             A += sparse.coo_matrix((mat, (rows0i, cols0j)), shape=(nall, nall))
    #             A += sparse.coo_matrix((-mat, (rows0i, cols1j)), shape=(nall, nall))
    #             A += sparse.coo_matrix((-mat, (rows1i, cols0j)), shape=(nall, nall))
    #             A += sparse.coo_matrix((mat, (rows1i, cols1j)), shape=(nall, nall))
    #     # print(f"{A.A=}")
    #     return A
    # def computeFormHdivPenaly(self, du, u, coef=1):
    #     raise NotImplemented()
    #     ncomp = self.ncomp
    #     dim, dV, ndofs = self.mesh.dimension, self.mesh.dV, self.nunknowns()
    #     ci0 = self.mesh.cellsOfInteriorFaces[:,0]
    #     ci1 = self.mesh.cellsOfInteriorFaces[:,1]
    #     assert np.all(ci1>=0)
    #     normalsS = self.mesh.normals[self.mesh.innerfaces,:dim]
    #     dS = linalg.norm(normalsS, axis=1)
    #     faces = self.mesh.faces[self.mesh.innerfaces]
    #     fi0, fi1 = self.mesh.facesOfCellsNotOnInnerFaces(ci0, ci1)
    #     massloc = barycentric.crbdryothers(dim)
    #     if isinstance(coef,(int,float)):
    #         scale = coef/(dV[ci0]+ dV[ci1])
    #     else:
    #         assert coef.shape == (self.mesh.ncells)
    #         scale = (coef[ci0] + coef[ci1]) / (dV[ci0] + dV[ci1])
    #     for icomp in range(ncomp):
    #         u = u.reshape((dim,ndofs), order='F')
    #         # print(f"{normalsS.shape} {u[:,fi0].shape=}")
    #         r = np.einsum('n,nj,kl,jnl->nk', normalsS[:,icomp]*scale, normalsS, massloc, u[:,fi0]-u[:,fi1])
    #         np.add.at(du, ncomp*fi0+icomp, r)
    #         np.add.at(du, ncomp*fi1+icomp, -r)
    #     # for icomp in range(ncomp):
    #     #     for jcomp in range(ncomp):
    #     #         uj = u[ncomp*fi0+jcomp]-u[ncomp*fi1+jcomp]
    #     #         r = np.einsum('n,kl,nl->nk', normalsS[:,icomp]*normalsS[:,jcomp]*scale, massloc, uj)
    #     #         np.add.at(du, ncomp*fi0+icomp, r)
    #     #         np.add.at(du, ncomp*fi1+icomp, -r)
    # def computeBdryNormalFlux(self, u, colors, bdrydata):
    #     flux, omega = np.zeros(shape=(len(colors),self.ncomp)), np.zeros(len(colors))
    #     for i,color in enumerate(colors):
    #         faces = self.mesh.bdrylabels[color]
    #         normalsS = self.mesh.normals[faces]
    #         dS = linalg.norm(normalsS, axis=1)
    #         omega[i] = np.sum(dS)
    #         bs, As = bdrydata.bsaved[color], bdrydata.Asaved[color]
    #         res = bs - As * u
    #         for icomp in range(self.ncomp):
    #             flux[i, icomp] = np.sum(res[icomp::self.ncomp])
    #             flux[i, icomp] = np.sum(self._get(res,icomp))
    #     return flux
    # def computeBdryMean(self, u, colors):
    #     mean = np.empty(shape=(self.ncomp, len(colors)))
    #     for icomp in range(self.ncomp):
    #         # mean[i] = self.fem.computeBdryMean(u[icomp::self.ncomp], colors)
    #         mean[i] = self.fem.computeBdryMean(self._get(u,icomp), colors)
    #     return mean


# ------------------------------------- #

if __name__ == '__main__':
    from src.meshes import testmeshes
    from src.meshes import plotmesh
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    mesh = testmeshes.backwardfacingstep(h=0.2)
    fem = CR1sys(ncomp=2, mesh=mesh)
    u = fem.test()
    point_data = fem.getPointData(u)
    fig = plt.figure(figsize=(10, 8))
    outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.2)
    plotmesh.meshWithBoundaries(mesh, fig=fig, outer=outer[0])
    plotmesh.meshWithData(mesh, point_data=point_data, title="P1 Test", alpha=1, fig=fig, outer=outer[1])
    plt.show()
