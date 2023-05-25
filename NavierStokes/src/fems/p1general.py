# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""
import numpy as np
import numpy.linalg as linalg
import scipy.sparse as sparse
from src.meshes.simplexmesh import SimplexMesh
# from src.fems import fem


#=================================================================#
# class P1general(fem.Fem):
class P1general():
    def __init__(self, **kwargs):
        pass
        # super().__init__(mesh=mesh)
        # for p,v in zip(['masslumpedvol', 'masslumpedbdry'], [False, False]):
        #     self.params_bool[p] = kwargs.pop(p, v)
        # for p, v in zip(['dirichletmethod', 'convmethod'], ['strong', 'supg']):
        #     self.params_str[p] = kwargs.pop(p, v)
        # if self.params_str['dirichletmethod'] == 'nitsche':
        #     self.params_float['nitscheparam'] = kwargs.pop('nitscheparam', 4)
        # if len(kwargs.keys()):
        #     raise ValueError(f"*** unused arguments {kwargs=}")
    def setMesh(self, mesh, innersides=False):
        self.mesh = mesh
        self.nloc = self.nlocal()
        if innersides: self.mesh.constructInnerFaces()
    def computeStencilCell(self, dofspercell):
        self.cols = np.tile(dofspercell, self.nloc).ravel()
        self.rows = np.repeat(dofspercell, self.nloc).ravel()
    # def interpolateCell(self, f):
    #     if isinstance(f, dict):
    #         b = np.zeros(self.mesh.ncells)
    #         for label, fct in f.items():
    #             if fct is None: continue
    #             cells = self.mesh.cellsoflabel[label]
    #             xc, yc, zc = self.mesh.pointsc[cells].T
    #             b[cells] = fct(xc, yc, zc)
    #         return b
    #     else:
    #         xc, yc, zc = self.mesh.pointsc.T
    #         return f(xc, yc, zc)
    def computeMatrixDiffusion(self, coeff, coeffM=None):
        ndofs = self.nunknowns()
        cellgrads = self.cellgrads[:,:,:self.mesh.dimension]
        mat = np.einsum('n,nil,njl->nij', self.mesh.dV*coeff, cellgrads, cellgrads)
        if coeffM: mat += self._computeMassMatrix(coeff=coeffM)
        return sparse.coo_matrix((mat.ravel(), (self.rows, self.cols)), shape=(ndofs, ndofs)).tocsr()
    def computeFormDiffusion(self, du, u, coeff):
        doc = self.dofspercell()
        cellgrads = self.cellgrads[:,:,:self.mesh.dimension]
        r = np.einsum('n,nil,njl,nj->ni', self.mesh.dV*coeff, cellgrads, cellgrads, u[doc])
        np.add.at(du, doc, r)

# ====================================================================================

#------------------------------
def test(self):
    import scipy.sparse.linalg as splinalg
    colors = self.mesh.bdrylabels.keys()
    bdrydata = self.prepareBoundary(colorsdir=colors)
    A = self.computeMatrixDiffusion(coeff=1)
    A = self.matrixBoundaryStrong(A, bdrydata=bdrydata)
    b = np.zeros(self.nunknowns())
    rhs = np.vectorize(lambda x,y,z: 1)
    b = self.computeRhsCell(b, rhs)
    self.vectorBoundaryStrongZero(b, bdrydata)
    return self.tonode(splinalg.spsolve(A, b))

# ------------------------------------- #

if __name__ == '__main__':
    trimesh = SimplexMesh(geomname="backwardfacingstep", hmean=0.3)
