# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""
import numpy as np
import scipy.linalg as linalg
# from src.fems import fem

#=================================================================#
class D0():
    def __init__(self, mesh=None):
        pass
        # super().__init__(mesh=mesh)
    def setMesh(self, mesh):
        self.mesh = mesh
    def nlocal(self): return 1
    def nunknowns(self): return self.mesh.ncells
    def tonode(self, u):
        unodes = np.zeros(self.mesh.nnodes)
        if u.shape[0] != self.mesh.ncells: raise ValueError(f"{u.shape=} {self.mesh.ncells=}")
        np.add.at(unodes, self.mesh.simplices.T, u.T)
        countnodes = np.zeros(self.mesh.nnodes, dtype=int)
        np.add.at(countnodes, self.mesh.simplices.T, 1)
        unodes /= countnodes
        return unodes
    def interpolate(self, f):
        xc, yc, zc = self.mesh.pointsc.T
        return f(xc, yc, zc)
    def massDot(self, b, f, coeff=1):
        dV = self.mesh.dV
        b += coeff*dV*f
        return b
    def computeRhsCells(self, b, rhsfct):
        rhs = self.interpolate(rhsfct)
        self.massDot(b, rhs)
        return b
    def computeRhsBoundary(self, b, colors, bdryfct):
        for color in colors:
            if not color in bdryfct or not bdryfct[color]: continue
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            xf, yf, zf = self.mesh.pointsf[faces].T
            nx, ny, nz = normalsS.T / dS
            neumanns = bdryfct[color](xf, yf, zf, nx, ny, nz)
            cells = self.mesh.cellsOfFaces[faces,0]
            # b[cells] -= dS * neumanns
        return b
    def computeBdryMean(self, b, colors):
        if not len(colors): return 0
        res, omega = 0, 0
        for color in colors:
            faces = self.mesh.bdrylabels[color]
            cells = self.mesh.cellsOfFaces[faces,0]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            res += dS*b[cells]
            omega += np.sum(dS)
        return res/omega

    def computeErrorL2(self, solexact, uh):
        xc, yc, zc = self.mesh.pointsc.T
        en = solexact(xc, yc, zc) - uh
        Men = np.zeros_like(en)
        return np.sqrt( np.dot(en, self.massDot(Men,en)) ), en
    def computeMean(self, fct):
        xc, yc, zc = self.mesh.pointsc.T
        return np.sum(self.mesh.dV*fct(xc,yc,zc))

# ------------------------------------- #
if __name__ == '__main__':
    raise NotImplementedError("no test")
