# -*- coding: utf-8 -*-
"""
Created on Sun Dec  4 18:14:29 2016

@author: becker
"""

from matplotlib import colors
import numpy as np
import scipy.linalg as linalg
import scipy.sparse as sparse
# from src.tools import barycentric, npext, checkmmatrix
from src.tools import barycentric
# from src.meshes import move, plotmesh
from src.fems import p1general
import src.fems.data, src.fems.rt0

#=================================================================#
class CR1(p1general.P1general):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def setMesh(self, mesh):
        super().setMesh(mesh)
        self.computeStencilCell(self.mesh.facesOfCells)
        self.cellgrads = self.computeCellGrads()
    def nlocal(self): return self.mesh.dimension+1
    def nunknowns(self): return self.mesh.nfaces
    def dofspercell(self): return self.mesh.facesOfCells
    def tonode(self, u):
        # print(f"{u=}")
        unodes = np.zeros(self.mesh.nnodes)
        if u.shape[0] != self.mesh.nfaces: raise ValueError(f"{u.shape=} {self.mesh.nfaces=}")
        scale = self.mesh.dimension
        np.add.at(unodes, self.mesh.simplices.T, np.sum(u[self.mesh.facesOfCells], axis=1))
        np.add.at(unodes, self.mesh.simplices.T, -scale*u[self.mesh.facesOfCells].T)
        countnodes = np.zeros(self.mesh.nnodes, dtype=int)
        np.add.at(countnodes, self.mesh.simplices.T, 1)
        unodes /= countnodes
        # print(f"{unodes=}")
        return unodes
    # def prepareAdvection(self, beta, scale):
    #     method = self.params_str['convmethod']
    #     rt = src.fems.rt0.RT0(mesh=self.mesh)
    #     betart = scale*rt.interpolate(beta)
    #     beta = rt.toCell(betart)
    #     convdata = src.fems.data.ConvectionData(beta=beta, betart=betart)
    #     dim = self.mesh.dimension
    #     self.mesh.constructInnerFaces()
    #     if method == 'upwalg' or method == 'lps':
    #          return convdata 
    #     elif method == 'supg':
    #         md = move.move_midpoints(self.mesh, beta, bound=1/dim)
    #         # self.md = move.move_midpoints(self.mesh, beta, candidates='all')
    #         # self.md.plot(self.mesh, beta, type='midpoints')
    #     elif method == 'supg2':
    #         md = move.move_midpoint_to_neighbour(self.mesh, betart)
    #         # self.md = move.move_midpoints(self.mesh, beta, candidates='all')
    #         # self.md = move.move_midpoints(self.mesh, beta, candidates='all')
    #         # print(f"{self.md.mus=}")
    #         # self.md.plot(self.mesh, beta, type='midpoints')
    #     elif method == 'upw':
    #         md = move.move_midpoints(self.mesh, beta, bound=1/dim)
    #         # self.md = move.move_midpoint_to_neighbour(self.mesh, betart)
    #         # self.md = move.move_midpoints(self.mesh, -beta, bound=1/d)
    #         # self.md = move.move_midpoints(self.mesh, -beta, candidates='all')
    #         # self.md.plot(self.mesh, beta, type='midpoints')
    #     elif method == 'upw2':
    #         md = move.move_midpoints(self.mesh, -beta, bound=1/dim)
    #     else:
    #         raise ValueError(f"don't know {method=}")
    #     convdata.md = md
    #     return convdata
    def computeCellGrads(self):
        normals, facesOfCells, dV = self.mesh.normals, self.mesh.facesOfCells, self.mesh.dV
        return (normals[facesOfCells].T * self.mesh.sigma.T / dV.T).T
    # strong bc
    def prepareBoundary(self, colorsdir, colorsflux=[]):
        bdrydata = src.fems.data.BdryData()
        bdrydata.facesdirall = np.empty(shape=(0), dtype=np.uint32)
        bdrydata.colorsdir = colorsdir
        for color in colorsdir:
            facesdir = self.mesh.bdrylabels[color]
            bdrydata.facesdirall = np.unique(np.union1d(bdrydata.facesdirall, facesdir))
        bdrydata.facesinner = np.setdiff1d(np.arange(self.mesh.nfaces, dtype=int), bdrydata.facesdirall)
        bdrydata.facesdirflux = {}
        for color in colorsflux:
            bdrydata.facesdirflux[color] = self.mesh.bdrylabels[color]
        return bdrydata
    def computeRhsNitscheDiffusion(self, nitsche_param, b, diffcoff, colorsdir, udir=None, bdrycondfct=None, coeff=1, lumped=False):
        # if self.params_str['dirichletmethod'] != 'nitsche': return
        if udir is None:
            udir = self.interpolateBoundary(colorsdir, bdrycondfct)
        # nitsche_param=self.params_float['nitscheparam']
        if not udir.shape[0] == self.mesh.nfaces:
            raise ValueError(f"{udir.shape[0]=} {self.mesh.nfaces=}")
        dim, faces = self.mesh.dimension, self.mesh.bdryFaces(colorsdir)
        cells = self.mesh.cellsOfFaces[faces,0]
        normalsS = self.mesh.normals[faces][:,:dim]
        dS, dV = np.linalg.norm(normalsS,axis=1), self.mesh.dV[cells]
        mat = np.einsum('f,fi,fji->fj', coeff*udir[faces]*diffcoff[cells], normalsS, self.cellgrads[cells, :, :dim])
        np.add.at(b, self.mesh.facesOfCells[cells], -mat)
        self.massDotBoundary(b, f=udir, colors=colorsdir, coeff=coeff*nitsche_param*diffcoff[cells] * dS/dV, lumped=lumped)
    def computeFormNitscheDiffusion(self, nitsche_param, du, u, diffcoff, colorsdir):
        # if self.params_str['dirichletmethod'] != 'nitsche': return
        # nitsche_param=self.params_float['nitscheparam']
        assert u.shape[0] == self.mesh.nfaces
        dim, faces = self.mesh.dimension, self.mesh.bdryFaces(colorsdir)
        cells = self.mesh.cellsOfFaces[faces,0]
        foc, normalsS, cellgrads = self.mesh.facesOfCells[cells], self.mesh.normals[faces][:,:dim], self.cellgrads[cells, :, :dim]
        dS, dV = np.linalg.norm(normalsS,axis=1), self.mesh.dV[cells]
        mat = np.einsum('f,fk,fik->fi', u[faces]*diffcoff[cells], normalsS, cellgrads)
        np.add.at(du, foc, -mat)
        mat = np.einsum('f,fk,fjk,fj->f', diffcoff[cells], normalsS, cellgrads, u[foc])
        np.add.at(du, faces, -mat)
        self.massDotBoundary(du, f=u, colors=colorsdir, coeff=nitsche_param*diffcoff[cells]* dS/dV)
    def computeMatrixNitscheDiffusion(self, nitsche_param, diffcoff, colors, coeff=1, lumped=False):
        nfaces, ncells, dim, nlocal  = self.mesh.nfaces, self.mesh.ncells, self.mesh.dimension, self.nlocal()
        # if self.params_str['dirichletmethod'] != 'nitsche': return sparse.coo_matrix((nfaces,nfaces))
        # nitsche_param=self.params_float['nitscheparam']
        faces = self.mesh.bdryFaces(colors)
        if not isinstance(coeff, (float,int)): assert coeff.shape[0]==faces.shape[0]
        cells = self.mesh.cellsOfFaces[faces, 0]
        normalsS = self.mesh.normals[faces][:, :dim]
        cols = self.mesh.facesOfCells[cells, :].ravel()
        rows = faces.repeat(nlocal)
        mat = np.einsum('f,fi,fji->fj', coeff*diffcoff[cells], normalsS, self.cellgrads[cells, :, :dim]).ravel()
        AN = sparse.coo_matrix((mat, (rows, cols)), shape=(nfaces, nfaces)).tocsr()
        # AD = sparse.diags(AN.diagonal(), offsets=(0), shape=(nfaces, nfaces))
        dS = np.linalg.norm(normalsS,axis=1)
        dV = self.mesh.dV[cells]
        # AD = sparse.coo_matrix((dS**2/dV,(faces,faces)), shape=(nfaces, nfaces))
        AD = self.computeBdryMassMatrix(colors=colors, coeff=coeff*diffcoff[cells]*nitsche_param*dS/dV, lumped=lumped)
        return AD - AN - AN.T
    def computeBdryNormalFluxNitsche(self, nitsche_param, u, colors, udir, diffcoff):
        # nitsche_param=self.params_float['nitscheparam']
        #TODO correct flux computation Nitsche
        flux= np.zeros(len(colors))
        nfaces, ncells, dim, nlocal  = self.mesh.nfaces, self.mesh.ncells, self.mesh.dimension, self.nlocal()
        facesOfCell = self.mesh.facesOfCells
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            cells = self.mesh.cellsOfFaces[faces, 0]
            normalsS = self.mesh.normals[faces,:dim]
            cellgrads = self.cellgrads[cells, :, :dim]
            foc = facesOfCell[cells]
            flux[i] = np.einsum('fj,f,fi,fji->', u[foc], diffcoff[cells], normalsS, cellgrads)
            dS = np.linalg.norm(normalsS,axis=1)
            dV = self.mesh.dV[cells]
            flux[i] -= self.massDotBoundary(b=None, f=u-udir, colors=[color], coeff=nitsche_param * diffcoff[cells]*dS/dV)
            # flux[i] /= np.sum(dS)
        return flux
    # def formBoundary(self, du, u, bdrydata, kheatcell, colorsdir):
    #     method = self.params_str['dirichletmethod']
    #     if method == 'new':
    #         nodedirall = bdrydata.nodedirall
    #         du[nodedirall] += bdrydata.A_dir_dir * u[bdrydata.nodedirall]
    #     elif method == "nitsche":
    #         self.computeFormNitscheDiffusion(du, u, kheatcell, colorsdir)
    def vectorBoundaryStrongEqual(self, du, u, bdrydata):
        # if self.params_str['dirichletmethod']=="nitsche": return
        facesdirall = bdrydata.facesdirall
        du[facesdirall] = u[facesdirall]
    def vectorBoundaryStrongZero(self, du, bdrydata):
        # if self.params_str['dirichletmethod']=="nitsche": return
        du[bdrydata.facesdirall] = 0
    def vectorBoundaryStrong(self, b, bdrycond, bdrydata):
        # method = self.params_str['dirichletmethod']
        # if method not in ['strong','new']: return
        facesdirflux, facesinner, facesdirall, colorsdir = bdrydata.facesdirflux, bdrydata.facesinner, bdrydata.facesdirall, bdrydata.colorsdir
        x, y, z = self.mesh.pointsf.T
        for color, faces in facesdirflux.items():
            bdrydata.bsaved[color] = b[faces]
        help = np.zeros_like(b)
        for color in colorsdir:
            faces = self.mesh.bdrylabels[color]
            if color in bdrycond.fct:
                dirichlet = bdrycond.fct[color]
                help[faces] = dirichlet(x[faces], y[faces], z[faces])
        # b[facesinner] -= bdrydata.A_inner_dir * help[facesdirall]
        # if method == 'strong':
        b[facesdirall] = help[facesdirall]
        # else:
        #     b[facesdirall] = bdrydata.A_dir_dir * help[facesdirall]
    def matrixBoundaryStrong(self, A, bdrydata):
        # method = self.params_str['dirichletmethod']
        # if method not in ['strong','new']: return
        facesdirflux, facesinner, facesdirall, colorsdir = bdrydata.facesdirflux, bdrydata.facesinner, bdrydata.facesdirall, bdrydata.colorsdir
        nfaces = self.mesh.nfaces
        for color, faces in facesdirflux.items():
            nb = faces.shape[0]
            help = sparse.dok_matrix((nb, nfaces))
            for i in range(nb): help[i, faces[i]] = 1
            bdrydata.Asaved[color] = help.dot(A)
        bdrydata.A_inner_dir = A[facesinner, :][:, facesdirall]
        help = np.ones((nfaces))
        help[facesdirall] = 0
        help = sparse.dia_matrix((help, 0), shape=(nfaces, nfaces))
        # A = help.dot(A.dot(help))
        diag = np.zeros((nfaces))
        if method == 'strong':
            diag[facesdirall] = 1.0
            diag = sparse.dia_matrix((diag, 0), shape=(nfaces, nfaces))
        else:
            bdrydata.A_dir_dir = self.dirichlet_strong*A[facesdirall, :][:, facesdirall]
            diag[facesdirall] = np.sqrt(self.dirichlet_strong)
            diag = sparse.dia_matrix((diag, 0), shape=(nfaces, nfaces))
            diag = diag.dot(A.dot(diag))
        A = help.dot(A)
        A += diag
        return A
    # interpolate
    def interpolate(self, f):
        x, y, z = self.mesh.pointsf.T
        return f(x, y, z)
    def interpolateBoundary(self, colors, f, lumped=False):
        """
        :param colors: set of colors to interpolate
        :param f: ditct of functions
        :return:
        """
        b = np.zeros(self.mesh.nfaces)
        if lumped:
            for color in colors:
                if not color in f or not f[color]: continue
                faces = self.mesh.bdrylabels[color]
                ci = self.mesh.cellsOfFaces[faces][:, 0]
                foc = self.mesh.facesOfCells[ci]
                mask = foc != faces[:, np.newaxis]
                fi = foc[mask].reshape(foc.shape[0], foc.shape[1] - 1)
                normalsS = self.mesh.normals[faces]
                dS = linalg.norm(normalsS, axis=1)
                normalsS = normalsS/dS[:,np.newaxis]
                nx, ny, nz = normalsS.T
                x, y, z = self.mesh.pointsf[faces].T
                try:
                    b[faces] = f[color](x, y, z, nx, ny, nz)
                except:
                    b[faces] = f[color](x, y, z)
            return b
        for color in colors:
            if not color in f or not f[color]: continue
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            normalsS = normalsS / dS[:, np.newaxis]
            nx, ny, nz = normalsS.T
            ci = self.mesh.cellsOfFaces[faces][:, 0]
            foc = self.mesh.facesOfCells[ci]
            x, y, z = self.mesh.pointsf[foc].T
            nx, ny, nz = normalsS.T
            import inspect
            # print(f"{len(inspect.signature(f[color]).parameters)=}")
            # print(f"{str(inspect.signature(f[color]).parameters)=}")
            # if 'nx' in str(inspect.signature(f[color])):
            if len(inspect.signature(f[color]).parameters) >= 6:
                # ff = f[color](x, y, z, nx[None,:], ny[None,:], nz[None,:])
                ff = f[color](x, y, z, nx, ny, nz)
            else:
                ff = np.vectorize(f[color])(x, y, z)
            np.put(b, foc, ff.T)
        return b
    # matrices
    def masslocal(self):
        dim = self.mesh.dimension
        scalemass = (2 - dim) / (dim + 1) / (dim + 2)
        massloc = np.tile(scalemass, (self.nloc, self.nloc))
        scale = (2 - dim + dim * dim) / (dim + 1) / (dim + 2)
        massloc.reshape((self.nloc * self.nloc))[::self.nloc + 1] = scale
        return massloc
    def _computeMassMatrix(self, coeff=1):
        dim, dV = self.mesh.dimension, self.mesh.dV
        return np.einsum('n,kl->nkl', coeff*dV, self.masslocal())
    def computeMassMatrix(self, coeff=1, lumped=False):
        if lumped:
            dim, dV = self.mesh.dimension, self.mesh.dV
            nfaces, facesOfCells = self.mesh.nfaces, self.mesh.facesOfCells
            mass = coeff/(dim+1)*dV.repeat(dim+1)
            rows = self.mesh.facesOfCells.ravel()
            return sparse.coo_matrix((mass, (rows, rows)), shape=(nfaces, nfaces)).tocsr()
        nfaces = self.mesh.nfaces
        mass = self._computeMassMatrix(coeff)
        return sparse.coo_matrix((mass.ravel(), (self.rows, self.cols)), shape=(nfaces, nfaces)).tocsr()
    def computeBdryMassMatrix(self, colors=None, coeff=1, lumped=False):
        nfaces, dim = self.mesh.nfaces, self.mesh.dimension
        massloc = barycentric.crbdryothers(dim)
        # lumped = False
        if colors is None: colors = self.mesh.bdrylabels.keys()

        if not isinstance(coeff, dict):
            faces = self.mesh.bdryFaces(colors)
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            if isinstance(coeff, (int,float)): dS *= coeff
            elif coeff.shape[0]==self.mesh.nfaces: dS *= coeff[faces]
            elif coeff.shape[0]==dS.shape[0]: dS *= coeff
            else: raise  ValueError(f"cannot handle {coeff=}")
            AD = sparse.coo_matrix((dS, (faces, faces)), shape=(nfaces, nfaces))
            if lumped: return AD
            ci = self.mesh.cellsOfFaces[faces][:,0]
            foc = self.mesh.facesOfCells[ci]
            mask = foc != faces[:, np.newaxis]
            # print(f"{mask=}")
            fi = foc[mask].reshape(foc.shape[0], foc.shape[1] - 1)
            # print(f"{massloc=}")
            cols = np.tile(fi, dim).ravel()
            rows = np.repeat(fi, dim).ravel()
            mat = np.einsum('n,kl->nkl', dS, massloc).ravel()
            return AD + sparse.coo_matrix((mat, (rows, cols)), shape=(nfaces, nfaces))


        assert(isinstance(coeff, dict))
        rows = np.empty(shape=(0), dtype=int)
        cols = np.empty(shape=(0), dtype=int)
        mat = np.empty(shape=(0), dtype=float)
        for color in colors:
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)*coeff[color]
            cols = np.append(cols, faces)
            rows = np.append(rows, faces)
            mat = np.append(mat, dS)
            if not lumped:
                ci = self.mesh.cellsOfFaces[faces][:,0]
                foc = self.mesh.facesOfCells[ci]
                mask = foc != faces[:, np.newaxis]
                fi = foc[mask].reshape(foc.shape[0], foc.shape[1] - 1)
                # print(f"{massloc=}")
                cols = np.append(cols, np.tile(fi, dim).ravel())
                rows = np.append(rows, np.repeat(fi, dim).ravel())
                mat = np.append(mat, np.einsum('n,kl->nkl', dS, massloc).ravel())
        # print(f"{mat=}")
        return sparse.coo_matrix((mat, (rows, cols)), shape=(nfaces, nfaces)).tocsr()
    def massDotBoundary(self, b=None, f=None, colors=None, coeff=1, lumped=False):
        #TODO CR1 boundary integrals: can do at ones since last index in facesOfCells is the bdry side:
        # assert np.all(faces == foc[:,-1])
        if colors is None: colors = self.mesh.bdrylabels.keys()
        massloc = barycentric.crbdryothers(self.mesh.dimension)
        if not isinstance(coeff, dict):
            faces = self.mesh.bdryFaces(colors)
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            if isinstance(coeff, (int,float)): dS *= coeff
            elif coeff.shape[0]==self.mesh.nfaces: dS *= coeff[faces]
            else: dS *= coeff
            if b is None: bsum = np.sum(dS*f[faces])
            else: b[faces] += dS*f[faces]
            ci = self.mesh.cellsOfFaces[faces][:, 0]
            foc = self.mesh.facesOfCells[ci]
            mask = foc!=faces[:,np.newaxis]
            fi = foc[mask].reshape(foc.shape[0],foc.shape[1]-1)
            r = np.einsum('n,kl,nl->nk', dS, massloc, f[fi])
            if b is None: return bsum+np.sum(r)
            # print(f"{np.linalg.norm(f[fi])=}")
            np.add.at(b, fi, r)
            return b
        assert(isinstance(coeff, dict))
        for color in colors:
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            dS *= coeff[color]
            b[faces] += dS*f[faces]
            if lumped: continue
            ci = self.mesh.cellsOfFaces[faces][:, 0]
            foc = self.mesh.facesOfCells[ci]
            mask = foc!=faces[:,np.newaxis]
            fi = foc[mask].reshape(foc.shape[0],foc.shape[1]-1)
            r = np.einsum('n,kl,nl->nk', dS, massloc, f[fi])
            # print(f"{np.linalg.norm(f[fi])=}")
            np.add.at(b, fi, r)
        return b
    def computeFormTransportCellWise(self, du, u, data, type):
        beta, betart = data.beta, data.betart
        nfaces, dim, dV, foc = self.mesh.nfaces, self.mesh.dimension, self.mesh.dV, self.mesh.facesOfCells
        cellgrads = self.cellgrads[:,:,:dim]
        if type=='centered':
            mat = np.einsum('n,njk,nk,i,nj -> ni', dV, cellgrads, beta, 1/(dim+1)*np.ones(dim+1),u[foc])
        elif type=='supg':
            mus = data.md.mus
            mat = np.einsum('n,njk,nk,ni,nj -> ni', dV, cellgrads, beta, 1-dim*mus,u[foc])
        else: raise ValueError(f"unknown type {type=}")
        np.add.at(du, foc, mat)
        self.massDotBoundary(du, u, coeff=-np.minimum(betart, 0))
    def computeMatrixTransportCellWise(self, data, type):
        beta, betart = data.beta, data.betart
        nfaces, dim, dV = self.mesh.nfaces, self.mesh.dimension, self.mesh.dV
        cellgrads = self.cellgrads[:,:,:dim]
        if type=='centered':
            # betagrad = np.einsum('njk,nk -> nj', cellgrads, beta)
            mat = np.einsum('n,njk,nk,i -> nij', dV, cellgrads, beta, 1/(dim+1)*np.ones(dim+1))
            # mat += np.einsum('n,nj,ni -> nij', dV*deltas, betagrad, betagrad)
        elif type=='supg':
            mus = data.md.mus
            mat = np.einsum('n,njk,nk,ni -> nij', dV, cellgrads, beta, 1-dim*mus)
        else: raise ValueError(f"unknown type {type=}")
        A = sparse.coo_matrix((mat.ravel(), (self.rows, self.cols)), shape=(nfaces, nfaces))
        return A - self.computeBdryMassMatrix(coeff=np.minimum(betart, 0), lumped=False)
    def computeMatrixJump(self, betart, mode='primal', monotone=False):
        dim, dV, nfaces, ndofs = self.mesh.dimension, self.mesh.dV, self.mesh.nfaces, self.nunknowns()
        nloc, dofspercell = self.nlocal(), self.dofspercell()
        innerfaces = self.mesh.innerfaces
        ci0 = self.mesh.cellsOfInteriorFaces[:,0]
        ci1 = self.mesh.cellsOfInteriorFaces[:,1]
        normalsS = self.mesh.normals[innerfaces]
        dS = linalg.norm(normalsS, axis=1)
        faces = self.mesh.faces[self.mesh.innerfaces]
        # ind0 = npext.positionin(faces, self.mesh.simplices[ci0])
        # ind1 = npext.positionin(faces, self.mesh.simplices[ci1])
        # fi0 = np.take_along_axis(self.mesh.facesOfCells[ci0], ind0, axis=1)
        # fi1 = np.take_along_axis(self.mesh.facesOfCells[ci1], ind1, axis=1)
        fi0, fi1 = self.mesh.facesOfCellsNotOnInnerFaces(ci0, ci1)
        ifaces = np.arange(nfaces)[innerfaces]
        A = sparse.coo_matrix((ndofs, ndofs))
        rows0 = np.repeat(fi0, nloc-1).ravel()
        cols0 = np.tile(fi0,nloc-1).ravel()
        rows1 = np.repeat(fi1, nloc-1).ravel()
        cols1 = np.tile(fi1,nloc-1).ravel()
        massloc = barycentric.crbdryothers(self.mesh.dimension)
        if mode == 'primal':
            mat = np.einsum('n,kl->nkl', np.minimum(betart[innerfaces], 0) * dS, massloc).ravel()
            A -= sparse.coo_matrix((mat, (rows0, cols0)), shape=(ndofs, ndofs))
            A += sparse.coo_matrix((mat, (rows0, cols1)), shape=(ndofs, ndofs))
            mat = np.einsum('n,kl->nkl', np.maximum(betart[innerfaces], 0)*dS, massloc).ravel()
            A -= sparse.coo_matrix((mat, (rows1, cols0)), shape=(ndofs, ndofs))
            A += sparse.coo_matrix((mat, (rows1, cols1)), shape=(ndofs, ndofs))
        elif mode =='dual':
            mat = np.einsum('n,kl->nkl', np.minimum(betart[innerfaces], 0) * dS, massloc).ravel()
            A += sparse.coo_matrix((mat, (rows0, cols1)), shape=(ndofs, ndofs))
            A -= sparse.coo_matrix((mat, (rows1, cols1)), shape=(ndofs, ndofs))
            mat = np.einsum('n,kl->nkl', np.maximum(betart[innerfaces], 0) * dS, massloc).ravel()
            A += sparse.coo_matrix((mat, (rows0, cols0)), shape=(ndofs, ndofs))
            A -= sparse.coo_matrix((mat, (rows1, cols0)), shape=(ndofs, ndofs))
        elif mode =='centered':
            mat = np.einsum('n,kl->nkl', betart[innerfaces] * dS, massloc).ravel()
            A += sparse.coo_matrix((mat, (rows0, cols0)), shape=(ndofs, ndofs))
            A -= sparse.coo_matrix((mat, (rows1, cols1)), shape=(ndofs, ndofs))
        else:
            raise ValueError(f"unknown {mode=}")
        return A
    def computeFormJump(self, du, u, betart, mode='primal'):
        dim, dV, nfaces, ndofs = self.mesh.dimension, self.mesh.dV, self.mesh.nfaces, self.nunknowns()
        nloc, dofspercell = self.nlocal(), self.dofspercell()
        if not hasattr(self.mesh,'innerfaces'): self.mesh.constructInnerFaces()
        innerfaces = self.mesh.innerfaces
        ci0 = self.mesh.cellsOfInteriorFaces[:,0]
        ci1 = self.mesh.cellsOfInteriorFaces[:,1]
        normalsS = self.mesh.normals[innerfaces]
        dS = linalg.norm(normalsS, axis=1)
        faces = self.mesh.faces[self.mesh.innerfaces]
        # ind0 = npext.positionin(faces, self.mesh.simplices[ci0])
        # ind1 = npext.positionin(faces, self.mesh.simplices[ci1])
        # fi0 = np.take_along_axis(self.mesh.facesOfCells[ci0], ind0, axis=1)
        # fi1 = np.take_along_axis(self.mesh.facesOfCells[ci1], ind1, axis=1)
        fi0, fi1 = self.mesh.facesOfCellsNotOnInnerFaces(ci0, ci1)
        # fi0, fi1 = self.mesh.facesOfCellsNotOnFaces(faces, ci0, ci1)
        # ifaces = np.arange(nfaces)[innerfaces]
        # A = sparse.coo_matrix((ndofs, ndofs))
        # rows0 = np.repeat(fi0, nloc-1).ravel()
        # cols0 = np.tile(fi0,nloc-1).ravel()
        # rows1 = np.repeat(fi1, nloc-1).ravel()
        # cols1 = np.tile(fi1,nloc-1).ravel()
        massloc = barycentric.crbdryothers(self.mesh.dimension)
        if mode == 'primal':
            mat = np.einsum('n,kl,nl->nk', np.minimum(betart[innerfaces], 0) * dS, massloc, u[fi1]-u[fi0])
            np.add.at(du, fi0, mat)
            mat = np.einsum('n,kl,nl->nk', np.maximum(betart[innerfaces], 0)*dS, massloc, u[fi1]-u[fi0])
            np.add.at(du, fi1, mat)
        elif mode =='dual':
            assert 0
        elif mode =='centered':
            assert 0
        else:
            raise ValueError(f"unknown {mode=}")
    def computeMassMatrixSupg(self, xd, coeff=1):
        raise NotImplemented(f"computeMassMatrixSupg")
    def computeMatrixTransportUpwindAlg(self, data):
        A = self.computeMatrixTransportCellWise(data, type='centered')
        # moins bon en L2, mais meilleur en H1
        # A += self.computeMatrixJump(betart)
        self.diffalg = checkmmatrix.diffusionForMMatrix(A)
        return A + self.diffalg
    def computeMatrixTransportUpwind(self, data, method):
        return self.computeMatrixTransportUpwindAlg(data)

        beta, betart, mus = data.beta, data.betart, data.md.mus
        nfaces, ncells, dim, dV = self.mesh.nfaces, self.mesh.ncells, self.mesh.dimension, self.mesh.dV
        normalsS, foc, cof = self.mesh.normals, self.mesh.facesOfCells, self.mesh.cellsOfFaces
        dbS = linalg.norm(normalsS, axis=1)*betart

        innerfaces = self.mesh.innerfaces
        infaces = np.arange(nfaces)[innerfaces]
        ci0 = self.mesh.cellsOfInteriorFaces[:, 0]
        ci1 = self.mesh.cellsOfInteriorFaces[:, 1]
        matloc = np.ones(shape=(dim+1))/(dim+1)
        cols = np.repeat(infaces, dim + 1).ravel()
        #centered
        # rows0 = foc[ci0].ravel()
        # rows1 = foc[ci1].ravel()
        # mat = np.einsum('n,i->ni', dbS[infaces], matloc).ravel()
        # A = sparse.coo_matrix((mat, (rows0, cols)), shape=(nfaces, nfaces))
        # A -= sparse.coo_matrix((mat, (rows1, cols)), shape=(nfaces, nfaces))

        if method=='upw2':
            rows0 = foc[ci0].ravel()
            rows1 = foc[ci1].ravel()
            mat = np.einsum('n,ni->ni', dbS[infaces], 1-dim*mus[ci0]).ravel()
            A = -sparse.coo_matrix((mat, (cols, rows0)), shape=(nfaces, nfaces))
            mat = np.einsum('n,ni->ni', dbS[infaces], 1-dim*mus[ci1]).ravel()
            A += sparse.coo_matrix((mat, (cols, rows1)), shape=(nfaces, nfaces))
            faces = self.mesh.bdryFaces()
            ci0 = self.mesh.cellsOfFaces[faces, 0]
            rows0 = foc[ci0].ravel()
            cols = np.repeat(faces, dim + 1).ravel()
            # mat = np.einsum('n,ni->ni', dbS[faces], 1-dim*mus[ci0]).ravel()
            mat = np.einsum('n,i->ni', dbS[faces], matloc).ravel()
            A -= sparse.coo_matrix((mat, (cols,rows0)), shape=(nfaces, nfaces))
            A += self.computeMatrixJump(betart, mode='dual')
            A += self.computeBdryMassMatrix(coeff=np.maximum(betart, 0), lumped=False)
            # B = self.computeMatrixTransportCellWise(type='centered')
            # if not np.allclose(A.tocsr().A,B.tocsr().A):
            #     raise ValueError(f"{A.diagonal()=}\n{B.diagonal()=}")
            return A

        #supg
        rows0 = foc[ci0].ravel()
        rows1 = foc[ci1].ravel()
        mat = np.einsum('n,ni->ni', dbS[infaces], 1-dim*mus[ci0]).ravel()
        A = sparse.coo_matrix((mat, (rows0, cols)), shape=(nfaces, nfaces))
        mat = np.einsum('n,ni->ni', dbS[infaces], 1-dim*mus[ci1]).ravel()
        A -= sparse.coo_matrix((mat, (rows1, cols)), shape=(nfaces, nfaces))
        faces = self.mesh.bdryFaces()
        ci0 = self.mesh.cellsOfFaces[faces, 0]
        rows0 = foc[ci0].ravel()
        cols = np.repeat(faces, dim + 1).ravel()
        mat = np.einsum('n,ni->ni', dbS[faces], 1-dim*mus[ci0]).ravel()
        A += sparse.coo_matrix((mat, (rows0,cols)), shape=(nfaces, nfaces))
        A += self.computeMatrixJump(betart, mode='primal')
        A -= self.computeBdryMassMatrix(coeff=np.minimum(betart, 0), lumped=False)
        # B = self.computeMatrixTransportSupg(bdrylumped=False, method='supg')
        # if not np.allclose(A.tocsr().A,B.tocsr().A):
        #     raise ValueError(f"{A.diagonal()=}\n{B.diagonal()=}")

        # A += self.computeMatrixJump(betart, mode='primal')
        #upwind

        # mat0 = np.einsum('n,ni->ni', np.minimum(dbS[infaces],0), 1-dim*self.md.mus[self.md.cells[ci0]]).ravel()
        # mat1 = np.einsum('n,i->ni', np.minimum(dbS[infaces],0), matloc).ravel()
        # rows0 = foc[self.md.cells[ci0]].ravel()
        # rows1 = foc[ci1].ravel()
        # A = sparse.coo_matrix((mat0, (rows0, cols)), shape=(nfaces, nfaces))
        # A -= sparse.coo_matrix((mat1, (rows1, cols)), shape=(nfaces, nfaces))
        #
        # mat0 = np.einsum('n,i->ni', np.maximum(dbS[infaces],0), matloc).ravel()
        # mat1 = np.einsum('n,ni->ni', np.maximum(dbS[infaces],0), 1-dim*self.md.mus[self.md.cells[ci1]]).ravel()
        # rows0 = foc[ci0].ravel()
        # rows1 = foc[self.md.cells[ci1]].ravel()
        # A = sparse.coo_matrix((mat0, (rows0, cols)), shape=(nfaces, nfaces))
        # A -= sparse.coo_matrix((mat1, (rows1, cols)), shape=(nfaces, nfaces))


        # faces = self.mesh.bdryFaces()
        # ci0 = self.mesh.cellsOfFaces[faces, 0]
        # rows0 = foc[ci0].ravel()
        # cols = np.repeat(faces, dim + 1).ravel()
        # mat = np.einsum('n,i->ni', dbS[faces], matloc).ravel()
        # A += sparse.coo_matrix((mat, (rows0,cols)), shape=(nfaces, nfaces))
        #
        # A -= self.computeBdryMassMatrix(coeff=np.minimum(betart, 0), lumped=False)
        # A += self.computeMatrixJump(betart, mode='primal')

        # B = self.computeMatrixTransportCellWise(type='centered')
        #
        # A = A.tocsr()
        # B = B.tocsr()
        # if not np.allclose(A.A,B.A):
        #     raise ValueError(f"{A.diagonal()=}\n{B.diagonal()=}\n{A.todense()=}\n{B.todense()=}")

        return A.tocsr()

        # ind = np.arange(nfaces)[dbS>0]
        # # ind = np.arange(nfaces)[np.logical_and(self.mesh.innerfaces,dbS>0)]
        # ci = self.mesh.cellsOfFaces[ind,0].ravel()
        # faces = self.mesh.faces[ind]
        # ind0 = npext.positionin(faces, self.mesh.simplices[ci])
        # fi0 = np.take_along_axis(self.mesh.facesOfCells[ci], ind0, axis=1)
        # rows = fi0.ravel()
        # mat = dbS[ind].repeat(dim)
        # A += sparse.coo_matrix((mat.ravel(), (rows, rows)), shape=(nfaces, nfaces))
        # cols = ind.repeat(dim)
        # A -= sparse.coo_matrix((mat.ravel(), (rows, cols)), shape=(nfaces, nfaces))
        #
        # ind = np.arange(nfaces)[np.logical_and(self.mesh.innerfaces,dbS<0)]
        # ci = self.mesh.cellsOfFaces[ind,1].ravel()
        # faces = self.mesh.faces[ind]
        # ind0 = npext.positionin(faces, self.mesh.simplices[ci])
        # fi0 = np.take_along_axis(self.mesh.facesOfCells[ci], ind0, axis=1)
        # rows = fi0.ravel()
        # mat = dbS[ind].repeat(dim)
        # A -= sparse.coo_matrix((mat.ravel(), (rows, rows)), shape=(nfaces, nfaces))
        # cols = ind.repeat(dim)
        # A += sparse.coo_matrix((mat.ravel(), (rows, cols)), shape=(nfaces, nfaces))
        # sigma = self.mesh.sigma
        # mat = np.einsum('nj,i -> nij', dbS[foc] * sigma, np.ones(dim + 1)).ravel()
        # cols = np.array(self.cols)
        # m = np.where(mat > 0)
        # cols[m] = self.rows[m]
        # A = sparse.coo_matrix((mat, (self.rows, cols)), shape=(nfaces, nfaces))
        # # A += self.computeMatrixJump(betart, mode='primal')
        # A -= self.computeBdryMassMatrix(coeff=np.minimum(betart, 0))
    def computeMatrixTransportSupg(self, data, method):
        beta, betart, mus = data.beta, data.betart, data.md.mus
        if method=='supg2':
            beta, mus, deltas = beta, self.md.mus, self.md.deltas
            nfaces, dim, dV = self.mesh.nfaces, self.mesh.dimension, self.mesh.dV
            cellgrads = self.cellgrads[:,:,:dim]
            mat = np.einsum('n,njk,nk,ni -> nij', dV, cellgrads, beta, 1-dim*mus)
            print(f"{self.mesh.facesOfCells[self.md.cells]=}")
            rows = np.repeat(self.mesh.facesOfCells[self.md.cells], self.nloc).ravel()
            A = sparse.coo_matrix((mat.ravel(), (rows, self.cols)), shape=(nfaces, nfaces))
            A -= self.computeBdryMassMatrix(coeff=np.minimum(betart, 0), lumped=False)
            print(f"{A.diagonal()=}")
            return A
        A = self.computeMatrixTransportCellWise(data, type='supg')
        A += self.computeMatrixJump(betart)
        return A
    def computeMatrixTransportLps(self, data, **kwargs):
        A = self.computeMatrixTransportCellWise(data, type='centered')
        A += self.computeMatrixJump(data.betart)
        A += self.computeMatrixLps(data.betart, **kwargs)
        return A
    def computeFormTransportUpwind(self, du, u, data, method):
        return self.computeFormTransportUpwindAlg(du, u, data)
    def computeFormTransportUpwindAlg(self, du, u, data):
        self.computeFormTransportCellWise(du, u, data, type='centered')
        if hasattr(self,'diffalg'):
            du += self.diffalg@u
    def computeFormTransportSupg(self, du, u, data, method):
        self.computeFormTransportCellWise(du, u, data, type='supg')
        self.computeFormJump(du, u, data.betart)
    def computeFormTransportLps(self, du, u, data, **kwargs):
        self.computeFormTransportCellWise(du, u, data, type='centered')
        self.computeFormJump(du, u, data.betart)
        self.computeFormLps(du, u, data.betart, **kwargs)
    # def massDotSupg(self, b, f, data, coeff=1):
    #     if self.params_str['convmethod'][:4] != 'supg': return
    #     dim, facesOfCells, dV = self.mesh.dimension, self.mesh.facesOfCells, self.mesh.dV
    #     # beta, mus, deltas = beta, self.md.mus, self.md.deltas
    #     # cellgrads = self.cellgrads[:,:,:dim]
    #     # betagrad = np.einsum('njk,nk -> nj', cellgrads, beta)
    #     # r = np.einsum('n,ni->ni', deltas*dV*f[facesOfCells].mean(axis=1), betagrad)
    #     r = np.einsum('n,nk->nk', coeff*dV*f[facesOfCells].mean(axis=1), dim/(dim+1)-dim*data.md.mus)
    #     np.add.at(b, facesOfCells, r)
    #     return b
    # dotmat
    def massDotCell(self, b, f, coeff=1):
        assert f.shape[0] == self.mesh.ncells
        dimension, facesOfCells, dV = self.mesh.dimension, self.mesh.facesOfCells, self.mesh.dV
        massloc = 1/(dimension+1)
        np.add.at(b, facesOfCells, (massloc*coeff*dV*f)[:, np.newaxis])
        return b
    def massDot(self, b, f, coeff=1):
        dim, facesOfCells, dV = self.mesh.dimension, self.mesh.facesOfCells, self.mesh.dV
        scalemass = (2-dim) / (dim+1) / (dim+2)
        massloc = np.tile(scalemass, (self.nloc,self.nloc))
        massloc.reshape((self.nloc*self.nloc))[::self.nloc+1] = (2-dim + dim*dim) / (dim+1) / (dim+2)
        r = np.einsum('n,kl,nl->nk', coeff*dV, massloc, f[facesOfCells])
        np.add.at(b, facesOfCells, r)
        return b
    # rhs
    def computeRhsCell(self, b, rhscell):
        if rhscell is None: return b
        if isinstance(rhscell,dict):
            assert set(rhscell.keys())==set(self.mesh.cellsoflabel.keys())
            dimension, facesOfCells, dV = self.mesh.dimension, self.mesh.facesOfCells, self.mesh.dV
            scale = 1 / (dimension + 1)
            return b
            scale = 1 / (self.mesh.dimension + 1)
            for label, fct in rhscell.items():
                if fct is None: continue
                cells = self.mesh.cellsoflabel[label]
                xc, yc, zc = self.mesh.pointsc[cells].T
                bC = scale * fct(xc, yc, zc) * dV[cells]
                np.add.at(b, facesOfCells, bC)
        else:
            fp1 = self.interpolateCell(rhscell)
            self.massDotCell(b, fp1, coeff=1)
        return b
    # postprocess
    def computeErrorL2Cell(self, solexact, uh):
        xc, yc, zc = self.mesh.pointsc.T
        ec = solexact(xc, yc, zc) - np.mean(uh[self.mesh.facesOfCells], axis=1)
        return np.sqrt(np.sum(ec**2* self.mesh.dV)), ec
    def computeErrorL2(self, solexact, uh):
        x, y, z = self.mesh.pointsf.T
        en = solexact(x, y, z) - uh
        Men = np.zeros_like(en)
        return np.sqrt( np.dot(en, self.massDot(Men,en)) ), en
    def computeErrorFluxL2(self, solexact, uh, diffcell=None):
        xc, yc, zc = self.mesh.pointsc.T
        graduh = np.einsum('nij,ni->nj', self.cellgrads, uh[self.mesh.facesOfCells])
        errv = 0
        for i in range(self.mesh.dimension):
            solxi = solexact.d(i, xc, yc, zc)
            if diffcell is None: errv += np.sum((solxi - graduh[:, i]) ** 2 * self.mesh.dV)
            else: errv += np.sum(diffcell * (solxi - graduh[:, i]) ** 2 * self.mesh.dV)
        return np.sqrt(errv)
    def computeBdryMean(self, u, colors):
        mean, omega = np.zeros(len(colors)), np.zeros(len(colors))
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            omega[i] = np.sum(dS)
            mean[i] = np.sum(dS*u[faces])
        return mean/omega
    def comuteFluxOnRobin(self, u, faces, dS, uR, cR):
        uhmean =  np.sum(dS * u[faces])
        xf, yf, zf = self.mesh.pointsf[faces].T
        nx, ny, nz = np.mean(self.mesh.normals[faces], axis=0)
        if uR:
            try:
                uRmean =  np.sum(dS * uR(xf, yf, zf, nx, ny, nz))
            except:
                uRmean =  np.sum(dS * uR(xf, yf, zf))
        else: uRmean=0
        return cR*(uRmean-uhmean)
    def computeBdryNormalFlux(self, u, colors, bdrydata, bdrycond, diffcoff):
        flux, omega = np.zeros(len(colors)), np.zeros(len(colors))
        for i,color in enumerate(colors):
            faces = self.mesh.bdrylabels[color]
            normalsS = self.mesh.normals[faces]
            dS = linalg.norm(normalsS, axis=1)
            omega[i] = np.sum(dS)
            if color in bdrydata.bsaved.keys():
                bs, As = bdrydata.bsaved[color], bdrydata.Asaved[color]
                flux[i] = np.sum(As * u - bs)
            else:
                flux[i] = self.comuteFluxOnRobin(u, faces, dS, bdrycond.fct[color], bdrycond.param[color])
        return flux

# ------------------------------------- #
if __name__ == '__main__':
    from src.meshes import testmeshes
    from src.meshes import plotmesh
    import matplotlib.pyplot as plt
    mesh = testmeshes.backwardfacingstep(h=0.2)
    fem = CR1(mesh=mesh)
    u = fem.test()
    plotmesh.meshWithBoundaries(mesh)
    plotmesh.meshWithData(mesh, point_data={'u':u}, title="CR1 Test", alpha=1)
    plt.show()
