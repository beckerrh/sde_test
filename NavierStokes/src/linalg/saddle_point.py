import numpy as np
import scipy.sparse as sparse
# from NavierStokes.src.linalg import linalg
# from NavierStokes.src.tools import timer
from src import tools, linalg


#=================================================================#
class SaddlePointSystem():
    """
    A -B.T
    B  0
     or
    A -B.T 0
    B  0   -C^T
    0  C   0
    """
    def __repr__(self):
        string =  f" {self.singleA=} {self.A.shape=} {self.B.shape=}"
        if self.C: string += f"{self.C.shape=}"
        return string
    def __init__(self, vectorview, Blocks, singleA=True):
        self.vectorview = vectorview
        self.singleA, self.ncompA = singleA, vectorview.ncomps[0]
        self.stack_storage = vectorview.stack_storage
        self.nv_single = vectorview.ns[0]
        if len(Blocks)==2:
            self.A, self.B = Blocks
            self.C = None
        elif len(Blocks)==3:
            self.A, self.B, self.C = Blocks
        else:
            raise ValueError("can only handle two or three blocks")
        if self.singleA:
            self.matvec = self.matvec2_singleA if self.C is None else self.matvec3_singleA
        else:
            self.matvec = self.matvec2 if self.C is None else self.matvec3
    def _dot_A(self, b):
        if self.singleA:
            w = np.empty_like(b)
            if self.stack_storage:
                n = self.A.shape[0]
                for i in range(self.ncompA): w[i*n:(i+1)*n] = self.A.dot(b[i*n:(i+1)*n])
            else:
                for i in range(self.ncompA): w[i::self.ncompA] = self.A.dot(b[i::self.ncompA])
            return w
        return self.A.dot(b)
    def scale_matrix(self):
        DA = self.A.diagonal()
        assert np.all(DA>0)
        if self.singleA:
            AD = linalg.linalg.diagmatrix2systemdiagmatrix(1 / DA, self.ncompA, self.stack_storage)
        else:
            AD = sparse.diags(1/DA, offsets=(0), shape=self.A.shape)
        vs = sparse.diags(np.power(AD.diagonal(), 0.5), offsets=(0), shape=AD.shape)
        if self.singleA:
            vss = sparse.diags(np.power(DA, -0.5), offsets=(0), shape=self.A.shape)
            self.A = vss @ self.A @ vss
        else:
            self.A = vs@self.A@vs
        # DS = (self.B@AD@self.B.T).diagonal()
        # print(f"scale_matrix {DA.max()/DA.min()=:8.2f} {DS.max()/DS.min()=:8.2f}")
        # assert np.all(DS>0)
        # makes diag(tilde(B)@tilde(B.T)) = 1
        nb = self.B.shape[0]
        # ps = sparse.diags(np.power(DS, -0.5), offsets=(0), shape=(nb,nb))
        ps = sparse.diags(np.power((self.B@vs**2@self.B.T).diagonal(), -0.5), offsets=(0), shape=(nb,nb))
        # assert np.allclose(ps.data,ps2.data)
        # if not np.allclose(self.A.diagonal(), np.ones(self.A.shape[0])):
        #     raise ValueError(f"not ones on diagona\n{self.A.diagonal()=}")
        self.B = ps@self.B@vs
        self.scalings = [vs, ps]
        if self.C is None: return
        # sparse.diags(
        # DSP = (self.B@self.B.T).diagonal()
        # DSP = 1/DSP
        # print(f"{DSP=}")
        # print(f'{DSP.shape=} {self.C.shape=} {ps.shape=}')
        # ls = self.C@ps@DSP@ps@self.C.T
        # ls = self.C@ps**2@self.C.T
        nc = self.C.shape[0]
        ls = sparse.diags(np.power((self.C@ps**2@self.C.T).diagonal(), -0.5), offsets=(0), shape=(nc,nc))
        # ls = self.C@self.C.T
        # print(f"{ls.todense()=} {(self.C@self.C.T).todense()=}")
        # ls = np.sqrt(ls)
        self.C = ls@self.C@ps
        self.scalings.append(ls)
    def scale_vec(self, b):
        self.vectorview.scale(b, self.scalings)
    def dot(self, x):
        if self.C is None: return self.matvec2(x)
        return self.matvec3(x)
    def matvec3(self, x):
        v, p, lam = self.vectorview.get_parts(x)
        w = self._dot_A(v) - self.B.T.dot(p)
        q = self.B.dot(v) - self.C.T.dot(lam)
        return np.hstack([w, q, self.C.dot(p)])
    def matvec3_singleA(self, x):
        v, p, lam = self.vectorview.get_parts(x)
        w = - self.B.T.dot(p)
        if self.stack_storage:
            n = self.nv_single
            for icomp in range(self.ncompA): w[icomp*n:(icomp+1)*n] += self.A.dot(v[icomp*n:(icomp+1)*n])
        else:
            for icomp in range(self.ncompA): w[icomp::self.ncompA] += self.A.dot(v[icomp::self.ncompA])
        q = self.B.dot(v) - self.C.T.dot(lam)
        return np.hstack([w, q, self.C.dot(p)])
    def matvec2(self, x):
        v, p = self.vectorview.get_parts(x)
        w = self._dot_A(v) - self.B.T.dot(p)
        q = self.B.dot(v)
        return np.hstack([w, q])
    def matvec2_singleA(self, x):
        v, p = self.vectorview.get_parts(x)
        w = - self.B.T.dot(p)
        if self.stack_storage:
            n = self.nv_single
            for icomp in range(self.ncompA): w[icomp*n:(icomp+1)*n] += self.A.dot(v[icomp*n:(icomp+1)*n])
        else:
            for icomp in range(self.ncompA): w[icomp::self.ncompA] += self.A.dot(v[icomp::self.ncompA])
        q = self.B.dot(v)
        return np.hstack([w, q])
    def to_single_matrix(self):
        na, nb = self.A.shape[0], self.B.shape[0]
        nullP = sparse.dia_matrix((np.zeros(nb), 0), shape=(nb, nb))
        if self.singleA:
            A = linalg.matrix2systemdiagonal(self.A, self.ncompA)
        else:
            A = self.A
        A1 = sparse.hstack([A, -self.B.T])
        A2 = sparse.hstack([self.B, nullP])
        Aall = sparse.vstack([A1, A2])
        if not self.C:
            return Aall.tocsr()
        nc = self.C.shape[0]
        nullV = sparse.coo_matrix((1, self.na)).tocsr()
        ML = sparse.hstack([nullV, self.C])
        Abig = sparse.hstack([Aall, -ML.T])
        nullL = sparse.dia_matrix((np.zeros(nc), 0), shape=(nc, nc))
        Cbig = sparse.hstack([ML, nullL])
        Aall = sparse.vstack([Abig, Cbig])
        return Aall.tocsr()


#=================================================================#
class SaddlePointPreconditioner():
    """
    """
    def __repr__(self):
        s =  f"{self.method=}\n{self.type=}"
        if hasattr(self,'SV'): s += f"\n{self.SV=}"
        if hasattr(self,'SP'): s += f"\n{self.SP=}"
        return s
    def _get_schur_of_diag(self):
        # We suppose that diag(A) = Id !!!!!!!!!!!!!!!
        if self.AS.C is None:
            return self.AS.B @ self.AS.B.T
        return [self.AS.B @ self.AS.B.T, self.AS.C @ self.AS.C.T]

        if not np.allclose(self.AS.A.diagonal(), np.ones(self.AS.A.shape[0])):
            raise ValueError(f"{self.AS.A.data=}")
        # not necessary since diag(A) = Id
        if ret_diag: return self.AS.B @ AD @ self.AS.B.T, AD
        return self.AS.B @ AD @ self.AS.B.T

        if self.AS.singleA:
            AD = linalg.diagmatrix2systemdiagmatrix(1 / self.AS.A.diagonal(), self.AS.ncompA, self.stack_storage)
        else:
            AD = sparse.diags(1 / self.AS.A.diagonal(), offsets=(0), shape=self.AS.A.shape)
        if ret_diag: return self.AS.B @ AD @ self.AS.B.T, AD
        return self.AS.B @ AD @ self.AS.B.T
    def __init__(self, AS, **kwargs):
        if not isinstance(AS, SaddlePointSystem):
            raise ValueError(f"*** resuired arguments: AS (SaddlePointSystem)")
        self.vectorview = AS.vectorview
        self.stack_storage = AS.stack_storage
        self.AS = AS
        self.nv_single = AS.B.shape[1]//AS.ncompA
        self.nv = self.vectorview.starts[1]
        self.nvp = self.vectorview.starts[2]
        self.nall = self.vectorview.starts[-1]
        self.method = kwargs.pop('method', None)
        solver_p = kwargs.pop('solver_p', None)
        solver_v = kwargs.pop('solver_v', None)
        if solver_v is None:
            solver_v = {'method': 'pyamg', 'maxiter': 1, 'disp':0}
        if solver_p is None:
            solver_p = {'method': 'pyamg', 'maxiter': 1, 'disp':0, 'symmetric':True}
        solver_v['counter'] = '\tV '
        solver_v['matrix'] = self.AS.A
        solver_p['counter'] = '\tP '
        if self.AS.C is None:
            solver_p['matrix'] = self._get_schur_of_diag()
        else:
            psd, lsd = self._get_schur_of_diag()
            solver_p['matrix'] = psd
            self.lsd = lsd
        # print(f"{solver_v=}\n {solver_p=}")
        self.SP = linalg.linalg.getLinearSolver(**solver_p)
        self.SV = linalg.linalg.getLinearSolver(**solver_v)
        if len(kwargs.keys()): raise ValueError(f"*** unused arguments {kwargs=}")
        verbose_timer = kwargs.pop("verbose_timer", False)
        self.timer = tools.timer.Timer(verbose_del=verbose_timer)

    def update(self, A):
        self.AS = A
        self.SV.update(self.AS.A)
    def _solve_v(self,b):
        if self.AS.singleA:
            w = np.empty_like(b)
            if self.stack_storage:
                n = self.nv_single
                for i in range(self.AS.ncompA): w[i*n: (i+1)*n] = self.SV.solve(b=b[i*n: (i+1)*n])
            else:
                for i in range(self.AS.ncompA): w[i::self.AS.ncompA] = self.SV.solve(b=b[i::self.AS.ncompA])
            return w
        return self.SV.solve(b=b)
#=================================================================#
class BraessSarazin(SaddlePointPreconditioner):
    """
    Instead of
    --------
    A -B.T
    B  0
    --------
    solve
    --------
    alpha*diag(A)  -B.T
    B               0
    --------
    S = B*diag(A)^{-1}*B.T
    S p = alpha*g - B C^{-1}f
    C v = 1/alpha*f + 1/alpha*B.T*p
    """
    def __repr__(self):
        return self.__class__.__name__ + f"{self.alpha=}"
        return s
    def __init__(self, AS, **kwargs):
        self.alpha = kwargs.pop('alpha',10)
        super().__init__(AS, **kwargs)
    def solve(self, x):
        v, p = x[:self.nv], x[self.nv:]
        w = self.SV.solve(b=v)/self.alpha
        self.timer.add("solve_v")
        q = self.SP.solve(b=self.alpha*p-self.AS.B.dot(w))
        self.timer.add("solve_p")
        h = self.AS.B.T.dot(q)
        w += self.SV.solve(h)/self.alpha
        self.timer.add("solve_v")
        return np.hstack([w, q])
#=================================================================#
class Chorin(SaddlePointPreconditioner):
    """
    Instead of
    --------
    A -B.T
    B  0
    --------
    solve
    --------
    A  -B.T
    0   B@B.T
    --------
    """
    def __repr__(self):
        return self.__class__.__name__
    def __init__(self, AS, **kwargs):
        super().__init__(AS, **kwargs)
        if self.AS.C is None:
            self.solve = self.solve2
        else:
            self.solve = self.solve3
            # self.c = (self.AS.C @ self.AS.C.T).todense()[0,0]
            # print(f"{self.c=}")
    def solve_simple(self, x):
        v, p = x[:self.nv], x[self.nv:]
        q = self.SP.solve(b=p)
        w = self._solve_v(b=v+self.AS.B.T.dot(q))
        return np.hstack([w, q])
    def solve2(self, x):
        v, p = self.vectorview.get_parts(x)
        # v, p = x[:self.nv], x[self.nv:]
        w = self._solve_v(v)
        q = self.SP.solve(b=p-self.AS.B@w)
        w += self.AS.B.T@q
        return np.hstack([w, q])
    def solve3(self, x):
        v, p, lam = self.vectorview.get_parts(x)
        w = self._solve_v(v)
        self.timer.add("solve_v")
        q = self.SP.solve(b=p-self.AS.B@w)
        self.timer.add("solve_p")
        # lam = (lam-self.AS.C@q)/self.c
        lam -= self.AS.C@q
        q += self.SP.solve(b=self.AS.C.T@lam)
        self.timer.add("solve_p")
        w += self.AS.B.T@q
        return np.hstack([w, q, lam])

