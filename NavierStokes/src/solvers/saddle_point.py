import numpy as np
import scipy.sparse as sparse
from . import linalg

#=================================================================#
class SaddlePointSystem():
    """
    A -B.T
    B  0
     or
    A -B.T 0
    B  0   M^T
    0  M   0
    """
    def __repr__(self):
        string =  f" {self.singleA=} {self.A.shape=} {self.B.shape=}"
        if hasattr(self, 'M'): string += f"{self.M.shape=}"
        return string
    def __init__(self, stack_storage, nv_single, A, B, M=None, singleA=True, ncomp=None):
        self.stack_storage = stack_storage
        self.nv_single = nv_single
        if singleA and not ncomp:
            raise ValueError(f"{singleA=} needs 'ncomp'")
        if singleA:
            self.na = ncomp*A.shape[0]
        else:
            self.na = A.shape[0]
        self.singleA, self.ncomp = singleA, ncomp
        self.A, self.B = A, B
        self.nb = B.shape[0]
        self.nall = self.na + self.nb
        if M is not None:
            self.M = M
            self.nm = M.shape[0]
            self.nall += self.nm
        self.constr = hasattr(self, 'M')
        if self.singleA:
            self.matvec = self.matvec3_singleA if self.constr else self.matvec2_singleA
        else:
            self.matvec = self.matvec3 if self.constr else self.matvec2
    def _dot_A(self, b):
        if self.singleA:
            w = np.empty_like(b)
            if self.stack_storage:
                n = self.A.shape[0]
                for i in range(self.ncomp): w[i*n:(i+1)*n] = self.A.dot(b[i*n:(i+1)*n])
            else:
                for i in range(self.ncomp): w[i::self.ncomp] = self.A.dot(b[i::self.ncomp])
            return w
        return self.A.dot(b)
    def copy(self):
        M = None
        if hasattr(self, 'M'): M = self.M.copy()
        return SaddlePointSystem(self.A.copy(), self.B.copy(), M=M, singleA=self.singleA, ncomp=self.ncomp)
    def scale_A(self):
        assert not self.constr
        DA = self.A.diagonal()
        assert np.all(DA>0)
        if self.singleA:
            AD = linalg.diagmatrix2systemdiagmatrix(1/DA, self.ncomp, self.stack_storage)
        else:
            AD = sparse.diags(1/DA, offsets=(0), shape=self.A.shape)
        DS = (self.B@AD@self.B.T).diagonal()
        # print(f"scale_A {DA.max()/DA.min()=:8.2f} {DS.max()/DS.min()=:8.2f}")
        assert np.all(DS>0)
        nb = self.B.shape[0]
        self.vs = sparse.diags(np.power(AD.diagonal(), 0.5), offsets=(0), shape=AD.shape)
        self.ps = sparse.diags(np.power(DS, -0.5), offsets=(0), shape=(nb,nb))
        # print(f"{self.vs.data=}\n{self.ps.data=}")
        if self.singleA:
            # vss = sparse.dia_array((np.power(DA, -0.5), (0)), shape=self.A.shape)
            vss = sparse.diags(np.power(DA, -0.5), offsets=(0), shape=self.A.shape)
            self.A = vss @ self.A @ vss
        else:
            self.A = self.vs@self.A@self.vs
        if not np.allclose(self.A.diagonal(), np.ones(self.A.shape[0])):
            raise ValueError(f"not ones on diagona\n{self.A.diagonal()=}")
        self.B = self.ps@self.B@self.vs
    def scale_vec(self, b):
        # print(f"scale_rhs")
        bv, bp = b[:self.na], b[self.na:]
        b[:self.na] = self.vs@bv
        b[self.na:] = self.ps@bp
    def dot(self, x):
        if hasattr(self, "M"): return self.matvec3(x)
        return self.matvec2(x)
    def matvec3(self, x):
        v, p, lam = x[:self.na], x[self.na:self.na+self.nb], x[self.na+self.nb:]
        w = self.A.dot(v) - self.B.T.dot(p)
        q = self.B.dot(v)+ self.M.T.dot(lam)
        return np.hstack([w, q, self.M.dot(p)])
    def matvec2(self, x):
        v, p = x[:self.na], x[self.na:]
        # w = self.A.dot(v) - self.B.T.dot(p)
        w = self._dot_A(v) - self.B.T.dot(p)
        q = self.B.dot(v)
        return np.hstack([w, q])
    def matvec3_singleA(self, x):
        v, p, lam = x[:self.na], x[self.na:self.na+self.nb], x[self.na+self.nb:]
        w = - self.B.T.dot(p)
        if self.stack_storage:
            n = self.nv_single
            for icomp in range(self.ncomp): w[icomp*n:(icomp+1)*n] += self.A.dot(v[icomp*n:(icomp+1)*n])
        else:
            for icomp in range(self.ncomp): w[icomp::self.ncomp] += self.A.dot(v[icomp::self.ncomp])
        q = self.B.dot(v)+ self.M.T.dot(lam)
        return np.hstack([w, q, self.M.dot(p)])
    def matvec2_singleA(self, x):
        v, p = x[:self.na], x[self.na:]
        w = - self.B.T.dot(p)
        if self.stack_storage:
            n = self.nv_single
            for icomp in range(self.ncomp): w[icomp*n:(icomp+1)*n] += self.A.dot(v[icomp*n:(icomp+1)*n])
        else:
            for icomp in range(self.ncomp): w[icomp::self.ncomp] += self.A.dot(v[icomp::self.ncomp])
        q = self.B.dot(v)
        return np.hstack([w, q])
    def to_single_matrix(self):
        nullP = sparse.dia_matrix((np.zeros(self.nb), 0), shape=(self.nb, self.nb))
        raise ValueError(f"{self.singleA=}")
        if self.singleA:
            A = matrix2systemdiagonal(self.A, self.ncomp)
        else:
            A = self.A
        A1 = sparse.hstack([A, -self.B.T])
        A2 = sparse.hstack([self.B, nullP])
        Aall = sparse.vstack([A1, A2])
        if not hasattr(self, 'M'):
            return Aall.tocsr()
        nullV = sparse.coo_matrix((1, self.na)).tocsr()
        ML = sparse.hstack([nullV, self.M])
        Abig = sparse.hstack([Aall, ML.T])
        nullL = sparse.dia_matrix((np.zeros(1), 0), shape=(1, 1))
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
    def _get_schur_of_diag(self, ret_diag=False):
        if not np.allclose(self.AS.A.diagonal(), np.ones(self.AS.A.shape[0])):
            raise ValueError(f"{self.AS.A.data=}")
        if self.AS.singleA:
            AD = linalg.diagmatrix2systemdiagmatrix(1/self.AS.A.diagonal(), self.AS.ncomp, self.stack_storage)
        else:
            AD = sparse.diags(1 / self.AS.A.diagonal(), offsets=(0), shape=self.AS.A.shape)
        if ret_diag: return self.AS.B @ AD @ self.AS.B.T, AD
        return self.AS.B @ AD @ self.AS.B.T
    def __init__(self, AS, stack_storage, **kwargs):
        self.stack_storage = stack_storage
        if not isinstance(AS, SaddlePointSystem):
            raise ValueError(f"*** resuired arguments: AS (SaddlePointSystem)")
        self.AS = AS
        self.nv_single = AS.B.shape[1]//AS.ncomp
        constr = hasattr(AS, 'M')
        self.nv = self.AS.na
        self.nvp = self.AS.na + AS.nb
        self.nall = self.nvp
        if constr: self.nall += AS.nm
        # print(f"{AS=} {self.nv=} {self.nvp=} {self.nall=} {solver_v=} {solver_p=}")
        method = kwargs.pop('method', None)
        self.method = method
        if method is not None:
            if method == 'diag':
                self.matvecprec = self.pmatvec3_diag if constr else self.pmatvec2_diag
            elif method == 'triup':
                self.matvecprec = self.pmatvec3_triup if constr else self.pmatvec2_triup
            elif method == 'tridown':
                self.matvecprec = self.pmatvec3_tridown if constr else self.pmatvec2_tridown
            elif method == 'full':
                self.matvecprec = self.pmatvec3_full if constr else self.pmatvec2_full
            elif method[:3] == 'hss':
                ms = method.split('_')
                if len(ms) != 2: raise ValueError(f"*** needs 'hass_alpha'")
                self.alpha = float(ms[1])
                # solver_p['type'] = f"diag_{self.alpha**2}"
                # solver_p['method'] = f"pyamg"
                self.matvecprec = self.pmatvec3_hss if constr else self.pmatvec2_hss
            else:
                raise ValueError(f"*** unknwon {method=}\npossible values: 'diag', 'triup', 'tridown', 'full', hss_alpha")
        solver_p = kwargs.pop('solver_p', None)
        solver_v = kwargs.pop('solver_v', None)
        if solver_v is None:
            solver_v = {'method': 'pyamg', 'maxiter': 1, 'disp':0}
        # assert solver_p == None
        if solver_p is None:
            solver_p = {'method': 'pyamg', 'maxiter': 1, 'disp':0, 'symmetric':True}
        solver_v['counter'] = '\tV '
        solver_v['matrix'] = self.AS.A
        solver_p['counter'] = '\tP '
        solver_p['matrix'] = self._get_schur_of_diag()
        self.solver_v, self.solver_p = solver_v, solver_p
        # solver_p['matrix'] = self.AS.B @ self.AS.B.T
        # print(f"{solver_p['matrix'].shape=} {AD.shape=}")
        self.SP = linalg.getLinearSolver(**solver_p)
        self.SV = linalg.getLinearSolver(**solver_v)
        if len(kwargs.keys()): raise ValueError(f"*** unused arguments {kwargs=}")
    def update(self, A):
        self.AS = A
        # self.solver_v['matrix'] = self.AS.A
        # self.solver_p['matrix'] = self._get_schur_of_diag()
        # self.SP = linalg.getLinearSolver(**self.solver_p)
        # self.SV = linalg.getLinearSolver(**self.solver_v)
        self.SV.update(self.AS.A)
        # self.SP.update(self._get_schur_of_diag())
    def _solve_v(self,b):
        if self.AS.singleA:
            w = np.empty_like(b)
            if self.stack_storage:
                n = self.nv_single
                for i in range(self.AS.ncomp): w[i*n: (i+1)*n] = self.SV.solve(b=b[i*n: (i+1)*n])
            else:
                for i in range(self.AS.ncomp): w[i::self.AS.ncomp] = self.SV.solve(b=b[i::self.AS.ncomp])
            return w
        return self.SV.solve(b=b)
    def schurmatvec(self, x):
        v = self.AS.B.T.dot(x)
        v2 = self._solve_v(v)
        return self.AS.B.dot(v2)
    def pmatvec2_diag(self, x):
        v, p = x[:self.nv], x[self.nv:]
        w = self._solve_v(v)
        q = self.SP.solve(b=p)
        return np.hstack([w, q])
    def pmatvec3_diag(self, x):
        v, p, lam = x[:self.nv], x[self.nv:self.nvp], x[self.nvp:]
        w = self._solve_v(v)
        q = self.SP.solve(b=p)
        mu = self.MP.solve(lam)
        return np.hstack([w, q, mu])
    def pmatvec2_triup(self, x):
        v, p = x[:self.nv], x[self.nv:]
        q = self.SP.solve(b=p)
        w = self._solve_v(v+self.AS.B.T.dot(q))
        return np.hstack([w, q])
    def pmatvec2_tridown(self, x):
        v, p = x[:self.nv], x[self.nv:]
        w = self._solve_v(v)
        q = self.SP.solve(b=p-self.AS.B.dot(w))
        return np.hstack([w, q])
    def pmatvec2_full(self, x):
        v, p = x[:self.nv], x[self.nv:]
        w = self._solve_v(v)
        q = self.SP.solve(b=p-self.AS.B.dot(w))
        h = self.AS.B.T.dot(q)
        w += self._solve_v(h)
        return np.hstack([w, q])
    def pmatvec2_hss(self, x):
        v, p = x[:self.nv], x[self.nv:]
        q = self.SP.solve(b=p-1/self.alpha*self.AS.B.dot(v))
        w = self._solve_v(1/self.alpha*v + self.AS.B.T.dot(q))
        return np.hstack([w, q])
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
    def __init__(self, AS, stack_storage, **kwargs):
        self.alpha = kwargs.pop('alpha',10)
        super().__init__(AS, stack_storage, **kwargs)
    def solve(self, x):
        v, p = x[:self.nv], x[self.nv:]
        w = self.SV.solve(b=v)/self.alpha
        q = self.SP.solve(b=self.alpha*p-self.AS.B.dot(w))
        h = self.AS.B.T.dot(q)
        w += self.SV.solve(h)/self.alpha
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
        return s
    def __init__(self, AS, stack_storage, **kwargs):
        super().__init__(AS, stack_storage, **kwargs)
        constr = hasattr(AS, 'M')
        assert not constr
    def solve2(self, x):
        v, p = x[:self.nv], x[self.nv:]
        q = self.SP.solve(b=p)
        w = self._solve_v(b=v+self.AS.B.T.dot(q))
        return np.hstack([w, q])
    def solve(self, x):
        v, p = x[:self.nv], x[self.nv:]
        w = self._solve_v(v)
        q = self.SP.solve(b=p-self.AS.B@w)
        w += self.AS.B.T@q
        return np.hstack([w, q])
