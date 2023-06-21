# -*- coding: utf-8  -*-
"""

"""

import numpy as np
import sympy


#=================================================================#
class AnalyticalFunction():
    """
    computes numpy vectorized functions for the function and its dericatives up to two
    for a given expression, derivatives computed with sympy
    """
    def __repr__(self):
        return f"expr={str(self.expr)}"
        return f"dim={self.dim} expr={str(self.expr)}"
    def __call__(self, *x):
        return self.fct(*x)
    def __init__(self, expr, dim=3):
        if expr.find('x0') == -1 and expr.find('x1') == -1 and expr.find('x2') == -1:
            expr = expr.replace('x', 'x0')
            expr = expr.replace('y', 'x1')
            expr = expr.replace('z', 'x2')
        if dim==1 and expr.find('x0') == -1:
            expr = expr.replace('x', 'x0')
        if expr.find('t') == -1: self.has_time = False
        else: self.has_time = True
        self.dim, self.expr = dim, expr
        symbc = ""
        for i in range(dim): symbc += f"x{i},"
        if self.has_time: symbc += "t"
        else: symbc = symbc[:-1]
        s = sympy.symbols(symbc)
        # print(f"{expr=} {symbc=} {s=}")
        self.fct = np.vectorize(sympy.lambdify(symbc,expr))
        self.fct_x = []
        self.fct_xx = []
        if self.has_time:
            self.fct_t = np.vectorize(sympy.lambdify(symbc, sympy.diff(expr, s[-1])), otypes=[float])
        for i in range(dim):
            # self.fct_xxx = np.vectorize(sympy.lambdify(symbc, sympy.diff(expr, s[0], 3)),otypes=[float])
            # self.fct_xxxx = np.vectorize(sympy.lambdify(symbc, sympy.diff(expr, s[0], 4)),otypes=[float])
            if dim==1: fx = sympy.diff(expr, s)
            else: fx = sympy.diff(expr, s[i])
            self.fct_x.append(np.vectorize(sympy.lambdify(symbc, fx), otypes=[float]))
            self.fct_xx.append([])
            for j in range(dim):
                if dim == 1: fxx = sympy.diff(fx, s)
                else: fxx = sympy.diff(fx, s[j])
                self.fct_xx[i].append(np.vectorize(sympy.lambdify(symbc, fxx),otypes=[float]))
    def t(self, *x):
        return self.fct_t(*x)
    def d(self, i, *x):
        return self.fct_x[i](*x)
    def x(self, *x):
        return self.fct_x[0](*x)
    def y(self, *x):
        return self.fct_x[1](*x)
    def z(self, *x):
        return self.fct_x[2](*x)
    def dd(self, i, j, *x):
        return self.fct_xx[i][j](*x)
    def xxxx(self, *x):
        return self.fct_xxxx(*x)
    def xx(self, *x):
        return self.fct_xx[0][0](*x)
    def xxx(self, *x):
        return self.fct_xxx(*x)
    def xy(self, *x):
        return self.fct_xx[0][1](*x)
    def xz(self, *x):
        return self.fct_xx[0][2](*x)
    def yy(self, *x):
        return self.fct_xx[1][1](*x)
    def yx(self, *x):
        return self.fct_xx[1][0](*x)
    def yz(self, *x):
        return self.fct_xx[1][2](*x)
    def zz(self, *x):
        return self.fct_xx[2][2](*x)
    def zx(self, *x):
        return self.fct_xx[2][0](*x)
    def zy(self, *x):
        return self.fct_xx[2][1](*x)

#=================================================================#
def analyticalSolution(function, dim, ncomp=1, random=True):
    """
    defines some analytical functions to be used in validation

    returns analytical function (if ncomp==1) or list of analytical functions (if ncomp>1)

    parameters:
        function: name of function
        ncomp: size of list
        random: use random coefficients
    """
    solexact = []
    def _p(n):
        if random:
            p = (4 * np.random.rand(n) - 2) / 3
        else:
            p = [1.1 * (n - d) for d in range(n)]
        return p
    vars = ['x', 'y', 'z']
    p = _p(ncomp * 2*dim*dim)
    for i in range(ncomp):
        # print(f"{p=}")
        fct = '{:3.1f}'.format(p.pop())
        if function == 'Constant': pass
        elif function == 'Linear' or function == 'Quadratic':
            for d in range(dim): fct += "{:+3.1f}*{:1s}".format(p.pop(), vars[d])
            if function == 'Quadratic':
                for d in range(dim): fct += "{:+3.1f}*{:1s}**2".format(p.pop(), vars[d])
        elif function == 'Sinus':
            for d in range(dim): fct += "{:+3.1f}*sin({:1s})".format(p.pop(), vars[d])
        else:
            if ncomp==1: fct = function
            else: fct = function[i]
        solexact.append(AnalyticalFunction(expr=fct))
    if ncomp==1: return solexact[0]
    return solexact


# ------------------------------------------------------------------- #
if __name__ == '__main__':
    def test1D():
        u = AnalyticalFunction(dim=1, expr='x*x')
        print("u(2)", u(2))
        x = np.meshgrid(np.linspace(0, 2, 3))
        print("x", x, "\nu=", u.expr, "\nu(x)", u(x), "\nu.x(x)", u.x(x), "\nu.xx(x)", u.xx(x))
    def test2D():
        u = AnalyticalFunction(dim=2, expr='x*x*y + y*y')
        print("u(2,1)", u(2,1))
        x = np.meshgrid(np.linspace(0, 2, 3),np.linspace(0, 1, 2))
        print("x", x, "\nu=", u.expr, "\nu(x)", u(*x), "\nu.x(x)", u.x(*x), "\nu.xx(x)", u.xx(*x))

    def test2D_dyn():
        u = AnalyticalFunction(dim=2, expr='t + x*x*y + y*y')
        u = AnalyticalFunction(dim=2, expr='sin(pi*x+1)*cos(t)')
        print(f"{u(2,1,0)=}")
        print(f"{u(2,1,100)=}")
        print(f"{u.t(2,1,100)=}")

    test2D_dyn()



