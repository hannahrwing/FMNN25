# -*- coding: utf-8 -*-

from numpy import *
from matplotlib.pyplot import *
from scipy.optimize import minimize

class OptimizationMethod:
        
    def __init__(self, exact_line_search = True):
        self.exact_line_search = exact_line_search
    
    def __call__(self, problem, x0):
        H = self.hessian(x0, problem.func)
        x,H = self.step(H, x0, problem)
        x_old = x0
        tol = 1e-5
        while linalg.norm(problem.gradient(x)) > tol and linalg.norm(x-x_old) > tol:
            x_old = x
            x, H = self.step(H, x, problem)
        return x
    
    def step(self):
        raise NotImplementedError()
    
    
class Newton(OptimizationMethod):
    
    def step(self, H, x, problem):
        if problem.gradient == None:
            #calculate it
            pass
        else:
            s = -H @ problem.gradient(x)
        if self.exact_line_search:
            alpha = self.exact_search(x, s, problem.func)
        else:
            alpha = self.inexact_search(x, s, problem.func)
        x_new = x + alpha*s
        H_new = self.hessian(x, problem.func, H)
        return x_new, H_new
    
    def hessian(self):
        raise NotImplementedError()
        
    def exact_search(self, x, s, f):
        def func(alpha):
            return(f(x+alpha*s))
        return minimize(func, 0).x
    
    def inexact_search(self, x, s, f):
        pass


class ClassicNewton(Newton):
    
    def hessian(self, x, f, H_prev = None):
        n = len(x)
        G = zeros((n,n))
        h = 1e-3
        
        for i in range(n):
            for j in range(n):
                G[i,j] = (f(x + h*self._basisvec(n,(i,j),(1,1))) - f(x + h*self._basisvec(n,(i,j), (1,-1)))
                          - f(x + h*self._basisvec(n,(i,j),(-1,1))) + f(x + h*self._basisvec(n,(i,j),(-1,-1))))/(4*h**2)
        G = (G + G.T)/2
        return linalg.inv(G)
    
    def _basisvec(self, n, i, val):
        v = zeros(n)
        v[i[0]] += val[0]
        v[i[1]] += val[1]
        return v
