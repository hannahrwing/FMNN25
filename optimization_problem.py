# -*- coding: utf-8 -*-

from numpy import *
from matplotlib.pyplot import *
from optimization_method import ClassicNewton

class OptimizationProblem:
    
    def __init__(self, func, gradient = None):
        self.func = func
        self.gradient = gradient
    
    def plot(self):
        pass
    
    
    
def f(x):
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
    
def grad(x):
    return [-400*(x[1]-x[0]**2)*x[0] - 2*(1-x[0]), 200*(x[1]-x[0]**2)]    
    
if __name__ == '__main__':
    
    problem = OptimizationProblem(f, grad)
    method = ClassicNewton()
    x0 = [4,-800]
    sol = method(problem, x0)
    print(sol)