# -*- coding: utf-8 -*-

from numpy import *
from matplotlib.pyplot import *
from optimization_method import ClassicNewton, BFGS
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


class OptimizationProblem:
    
    def __init__(self, func, gradient = None):
        self.func = func
        self.gradient = gradient
    
    def plot(self, interval, steps):
        
        mpl.rcParams['figure.dpi'] = 300
        fig, ax = plt.subplots(1,1)
        X, Y = np.meshgrid(interval[0], interval[1])
        X, Y
        Z = np.array(100*(Y-X**2)**2 + (1-X)**2)
        
        N = 24
        big = linspace(20.1,1250, num = 5)
        small = linspace(0,3, num = 7)
        medium = linspace(3.1,20, num = 10)
        levels = np.concatenate((small, big), axis = 0)
        Z = 100*(Y-X**2)**2 + (1-X)**2
        cp = ax.contour(X, Y, Z, levels=levels,
                        colors = 'black')

        plt.clabel(cp, inline=1, fontsize=10)
        plt.setp(cp.collections , linewidth=0.75)
        plt.plot(steps[:,0], steps[:,1], linestyle = 'dashed', marker = 'o',
                 markersize = 2, color = 'red')
        plt.xlim((interval[0][0], interval[0][len(interval[0])-1]))
        plt.ylim((interval[1][0], interval[1][len(interval[1])-1]))
        plt.show()
    
    
    
def f(x):
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
    
def grad(x):
    return [-400*(x[1]-x[0]**2)*x[0] - 2*(1-x[0]), 200*(x[1]-x[0]**2)]    
    
if __name__ == '__main__':
    
    # problem = OptimizationProblem(f, grad)
    # method = ClassicNewton()
    # x0 = [0.5,3]
    # sol, steps = method(problem, x0)
    # print(sol)
    # interval = [linspace(-0.7, 2, num=1000), linspace(-1.5, 4, num = 1000)]
    # problem.plot(interval,steps)
    interval = [linspace(-0.7, 2, num=1000), linspace(-1.5, 4, num = 1000)]
    problem = OptimizationProblem(f, grad)
    method = BFGS(False)
    x0 = [0.5,3]
    sol, steps = method(problem, x0)
    print(steps)
    problem.plot(interval,steps)
