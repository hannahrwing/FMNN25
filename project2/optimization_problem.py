# -*- coding: utf-8 -*-

from numpy import *
from matplotlib.pyplot import *
from optimization_method import ClassicNewton, BFGS, DFP, GoodBroyden, SymmetricBroyden, BadBroyden
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import chebyquad_problem as cqp
import pandas as pd
import scipy.optimize as so
import random 
from tests import Test
from results import Results
from scipy import stats

class OptimizationProblem:
    
    def __init__(self, func, gradient = None):
        self.func = func
        if gradient is None:
            self.gradient = self.default_grad
        else:
            self.gradient = gradient
    
    def plot(self, interval, steps, title = None):
        
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
        cp = ax.contour(X, Y, Z, levels=levels, colors = 'black')

        plt.clabel(cp, inline=1, fontsize=10)
        plt.setp(cp.collections , linewidth=0.75)
        plt.plot(steps[:,0], steps[:,1], linestyle = 'dashed', marker = 'o',
                 markersize = 2, color = 'red')
        plt.xlim((interval[0][0], interval[0][len(interval[0])-1]))
        plt.ylim((interval[1][0], interval[1][len(interval[1])-1]))
        plt.title(title)
        plt.show()
    
    def default_grad(self, x):
        h = 1e-5
        grad = empty(len(x))
        
        for i in range(len(x)):
            x_plus =  x.copy() 
            x_min = x.copy()
            x_plus[i] += h
            x_min[i] -= h 
            grad[i] = (self.func(x_plus) - self.func(x_min))/(2*h)
            
        return grad
    

def rosenbrock(x):
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
    
def grad_rosenbrock(x):
    return [-400*(x[1]-x[0]**2)*x[0] - 2*(1-x[0]), 200*(x[1]-x[0]**2)]    


if __name__ == '__main__':
    problem1 = OptimizationProblem(rosenbrock)
    result1 = Results(problem1, name = "Rosenbrock")
    result1.show_plots_and_tables()
    
    problem2 = OptimizationProblem(cqp.chebyquad)
    result2 = Results(problem2, name = "Chebyquad")
    result2.show_plots_and_tables(num_points=4)

    hes_problem = OptimizationProblem(rosenbrock, grad_rosenbrock) 
    result3 = Results(hes_problem, name = "Hessian diff")
    result3.hessian_diff()
    
    