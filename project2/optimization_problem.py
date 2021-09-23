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
        Z = np.array(self.func([X,Y]))
        N = 24
        big = linspace(20.1,1250, num = 5)
        small = linspace(0,3, num = 7)
        medium = linspace(3.1,20, num = 10)
        levels = np.concatenate((small, big), axis = 0)
        cp = ax.contour(X, Y, Z, levels=levels, colors = 'black')

        plt.clabel(cp, inline=1, fontsize=10)
        plt.setp(cp.collections , linewidth=0.75)
        plt.plot(steps[:,0], steps[:,1], linestyle = 'dashed', marker = 'o',
                 markersize = 2, color = 'red')
        plt.xlim((interval[0][0], interval[0][len(interval[0])-1]))
        plt.ylim((interval[1][0], interval[1][len(interval[1])-1]))
        plt.title(title)
        plt.show()
        
    #def plot(self, interval, steps, title = None):
    
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

def booth(x):
    return (x[0] + 2 * x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

def beale(x):
    return (1.5 - x[0] + x[0] * x[1])**2 + (2.25 - x[0] + x[0] * x[1]**2)**2 + (2.625 - x[0] + x[0]*  x[1]**3)**2

def sphere(x):
    return x[1]**2 + x[0]**2

def rosenbrock(x):
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
    
def grad_rosenbrock(x):
    return [-400*(x[1]-x[0]**2)*x[0] - 2*(1-x[0]), 200*(x[1]-x[0]**2)]    


if __name__ == '__main__':
    # problem1 = OptimizationProblem(rosenbrock)
    # test1 = Test(problem1, name = "Rosenbrock")
    # problem2 = OptimizationProblem(cqp.chebyquad)
    # test2 = Test(problem2, name = "Chebyquad")
    # test1.test()
    # test2.test(num_points=4)
    # problem3 = OptimizationProblem(sphere)
    # test3 = Test(problem3, name = "Sphere")
    # test3.test()
    # problem4 = OptimizationProblem(beale)
    # test4 = Test(problem4, name = "Beale", methods = [BFGS(True, name = 'BFGS')])
    # test4.test()
    problem5 = OptimizationProblem(booth)
    test5 = Test(problem5, name = "Booth", methods = [BFGS(True, name = 'BFGS')])
    test5.test()
    
    
    
    # hes_problem = OptimizationProblem(rosenbrock, grad_rosenbrock)
    # method = BFGS(True, calc_hes=True)
    # _,_,hessians, default_hessians = method(hes_problem, [-10, 5])
    # hessians = np.array(hessians)
    # default_hessians = np.array(default_hessians)
    # norms = [np.linalg.norm(x) for x in (hessians - default_hessians)[1:]]
    # norms_d = [np.linalg.norm(x) for x in default_hessians]
    
    # loged =  np.log(norms)
    # plt.plot(loged)
    # plt.ylabel('Differance in norms')
    # plt.xlabel("k")
    # slope, intercept, r_value, p_value, std_err = stats.linregress(list(range(len(norms))),loged)
    # x = linspace(0,len(loged))
    # plt.plot(x,intercept + slope * x)