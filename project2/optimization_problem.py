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
import time


class OptimizationProblem:
    
    def __init__(self, func, gradient = None):
        self.func = func
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
    
    
    
def rosenbrock(x):
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2
    
def grad_rosenbrock(x):
    return [-400*(x[1]-x[0]**2)*x[0] - 2*(1-x[0]), 200*(x[1]-x[0]**2)]    
    
if __name__ == '__main__':
    plt.close('all')
    
    method_names = ['Good Broyden', 'Bad Broyden', 'Symmetric Broyden',
                    'DFP', 'BFGS']
    methods_exact = [GoodBroyden(True), BadBroyden(True),
                     SymmetricBroyden(True), DFP(True), BFGS(True)]
    methods_inexact = [GoodBroyden(False), BadBroyden(False),
                     SymmetricBroyden(False), DFP(False), BFGS(False)]
    df_rosenbrock = pd.DataFrame({'Method' : method_names,
                                  'Function' : ['Rosenbrock' for i in range(len(method_names))] ,
                                  'Starting Point': None,
                                  'Minima': None,
                                  'Real Minima' : None})
    problem = OptimizationProblem(rosenbrock, grad_rosenbrock)
    interval = [linspace(-0.7, 2, num=1000), linspace(-1.5, 4, num = 1000)]
    
    
    for i in range(len(methods_exact)):
        x0 = 10 * np.random.random(2) - 5
        method = methods_exact[i]
        sol, steps = method(problem, x0)
        df_rosenbrock['Minima'][i] = np.round(sol,3)
        df_rosenbrock['Real Minima'][i] = np.round(so.fmin_bfgs(rosenbrock,x0, grad_rosenbrock),5)
        df_rosenbrock['Starting Point'][i] = np.round(x0,2)
        problem.plot(interval,steps, title = df_rosenbrock['Method'][i])
    pd.options.display.max_columns = None
    pd.set_option("display.max_rows", None)
    print('Rosenbrock Function')
    display(df_rosenbrock)
    