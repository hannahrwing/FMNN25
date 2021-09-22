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

def test_rosebrock():
    methods = [GoodBroyden(True, name = 'Good Broyden'), BadBroyden(True, name = 'Bad Broyden'),
                     SymmetricBroyden(True, name = 'Symmetric Broyden'), DFP(True, name = 'DFP'),
                     BFGS(True, name = 'BFGS'), GoodBroyden(False, name = 'Good Broyden'),
                     BadBroyden(False, name = 'Bad Broyden'), SymmetricBroyden(False, name = 'Symmetric Broyden'),
                     DFP(False, name = 'DFP'), BFGS(False, name = 'BFGS')]
    df_rosenbrock = pd.DataFrame({'Method' : [x.name for x in methods],
                                  'Exact/Inexact' : ["Exact" if x.exact_line_search
                                                     else "Inexact" for x in methods],
                                  'Starting Point': None,
                                  'Minima': None,
                                  'Real Minima' : None})
    problem = OptimizationProblem(rosenbrock, grad_rosenbrock)
    interval = [linspace(-0.7, 2, num=1000), linspace(-1.5, 4, num = 1000)]
    
    for i in range(len(methods)):
        x0 = 10 * np.random.random(2) - 5
        #x0 = [0.5, 3]
        df_rosenbrock['Minima'][i], steps = methods[i](problem, x0)
        if df_rosenbrock['Minima'][i] is not None:
            df_rosenbrock['Minima'][i] = np.round(df_rosenbrock['Minima'][i],2)
        else:
            df_rosenbrock['Minima'][i] = None
        df_rosenbrock['Real Minima'][i] = np.round(so.fmin_bfgs(rosenbrock,x0, grad_rosenbrock, disp=False),5)
        df_rosenbrock['Starting Point'][i] = np.round(x0,2)
        problem.plot(interval,steps, title = df_rosenbrock['Method'][i] + " " + df_rosenbrock['Exact/Inexact'][i])
    pd.options.display.max_columns = None
    pd.set_option("display.max_rows", None)
    df_rosenbrock = df_rosenbrock.fillna(value="Does not Converge")
    print('Rosenbrock Function')
    display(df_rosenbrock)

if __name__ == '__main__':
    test_rosebrock()