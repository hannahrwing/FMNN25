#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 08:53:36 2021

@author: nils
"""
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

def get_all_methods():
    return [ClassicNewton(True), GoodBroyden(True),
            BadBroyden(True), SymmetricBroyden(True),
            DFP(True), BFGS(True),
            ClassicNewton(False),GoodBroyden(False),
           BadBroyden(False), SymmetricBroyden(False),
           DFP(False), BFGS(False)]

class Results():
    
    def __init__(self, problem, methods = None, name = ""):
        self.problem = problem
        self.name = name
        if methods == None:
            self.methods = get_all_methods()
        
             
    def show_plots_and_tables(self, num_points = 2):
        methods = self.methods
        df = pd.DataFrame({'Method' : [type(x).__name__ for x in methods],
                           'Exact/Inexact' : ["Exact" if x.exact_line_search
                                              else "Inexact" for x in methods],
                           'Starting Points': None,
                           'Minima': None,
                           'Real Minima' : None})
        problem = self.problem
        
        if num_points != 2:
            x0 = linspace(0,1,num_points)
        else:
            x0 = [-0.5, 3]
        
        for i in range(len(methods)):
            
            df['Minima'][i] = methods[i](problem, x0)
            if df['Minima'][i] is not None:
                df['Minima'][i] = np.round(df['Minima'][i],4)
            else:
                df['Minima'][i] = None
            df['Real Minima'][i] = df['Real Minima'][i] = np.round(so.fmin_bfgs(self.problem.func,x0, self.problem.gradient, disp=False),5)
            df['Starting Points'][i] = np.round(x0,2)
            #steps = list(methods[i].generate(self.problem,x0))
            steps = np.array(x0)
            for x, H in methods[i].generate(self.problem, x0):
                steps = np.vstack((steps, x))
            #steps = list(methods[i].generate(self.problem, x0))
            if len(x0)==2:
                interval = [linspace(-0.7, 2, num=1000), linspace(-1.5, 4, num = 1000)]
                problem.plot(interval,steps, title = df['Method'][i] + " " + df['Exact/Inexact'][i])
        pd.options.display.max_columns = None
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        df = df.fillna(value="Does not Converge")
        print(self.name)
        display(df)
        
    def hessian_diff(self):
        method = BFGS(True, calc_hes=True)
        _,_,hessians, d_hessians = method(self.problem, [-10, 5])
        hessians = np.array(hessians)
        d_hessians = np.array(d_hessians)
        norms = [np.linalg.norm(x) for x in (hessians - d_hessians)[1:]]
        norms_d = [np.linalg.norm(x) for x in d_hessians]
        loged =  np.log(norms)
        plt.plot(loged)
        plt.ylabel('Differance in norms')
        plt.xlabel("k")
        slope, intercept, r_value, p_value, std_err = stats.linregress(list(range(len(norms))),loged)
        x = linspace(0,len(loged))
        plt.plot(x,intercept + slope * x)
        
    def hessian_diff_2(self):
        matplotlib.rcParams['mathtext.fontset'] = 'stix'
        matplotlib.rcParams['font.family'] = 'STIXGeneral'
        
        
        x0 = [-10, 5]
        method = BFGS(True)
        hessians = []
        d_hessians = []
        for x,h in method.generate(self.problem, x0):
           hessians.append(h)
           d_hessians.append(method.default_hessian(x, self.problem.func))
           
        hessians = np.array(hessians)
        d_hessians = np.array(d_hessians)
        norms = [np.linalg.norm(x) for x in (hessians - d_hessians)[1:]]
        norms_d = [np.linalg.norm(x) for x in d_hessians]
        loged =  np.log(norms)
        plt.plot(loged)
        plt.ylabel('$ln(|\Delta H|_{F}$)', fontsize = 18)
        plt.xlabel("k")
        slope, intercept, r_value, p_value, std_err = stats.linregress(list(range(len(norms))),loged)
        x = linspace(0,len(loged))
        plt.plot(x,intercept + slope * x)
        