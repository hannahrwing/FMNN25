#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 08:53:36 2021

@author: nils
"""
import pandas as pd
from numpy import *
import numpy as np
from optimization_method import ClassicNewton, BFGS, DFP, GoodBroyden, SymmetricBroyden, BadBroyden
import scipy.optimize as so
import chebyquad_problem as cqp

def get_all_methods():
    return [ClassicNewton(True, name = 'Classic Newton'), GoodBroyden(True, name = 'Good Broyden'),
            BadBroyden(True, name = 'Bad Broyden'), SymmetricBroyden(True, name = 'Symmetric Broyden'),
            DFP(True, name = 'DFP'), BFGS(True, name = 'BFGS'),
            ClassicNewton(True, name = 'Classic Newton'),GoodBroyden(False, name = 'Good Broyden'),
           BadBroyden(False, name = 'Bad Broyden'), SymmetricBroyden(False, name = 'Symmetric Broyden'),
           DFP(False, name = 'DFP'), BFGS(False, name = 'BFGS')]

class Test():
    
    def __init__(self, problem, methods = None):
        self.problem = problem
        if methods == None:
            self.methods = get_all_methods()
             
    def test(self, num_points = 0):
        methods = self.methods
        df = pd.DataFrame({'Method' : [x.name for x in methods],
                           'Exact/Inexact' : ["Exact" if x.exact_line_search
                                              else "Inexact" for x in methods],
                           'Starting Points': None,
                           'Minima': None,
                           'Real Minima' : None})
        problem = self.problem
        
        if num_points != 0:
            x0 = linspace(0,1,num_points)
        else:
            x0 = [0.5, 3]
        
        for i in range(len(methods)):
            
            df['Minima'][i], steps = methods[i](problem, x0)
            if df['Minima'][i] is not None:
                df['Minima'][i] = np.round(df['Minima'][i],4)
            else:
                df['Minima'][i] = None
            df['Real Minima'][i] = df['Real Minima'][i] = np.round(so.fmin_bfgs(self.problem.func,x0, self.problem.gradient, disp=False),5)
            df['Starting Points'][i] = np.round(x0,2)
            
            if len(x0)==2:
                interval = [linspace(-0.7, 2, num=1000), linspace(-1.5, 4, num = 1000)]
                problem.plot(interval,steps, title = df['Method'][i] + " " + df['Exact/Inexact'][i])
        pd.options.display.max_columns = None
        pd.set_option("display.max_rows", None)
        pd.set_option("display.max_columns", None)
        df = df.fillna(value="Does not Converge")
        print('Chebyquad Function')
        display(df)