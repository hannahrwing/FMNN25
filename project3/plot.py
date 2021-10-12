#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 09:36:29 2021

@author: nils
"""
import numpy as np
import scipy.linalg as l
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator
from matplotlib import cm

class Plotter():
    def __init__(self, sols):
        self.sols = sols
        
        
    def __call__(self):
        
        nx1 = np.shape(self.sols[0])[1]
        ny1 = np.shape(self.sols[0])[0]
        nx2 = np.shape(self.sols[1])[1]
        ny2 = np.shape(self.sols[1])[0]
        nx3 = np.shape(self.sols[2])[1]
        ny3 = np.shape(self.sols[2])[0]
        Nx = nx1 + nx2 + nx3
        Ny = ny2
        big = np.zeros((Ny, Nx))
        
        big[0:0+self.sols[0].shape[0], 0:0+self.sols[0].shape[1]] += self.sols[0]
        big[0:0 + self.sols[1].shape[0], nx1 : nx1+self.sols[1].shape[1]] += self.sols[1]
        big[nx1:nx1 + self.sols[2].shape[0], nx1+nx2 : nx1+nx2+self.sols[2].shape[1]] += self.sols[2]
        
        
        X, Y = np.meshgrid(np.linspace(0,3,big.shape[1]), np.linspace(0,2,big.shape[0]))
        big[big == 0.0] = np.NaN
        fig = plt.figure(dpi=600)

        ax = fig.add_subplot(111)
        
        
        
       
        cp = ax.contourf(X,Y, big)
        cbar = fig.colorbar(cp)
        cbar.ax.set_ylabel('verbosity coefficient')
        
        plt.show()