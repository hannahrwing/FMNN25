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
        t_normal = 15
        t_hot = 40
        t_cold = 5
        
        
        nx1 = np.shape(self.sols[0])[1]
        ny1 = np.shape(self.sols[0])[0]
        nx2 = np.shape(self.sols[1])[1]
        ny2 = np.shape(self.sols[1])[0]
        nx3 = np.shape(self.sols[2])[1]
        ny3 = np.shape(self.sols[2])[0]
        Nx = nx1 + nx2 + nx3
        Ny = ny2
        big = np.zeros((Ny+2, Nx+2))
        #insert solution
        big[1:1+self.sols[0].shape[0], 1:1+self.sols[0].shape[1]] += self.sols[0]
        big[1:1 + self.sols[1].shape[0], nx1+1: nx1+1+self.sols[1].shape[1]] += self.sols[1]
        big[nx1+1:nx1+1 + self.sols[2].shape[0], nx1+nx2+1 : nx1+nx2+1+self.sols[2].shape[1]] += self.sols[2]
        
        #insert boundry cons
        #room 1
        big[1:1+self.sols[0].shape[0],0] = t_hot #bottom left
        big[0,0:0+self.sols[0].shape[1]+1] = t_normal #bottom
        big[ny1 + 1,0:0+self.sols[0].shape[1]] = t_normal #top
        #room2 
        big[0,nx1+1: nx1+1+self.sols[1].shape[1]] = t_cold  #botoom
        big[0: ny1+1,nx1+nx2+1] = t_normal #bottom right
        big[ny1 + 1::, nx1] = t_normal #top left
        big[-1,nx1+1: nx1+1+self.sols[1].shape[1]] = t_hot
        #room3
        big[-1,nx1+nx2+1::] = t_normal #top
        big[ny1 + 2: ny1 + 2 + ny3,-1] = t_hot #right
        big[ny1 + 1, nx1 + nx2 + 1::] = t_normal #bottom
        
        X, Y = np.meshgrid(np.linspace(0,3,big.shape[1]), np.linspace(0,2,big.shape[0]))
        big[big == 0.0] = np.NaN
        fig = plt.figure(dpi=600)

        ax = fig.add_subplot(111)
        
        
        
        
        cp = ax.contourf(X,Y, big, cmap=cm.coolwarm)
        cbar = fig.colorbar(cp)
        cbar.ax.set_ylabel('Temperature [C]')
        
        plt.show()