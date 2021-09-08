# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 10:18:07 2021

@author: willi
"""
from numpy import *
from matplotlib.pyplot import *


#hejhej
class cubic_spline:
    def __init__(self, grid, control_points):
        self.grid = grid
        self.control_points = control_points
        
    def __call__(self):
        pass
        
    
    def blossom(self,u):
        #check if in interval
        index = self.hot_interval(u)
        d = self.get_control_points(index)
        dA = self.interpolation(d[0], d[1], u, index-2, index+1)
        dB = self.interpolation(d[1], d[2], u, index-1, index+2)
        dC = self.interpolation(d[2], d[3], u, index, index+2)
        dAB = self.interpolation(dA, dB, u, index-1, index+1)
        dBC = self.interpolation(dB, dC, u, index, index+2)
        return self.interpolation(dAB, dBC, u, index, index+1)
        
        
    def interpolation(self,d1,d2,u,leftmost,rightmost):
        alpha =  self.alpha(u,leftmost,rightmost)
        
        return alpha*d1 + (1-alpha)*d2
        
    def alpha(self,u,leftmost,rightmost):
        if (leftmost == rightmost):
            return 0
        return (self.grid[rightmost] - u)/ (self.grid[rightmost]-self.grid[leftmost])
    
    #returns index
    def hot_interval(self,u):
        for i in range(len(self.grid)):
            if (u < self.grid[i]):
                return i-1
            
    def get_control_points(self,index):
        return self.control_points[index-2:index+2]
    
    
    