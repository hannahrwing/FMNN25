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
        uvec = linspace(self.grid[2], self.grid[-3], 1000)
        sol = zeros((len(uvec), 2))
        
        for i in range(len(uvec)):
            sol[i,:] = self.blossom(uvec[i])
    
        return sol
    
    def blossom(self,u):
        index = self.hot_interval(u)
        d = self.get_control_points(index)
        dA = self.interpolation(d[0], d[1], u, index-2, index+1)
        dB = self.interpolation(d[1], d[2], u, index-1, index+2)
        dC = self.interpolation(d[2], d[3], u, index, index+3)
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
            if (u == self.grid[2]):
                return 2
            if (u <= self.grid[i]):
                return i-1

            
    def get_control_points(self,index):
        return self.control_points[index-2:index+2]
    
    def plot(self, sol, deboor = False):
        x = sol[:,0]
        y = sol[:,1]
        plot(x, y)
        
        if(deboor):
            plot(self.control_points[:,0], self.control_points[:,1], linestyle = 'dashed' )
            
            
        
        
control_points = array([[-1,-1],[2,-1],[3,-2],[3,1],[6,3],[1,2],[-1,2],[-2,0]])

m = 10
grid = zeros(m)
grid[3:-3] = linspace(0,1,m-6)
grid[0:3] = -0.1
grid[-3:m] = 1.1

spline = cubic_spline(grid, control_points)
sol = spline()
spline.plot(sol, True)
        
        
#vectorize?
#blossom recursive?
#private help methods?
#inheritance?
#plot method parameters
        
    
    
    