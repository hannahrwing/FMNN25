# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 10:18:07 2021

"""
from numpy import *
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.pyplot import *
from test_points_2 import control_points_2, grid_2


class CubicSpline:
    def __init__(self, grid, control_points):
        if not len(grid) == len(control_points) + 2:
            raise Exception("# of grid points are expected to be 2 more than # of control points")
        grid = sort(grid)
        self.grid = grid
        self.control_points = control_points

        
    def __call__(self):
        uvec = linspace(self.grid[2], self.grid[-3], 1000)
        sol = [self.blossom(u) for u in uvec]
    
        return sol
    
    def blossom(self,u):
        index = self._hot_interval(u)
        d = self.get_control_points(index)
        dA = self._interpolation(d[0], d[1], u, index-2, index+1)
        dB = self._interpolation(d[1], d[2], u, index-1, index+2)
        dC = self._interpolation(d[2], d[3], u, index, index+3)
        dAB = self._interpolation(dA, dB, u, index-1, index+1)
        dBC = self._interpolation(dB, dC, u, index, index+2)
        return self._interpolation(dAB, dBC, u, index, index+1)
        
        
    def _interpolation(self,d1,d2,u,leftmost,rightmost):
        alpha = self._alpha(u,leftmost,rightmost)
        
        return alpha*array(d1) + (1-alpha)*array(d2)
        
    def _alpha(self,u,leftmost,rightmost):
        if (leftmost == rightmost):
            return 0
        return (self.grid[rightmost] - u)/ (self.grid[rightmost]-self.grid[leftmost])
    
    #returns index
    def _hot_interval(self,u):
        for i in range(len(self.grid)):
            if (u == self.grid[2]):
                return 2
            elif (u <= self.grid[i]):
                return i-1

            
    def get_control_points(self,index):
        return self.control_points[index-2:index+2]
    
    def plot(self, sol, de_Boor = False):
        x = sol[:,0]
        y = sol[:,1]
        plot(x, y)
        
        if(de_Boor):
            de_Boor_x = [self.control_points[i][0] for i in range(len(self.control_points))]
            de_Boor_y = [self.control_points[i][1] for i in range(len(self.control_points))]             
            plot(de_Boor_x, de_Boor_y, linestyle = 'dashed' )
        plt.show()
        plt.close()
        
        
def basis_function(i, grid, k = 3):
    if k==3:
        grid = insert(grid, 0, 0)
        grid = append(grid, grid[-1]+1)
        i = i+1 #Because of the padded grid
    
    def base_case_function(u):
        return 0 if (grid[i-1] == grid[i]) else (1 if grid[i-1] <= u < grid[i] else 0)
    
    if k == 0:
        return base_case_function
    else:
        def iteration_function(u):
            if grid[i+k-1] == grid[i-1]: #hmmm
                factor_1 = 0
            else:
                factor_1 = (u - grid[i-1]) / (grid[i+k-1] - grid[i-1])
                
            if grid[i+k] == grid[i]:
                factor_2 = 0
            else:
                factor_2 = (grid[i+k] - u ) / (grid[i+k] - grid[i])
            return factor_1 * basis_function(i, grid, k-1)(u) + factor_2 * basis_function(i+1, grid, k-1)(u)
        return iteration_function
        


            

# def basis_functions(grid, i, k = 3, factor = 1):
#     if k == 0:
#         #return basis
#         return lambda u : 0 if (grid[i-1] == grid[i]) else (1 if grid[i-1] <= u < grid[i] else 0)
    
#     return lambda u: (u - grid[i-1]) / (grid[i+k-1] - grid[i-1]) * basis_functions(grid, i, k-1) \
#                     + (grid[i+k] -u ) / (grid[i+k] - grid[i]) * basis_functions(grid, i+1, k-1) 
    
# def basis_functions(grid, i, k = 3, factor = 1):
#     if k == 0:
#         #return basis
#         return lambda u : 0 if (grid[i-1] == grid[i]) else (factor if grid[i-1] <= u < grid[i] else 0)
    
#     else:
#         factor_1 = lambda u: (u - grid[i-1]) / (grid[i+k-1] - grid[i-1])
#         factor_2 = lambda u: (grid[i+k] -u ) / (grid[i+k] - grid[i])
#         return basis_functions(grid,i,k-1,factor = factor) + basis_functions(grid, i+1, k-1, factor = factor)

# class BasisFunctions():
#     def __init__(self, grid, i, k):
#         self.i = i
#         self.grid = grid
#         self.k = k

#     def __call__(self, u):
#         grid = self.grid
#         k = self.k
#         i = self.i
        
#         if k == 0:
#             return 0 if (grid[i-1] == grid[i]) else (1 if grid[i-1] <= u < grid[i] else 0)
#         else:
#             factor_1 = (u - grid[i-1]) / (grid[i+k-1] - grid[i-1])
#             factor_2 = (grid[i+k] - u ) / (grid[i+k] - grid[i])
#             return BasisFunctions(grid,i,k-1)(u) * factor_1 + BasisFunctions(grid, i+1, k-1)(u) * factor_2


if __name__ =='__main__':       
    #control_points = array([[-1,5],[2,-1],[3,-2],[3,1],[6,3],[1,2],[-1,2],[-2,0]])
    control_points = control_points_2
    grid = grid_2
    
    spline = CubicSpline(grid, control_points)
    sol = spline()
    
    N5 = spline.basis_function(15)
    x = linspace(0, 1, 1000)
    y = [N5(u) for u in x]
    plot(x,y)
        
        
#vectorize?
#inheritance?
#plot method parameters
        
    
    
    