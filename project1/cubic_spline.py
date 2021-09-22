# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 10:18:07 2021

"""
import unittest
from numpy import *
from matplotlib.pyplot import *
from test_points_2 import control_points_2, grid_2
import os

class CubicSpline:
    def __init__(self, grid, control_points):
        if not len(grid) == len(control_points) + 2:
            raise Exception("# of grid points are expected to be 2 more than # of control points")
        grid = sort(grid)
        self.grid = grid
        self.control_points = control_points

        
    def __call__(self):
        uvec = linspace(self.grid[2], self.grid[-3], 1000)
        sol = array([list(self.blossom(u)) for u in uvec])
        return sol
    
    
    def blossom(self,u,index_given = False, index = 0):
        """
        Runs the blossom recursion to obtain s(u). 
        Can also be used to return the blossoms 
        for the add-on assignment. 
        """
        if(not index_given):
            index = self._hot_interval(u)
        d = self._get_control_points(index)
        dA = self._interpolation(d[0], d[1], u, index-2, index+1) 
        dB = self._interpolation(d[1], d[2], u, index-1, index+2)
        dC = self._interpolation(d[2], d[3], u, index, index+3)
        dAB = self._interpolation(dA, dB, u, index-1, index+1) 
        dBC = self._interpolation(dB, dC, u, index, index+2)
        if(index_given):
            return [dAB, dA, d[0]]
        else:
            return self._interpolation(dAB, dBC, u, index, index+1)
        
        
    def _interpolation(self,d1,d2,u,leftmost,rightmost):
        """
        Determines the scalar factor alpha(u) and 
        thereafter uses linear interpolation. 
        """
        if (leftmost == rightmost):
            alpha = 0;
        alpha = (self.grid[rightmost] - u)/ (self.grid[rightmost]-self.grid[leftmost])
        
        return alpha*array(d1) + (1-alpha)*array(d2)


    def _hot_interval(self,u):
        """
        Finds the "hot" interval.
        """
        for i in range(len(self.grid)):
            if (u == self.grid[2]):
                return 2
            elif (u <= self.grid[i]):
                return i-1

            
    def _get_control_points(self,index):
        """
        Selects the four control points 
        corresponding to the hot interval.
        """
        return self.control_points[index-2:index+2]
    
    
    def plot(self, sol, de_Boor = False, addon_blossoms = False, addon_index = 0):
        """
        Plots the solution along with 
        (optional) De Boor control points and (optional) blossoms. 
        """
        x = sol[:,0]
        y = sol[:,1]
        plot(x, y)
        
        leg = ["Spline"]
        
        if(addon_blossoms):
            
            uvec = linspace(self.grid[addon_index-2], self.grid[addon_index+2], 100)
            d_points = array([list(self.blossom(u, True, addon_index)) for u in uvec])
            
            dAB_x = [d_point[0, 0] for d_point in d_points]
            dAB_y = [d_point[0, 1] for d_point in d_points]
            dA_x = [d_point[1, 0] for d_point in d_points]
            dA_y = [d_point[1, 1] for d_point in d_points]
            
            plot(dAB_x, dAB_y, color='red')
            plot(dA_x, dA_y, color='cyan')
            plot(d_points[0][2,0],d_points[0][2,1], marker='*' )
            leg.extend(("d[u, u, u$_i$]", "d[u, u$_{i-1}$, u$_i$]", "d[u$_{i-2}$, u$_{i-1}$, u$_i$]"))
            
        if(de_Boor):
            de_Boor_x = [self.control_points[i][0] for i in range(len(self.control_points))]
            de_Boor_y = [self.control_points[i][1] for i in range(len(self.control_points))]             
            plot(de_Boor_x, de_Boor_y, linestyle = 'dashed' )
            leg.append("Control points")
        
        legend(leg)
        show()
        close()
        
        
def basis_function(i, grid, k = 3):
    """
    Returns the i:th basis function. 
    """
    if k == 3:
        grid = insert(grid, 0, 0)
        grid = append(grid, grid[-1]+1)
        i = i+1 #Because of the padded grid
    
    def base_case_function(u):
        """
        Returns value of basis function for k == 0. 
        """
        return 0 if (grid[i-1] == grid[i]) else (1 if grid[i-1] <= u < grid[i] else 0)
    
    if k == 0:
        return base_case_function
    else:
        def iteration_function(u):
            """
            Returns value of basis function for k != 0.
            """
            if grid[i+k-1] == grid[i-1]: #if nodes coincide
                factor_1 = 0
            else:
                factor_1 = (u - grid[i-1]) / (grid[i+k-1] - grid[i-1])
                
            if grid[i+k] == grid[i]: #if nodes coincide
                factor_2 = 0
            else:
                factor_2 = (grid[i+k] - u ) / (grid[i+k] - grid[i])
                
            return factor_1 * basis_function(i, grid, k-1)(u) + factor_2 * basis_function(i+1, grid, k-1)(u)
        
        return iteration_function
        

if __name__ =='__main__':       
    control_points = control_points_2
    grid = grid_2
    
    spline = CubicSpline(grid, control_points)
    basis_functions = [basis_function(i,grid) for i in range(5,20)]
    sol = spline()
    
    x = linspace(0, 1, 1000)
    y = [[basis_functions[i](u) for u in x] for i in range(len(basis_functions))]
    
    figure(dpi = 600)
    title("Basis functions")
    for i in range(len(y)):
        plot(x,y[i])
        
    figure(dpi = 600)
    title("B-spline solution")
    spline.plot(sol, de_Boor=True, addon_blossoms=True, addon_index=2)
    
    os.system("python -m unittest discover")
    show(block=False)
    