#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 19:23:20 2021

@author: nils
"""

import numpy as np
import math


def get_matrix_domain_2(delta_x, nx, ny):
    main_diag = -4*np.ones(nx*ny)     
    sub_diag = np.ones(nx * ny - 1)
    sup_diag = np.ones(nx * ny - 1)
    sub_diag[nx-1:nx*ny-nx+2:nx] = 0 
    sup_diag[nx-1::nx] = 0 
    diag2 = np.ones(nx*(ny-1))
    
    A = (np.diag(main_diag) #main diag
            + np.diag(sup_diag,k=1) + np.diag(sub_diag,k=-1) #subdiag 1
            + np.diag(diag2, k = -nx) + np.diag(diag2, k = nx)) *  1/delta_x**2 #subdiag 2
    
    return A

def get_matrix_domain_outer(delta_x, nx, ny, domain_1 = True):
    A0 = get_matrix_domain_2(delta_x, nx, ny)
    
    if domain_1: 
        index = np.arange(nx-1, nx*ny, nx)
    else: 
        index = np.arange(0, nx*ny, nx)
        
    A0[index, index] = -3/delta_x**2

    return A0       


def get_neumann(sol, delta_x, t_gamma_1, t_gamma_2):
    ny, nx = sol.shape
    
    derivative_l = (sol[0:math.floor(ny/2), 0] - t_gamma_1)/delta_x
    derivative_r = (sol[math.floor(ny/2)+1::, -1] - t_gamma_2)/delta_x
    
    return derivative_l, derivative_r

        


    