import numpy as np
import scipy.linalg as l
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator
from matplotlib import cm

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

def get_matrix_domain_1(delta_x, nx, ny):
    A0 = get_matrix_domain_2(delta_x, nx, ny)
    for i in np.arange(nx-1,len(A0[1]), nx):
        A0[i][i] = -3/delta_x**2

    return A0    

def get_matrix_domain_3(delta_x, nx, ny):
    A0 = get_matrix_domain_2(delta_x, nx, ny) 
    for i in np.arange(0,len(A0[1]), nx):
        A0[i][i] = -3/delta_x**2
        
    
    return A0   

def get_neumann(sol, delta_x, t_gamma_1, t_gamma_2):
    ny, nx = sol.shape
    
    derivative_l = (sol[0:math.floor(ny/2), 0] - t_gamma_1)/delta_x
    derivative_r = (sol[math.floor(ny/2)+1::, -1] - t_gamma_2)/delta_x
    
    return derivative_l, derivative_r

def domain_2(delta_x, t_gamma_1, t_gamma_2):
    t_wf = 5
    t_H = 40
    t_normal = 15
    L = 1
    nx = int(1/delta_x - 1)
    ny = 2 * L * nx + 1

    
    A = get_matrix_domain_2(delta_x, nx, ny)
    rhs = np.zeros(nx*ny)
    
    rhs[0:int(ny/2)*nx - nx+1:nx] -= t_gamma_1 #bottom left
    rhs[int(ny/2)*nx::nx] -= t_normal #top left
    rhs[nx-1:int(ny/2)*nx:nx] -= t_normal #bottom right
    rhs[int(ny/2)*nx + nx -1:: nx] -= t_gamma_2 #top right
    rhs[0:nx] -= t_wf #bottom
    rhs[-nx ::] -= t_H #top
    rhs *= 1/delta_x**2

    solution = l.solve(A,rhs).reshape(ny, nx)
    
    # plot domain 2
    X, Y = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,2,ny))
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, solution, cmap=cm.coolwarm, linewidth=0)
    plt.show()
    
    return solution

def domain_1(delta_x, derivative):
    t_H = 40
    t_normal = 15
    nx = int(1/delta_x)
    
    ny = int(1/delta_x-1)

    A = get_matrix_domain_1(delta_x, nx, ny)
    rhs = np.zeros(nx * ny)
    
    rhs[0::nx] -= t_H / delta_x**2 #left
    rhs[nx-1::nx] -= derivative / delta_x #right
    rhs[0:nx] -= t_normal / delta_x**2 #bottom
    rhs[-nx::] -= t_normal / delta_x**2 #top
    
    solution = l.solve(A,rhs).reshape(ny, nx)
    
    # plot domain 1
    X, Y = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny))
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, solution, cmap=cm.coolwarm, linewidth=0)
    plt.show()
    
    
def domain_3(delta_x, derivative):
    t_H = 40
    t_normal = 15
    nx = int(1/delta_x)
    
    ny = int(1/delta_x-1)

    A = get_matrix_domain_3(delta_x, nx, ny)
    rhs = np.zeros(nx * ny)
    
    rhs[0::nx] -= derivative / delta_x #left
    rhs[nx-1::nx] -= t_H / delta_x**2 # right
    rhs[0:nx] -= t_normal / delta_x**2 #bottom
    rhs[-nx::] -= t_normal / delta_x**2 #top
    
    solution = l.solve(A,rhs).reshape(ny, nx)
    
    # plot domain 3
    X, Y = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny))
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, solution, cmap=cm.coolwarm, linewidth=0)
    plt.show()

if __name__ == '__main__':
    delta_x = float(1/20)
    t_gamma_1 = 20
    t_gamma_2 = 20
    
    domain_2_sol = domain_2(delta_x, t_gamma_1, t_gamma_2)
    derivative_l, derivative_r = get_neumann(domain_2_sol, delta_x, t_gamma_1, t_gamma_2)
    
    domain_1_sol = domain_1(delta_x, derivative_l)
    print(get_matrix_domain_1(delta_x, int(1/delta_x), int(1/delta_x - 1)) * delta_x**2)
    
    domain_3_sol = domain_3(delta_x, derivative_r)
    print(get_matrix_domain_3(delta_x, int(1/delta_x), int(1/delta_x - 1)) * delta_x**2)
    
    
    
    