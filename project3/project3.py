import numpy as np
import scipy.linalg as l
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator
from matplotlib import cm

def get_matrix_domain_2(delta_x, nx, ny):
    main_diag = [-4 for x in range(nx * ny)]     
    sub_diag = np.ones(nx * ny - 1)
    sup_diag = np.ones(nx * ny - 1)
    sub_diag[nx-1:nx*ny-nx+2:nx] = 0 
    sup_diag[nx-1::nx] = 0 

    diag2 = [1 for x in range(nx * (ny - 1))]
    A = (np.diag(main_diag) #main diag
            + np.diag(sup_diag,k=1) + np.diag(sub_diag,k=-1) #subdiag 1
            + np.diag(diag2, k = -nx) + np.diag(diag2, k = nx)) *  1/delta_x**2 #subdiag 2
    
    return A


def get_neumann(sol, delta_x, t_gamma_1, t_gamma_2):
    ##### finite diff
    # send solution + RV to get neumann 
    # diff between RV and inner point 
    ny, nx = sol.shape
    
    derivative_l = (sol[0:math.floor(ny/2), 0] - t_gamma_1)/delta_x
    derivative_r = (sol[0:math.floor(ny/2), -1] - t_gamma_2)/delta_x
    
    return derivative_l, derivative_r

def room_2(delta_x, t_gamma_1, t_gamma_2):
    t_wf = 5
    t_H = 40
    t_normal = 15
    L = 1
    nx = int(1/delta_x - 1)
    if nx % 2 == 0:
        ny = 2 * L * nx + 1
    else:
        ny = 2 * L * nx
    
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
    
    # plot 
    X, Y = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,2,ny))
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, solution, cmap=cm.coolwarm, linewidth=0)
    plt.show()
    
    return solution


if __name__ == '__main__':
    delta_x = float(1/20)
    t_gamma_1 = 20
    t_gamma_2 = 20
    
    room_2_sol = room_2(delta_x, t_gamma_1, t_gamma_2)
    iteration = get_neumann(room_2_sol, delta_x, t_gamma_1, t_gamma_2)
    print(iteration)
    
    
    