import numpy as np
import scipy.linalg as l
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator
from matplotlib import cm
# FIX FIRST SUPERDIAGONAL TO MAKE EVERYTHING WORK
def get_matrix_domain_2(delta_x):
    L = 1
    nx = int(1/delta_x - 1)
    if nx % 2 == 0:
        ny = 2 * L * nx + 1
    else:
        ny = 2 * L * nx
    
    main_diag = [-4 for x in range(nx * ny)]     
    sub_diag = np.ones(nx * ny - 1)
    sup_diag = np.ones(nx * ny - 1)
    sub_diag[nx-1:nx*ny-nx+2:nx] = 0 # typ nåt sånt
    sup_diag[nx-1::nx] = 0 # typ nåt sånt

    diag2 = [1 for x in range(nx * (ny - 1))]
    A = (np.diag(main_diag) #main diag
            + np.diag(sup_diag,k=1) + np.diag(sub_diag,k=-1) #subdiag 1
            + np.diag(diag2, k = -nx) + np.diag(diag2, k = nx)) *  1/delta_x**2 #subdiag 2
    
    return A
    

def dirichlet_neumann_iteration():
    t_wf = 5
    t_gamma_1 = 20
    t_gamma_2 = 20
    t_H = 40
    t_normal = 15
    L = 1
    delta_x = float(1/20)
    nx = int(1/delta_x - 1)
    if nx % 2 == 0:
        ny = 2 * L * nx + 1
    else:
        ny = 2 * L * nx
    
    A = get_matrix_domain_2(delta_x)
    rhs = np.zeros(nx*ny)
    
    rhs[0:int(ny/2)*nx - nx+1:nx] -= t_gamma_1 #bottom left
    rhs[int(ny/2)*nx::nx] -= t_normal #top left
    rhs[nx-1:int(ny/2)*nx:nx] -= t_normal #bottom right
    rhs[int(ny/2)*nx + nx -1:: nx] -= t_gamma_2 #top right
    rhs[0:nx] -= t_wf #Bottom
    rhs[-nx ::] -= t_H # top
    #print(rhs.reshape((ny,nx)))
    rhs *= 1/delta_x**2

    solution = l.solve(A,rhs)
    
    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #surf = ax.plot_surface(X, Y, solution.reshape(nx,ny), cmap=cm.coolwarm, linewidth=0)

    X, Y = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,2,ny))
    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, solution.reshape((ny,nx)), cmap=cm.coolwarm, linewidth=0)
    plt.show()
    #print(solution.reshape((ny,nx)))
    #print(np.shape(solution.reshape((ny,nx))))
    #print(right)
    #plt.close()
    #plt.imshow(A)
    #plt.show()


if __name__ == '__main__':
    dirichlet_neumann_iteration()