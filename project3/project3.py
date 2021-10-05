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
    ny = 2 * L * nx + 1
    
    diag0 = [-4 for x in range(nx * ny)]     
    diag1 = [1 for x in range(nx * ny - 1)]
    diag1[0:nx] = 0 # typ nåt sånt
    diag1[-nx::] = 0 # typ nåt sånt
    
    diag2 = [1 for x in range(nx * (ny - 1))]
    A = (np.diag(diag0) #main diag
            + np.diag(diag1,k=1) + np.diag(diag1,k=-1) #subdiag 1
            + np.diag(diag2, k = -nx) + np.diag(diag2, k = nx)) *  1/delta_x**2  #subdiag 2
    

def dirichlet_neumann_iteration():
    t_wf = 40
    t_gamma_1 = 10
    t_gamma_2 = 10
    t_H = 40
    t_normal = 15
    L = 1
    delta_x = float(1/4)
    nx = int(1/delta_x - 1)
    ny = 2 * L * nx + 1
    A = get_matrix_domain_2(delta_x)
    right = np.zeros(nx*ny)
    
    right[0:int(ny/2)*nx:nx] = -t_gamma_1 #bottom left
    right[nx-1 + int(ny/2)*nx :: nx] = -t_gamma_2 #top right
    temp = [i for i in range(nx*ny)]
    #print(temp[0:int(ny/2)*nx:nx])
    right[nx-1:int(ny/2)*nx:nx] = -t_normal #bottom right
    right[int(ny/2)*nx::nx] = -t_normal #top left
    
    right[0:nx] = -t_wf #Bottom
    right[-nx ::] = -t_H # top
    print(right)
    right *= 1/delta_x**2
    solution = l.solve(A,right)
    
    #fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #surf = ax.plot_surface(X, Y, solution.reshape(nx,ny), cmap=cm.coolwarm, linewidth=0)

    X, Y = np.meshgrid(np.linspace(0,1,nx), np.linspace(0,1,ny))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, solution.reshape((ny,nx)), cmap=cm.coolwarm, linewidth=0)
    plt.show()
    #print(solution.reshape((ny,nx)))
    #print(np.shape(solution.reshape((ny,nx))))
    #print(right)
    #plt.close()
    print(A)
    #plt.imshow(A)
    #plt.show()


if __name__ == '__main__':
    dirichlet_neumann_iteration()