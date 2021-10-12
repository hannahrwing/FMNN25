import numpy as np
import scipy.linalg as l
import matplotlib.pyplot as plt
import math
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator
from matplotlib import cm
#
from plot import Plotter

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
    print(A)
    rhs = np.zeros(nx*ny)
    
    rhs[0:int(ny/2)*nx - nx+1:nx] -= t_gamma_1 #bottom left
    rhs[int(ny/2)*nx::nx] -= t_normal #top left
    rhs[nx-1:int(ny/2)*nx:nx] -= t_normal #bottom right
    rhs[math.ceil(ny/2)*nx + nx-1:: nx] -= t_gamma_2 #top right
    
    rhs[0:nx] -= t_wf #bottom
    rhs[-nx ::] -= t_H #top
    rhs *= 1/delta_x**2

    solution = l.solve(A,rhs).reshape(ny, nx)
    
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
    return solution
    
    
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
    return solution


def mpi():
    from mpi4py import MPI
    delta_x = float(1/20)
    nx2 = int(1/delta_x-1)
    ny2 = 2*nx2+1
    
    nx13 = nx2+1
    ny13 = nx2
    
    t_gamma_1 = 20
    t_gamma_2 = 20
    
    omega = 0.8
    
    sol_old1, sol_old2, sol_old3 = 0,0,0
    
    comm = MPI.Comm.Clone(MPI.COMM_WORLD)
    rank = comm.Get_rank()
    
    for i in range(10):
        
        if rank == 1:
            domain_2_sol = domain_2(delta_x, t_gamma_1, t_gamma_2)
            derivative_l, derivative_r = get_neumann(domain_2_sol, delta_x, t_gamma_1, t_gamma_2)
            
            comm.Send([derivative_l, MPI.DOUBLE], dest=0, tag=69)
            comm.Send([derivative_r, MPI.DOUBLE], dest=2, tag=420)
            
        if rank == 0:
            derivative = np.empty(math.floor(ny2/2), dtype=float)
            comm.Recv(derivative, source=1, tag=69)
            
            domain_1_sol = domain_1(delta_x, derivative)
            comm.Send([domain_1_sol, MPI.DOUBLE], dest=1, tag=666)
            
        if rank == 2:
            derivative = np.empty(math.floor(ny2/2), dtype=float)
            comm.Recv(derivative, source=1, tag=420)
            
            domain_3_sol = domain_3(delta_x, derivative)
            comm.Send([domain_3_sol, MPI.DOUBLE], dest=1, tag=1)
        
        if rank ==  1:
            domain_1_sol = np.empty((ny13,nx13), dtype=float)
            domain_3_sol = np.empty((ny13,nx13), dtype=float)
            comm.Recv([domain_1_sol, MPI.DOUBLE], source=0, tag=666)
            comm.Recv([domain_3_sol, MPI.DOUBLE], source=2, tag=1)
            
            if not i == 0:
                domain_1_sol = omega*domain_1_sol + (1-omega)*sol_old1
                domain_2_sol = omega*domain_2_sol + (1-omega)*sol_old2
                domain_3_sol = omega*domain_3_sol + (1-omega)*sol_old3
            sol_old1, sol_old2, sol_old3 = domain_1_sol, domain_2_sol, domain_3_sol
            
            t_gamma_1 = domain_1_sol[:,-1]
            t_gamma_2 = domain_3_sol[:,0]
                
    if rank == 1:      
        sols = [domain_1_sol, domain_2_sol, domain_3_sol]
        print(sols)
        plotter = Plotter(sols)()
        
def non_mpi():
    delta_x = float(1/20)
    nx2 = int(1/delta_x-1)
    ny2 = 2*nx2+1
    nx13 = nx2+1
    ny13 = nx2
    
    t_gamma_1 = 10
    t_gamma_2 = 10
    
    omega = 0.8
    
    sol_old1, sol_old2, sol_old3 = 0,0,0
    
    for i in range(10):
        domain_2_sol = domain_2(delta_x, t_gamma_1, t_gamma_2)
        derivative_l, derivative_r = get_neumann(domain_2_sol, delta_x, t_gamma_1, t_gamma_2)
        domain_1_sol = domain_1(delta_x, derivative_l)
        domain_3_sol = domain_3(delta_x, derivative_r)
        sols = sols = [domain_1_sol, domain_2_sol, domain_3_sol]
        plotter = Plotter(sols)()
        if i != 0:
            domain_1_sol = omega*domain_1_sol + (1-omega)*sol_old1
            domain_2_sol = omega*domain_2_sol + (1-omega)*sol_old2
            domain_3_sol = omega*domain_3_sol + (1-omega)*sol_old3
            
        sol_old1, sol_old2, sol_old3 = domain_1_sol.copy(), domain_2_sol.copy(), domain_3_sol.copy()
        t_gamma_1 = domain_1_sol[:,-1]
        t_gamma_2 = domain_3_sol[:,0]
        
    
    
if __name__ == '__main__':
    non_mpi()
        


    