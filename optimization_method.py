# -*- coding: utf-8 -*-

from numpy import *
from matplotlib.pyplot import *
from scipy.optimize import minimize

class OptimizationMethod:
        
    def __init__(self, exact_line_search = True):
        self.exact_line_search = exact_line_search
        
        
    def __call__(self, problem, x0):
        H = self.defualt_hessian(x0, problem.func)
        x, H = self.step(H, x0, problem)
        x_old = x0
        tol = 1e-15
        steps = [x_old]
        while linalg.norm(problem.gradient(x)) > tol and linalg.norm(x-x_old) > tol:
            
            x_old = x
            x, H = self.step(H, x, problem)
            steps = np.vstack((steps, x_old))
        return x, steps
    def step(self):
        raise NotImplementedError()
        
    
class Newton(OptimizationMethod):
    
    
    def step(self, H, x, problem):
        if problem.gradient == None:
            #calculate it
            pass
        else:
            s = -H @ problem.gradient(x)
        if self.exact_line_search:
            alpha = self.exact_search(x, s, problem.func)
        else:
            alpha = self.inexact_search(x, s, problem.func)
        x_new = x + alpha*s
        #H_new = self.hessian( x_new, problem.func, H) #xnew? said x before
        H_new = self.hessian(x, x_new, problem, H) #xnew? said x before
        return x_new, H_new
    
    def defualt_hessian(self, x, f):
       n = len(x)
       G = zeros((n,n))
       h = 1e-3
       
       for i in range(n):
           for j in range(n):
               G[i,j] = (f(x + h*self._basisvec(n,(i,j),(1,1))) - f(x + h*self._basisvec(n,(i,j), (1,-1)))
                         - f(x + h*self._basisvec(n,(i,j),(-1,1))) + f(x + h*self._basisvec(n,(i,j),(-1,-1))))/(4*h**2)
       G = (G + G.T)/2
       return linalg.inv(G)
   
    def _basisvec(self, n, i, val):
        v = zeros(n)
        v[i[0]] += val[0]
        v[i[1]] += val[1]
        return v
   
    def hessian(self):
        raise NotImplementedError()
        
    def exact_search(self, x, s, f):
        
        return minimize(self.phi_func(x,s,f), 0).x
    
    def phi_func(self, x, s ,f):
        def phi(alpha):
            return f(x + alpha*s)
        return phi
    
    def inexact_search(self, x, s, f):
        #Powell-Wolfe
        sigma = 0.4999
        rho = 0.99999999
        alpha_minus = 2
        phi = self.phi_func(x, s, f)
        
        h = 1e-3
        phi_prime_0 = self.derivative(phi, 0)
        
        while phi(alpha_minus) > phi(0) + sigma*alpha_minus*phi_prime_0:
            alpha_minus = alpha_minus/2
            
        alpha_plus = alpha_minus
        
        while phi(alpha_plus) <= phi(0) + sigma*alpha_plus*phi_prime_0:
            alpha_plus *= 2
         
        
        while self.derivative(phi, alpha_minus) < rho * phi_prime_0:
            alpha_0 = (alpha_plus + alpha_minus)/2
            
            if phi(alpha_0) <= phi(0) + sigma*alpha_0*phi_prime_0: 
                alpha_minus = alpha_0
            else:
                alpha_plus = alpha_0
        return alpha_minus
    
    def get_gamma_delta(self, x, x_old, problem):
        delta = np.reshape(x - x_old, (len(x),1))
        if problem.gradient == None:
                pass
                #Calculate pls
        gamma = np.reshape(np.array(problem.gradient(x)) - np.array(problem.gradient(x_old)), (len(x),1))
        return delta, gamma
        
    def derivative(self, f, x):
        h = 1e-5
        return (f(x+h) - f(x-h))/(2*h)


class ClassicNewton(Newton):
    
    
    def hessian(self, x, f, H_prev = None):
        n = len(x)
        G = zeros((n,n))
        h = 1e-3
        
        for i in range(n):
            for j in range(n):
                G[i,j] = (f(x + h*self._basisvec(n,(i,j),(1,1))) - f(x + h*self._basisvec(n,(i,j), (1,-1)))
                          - f(x + h*self._basisvec(n,(i,j),(-1,1))) + f(x + h*self._basisvec(n,(i,j),(-1,-1))))/(4*h**2)
        G = (G + G.T)/2
        return linalg.inv(G)
    
    def _basisvec(self, n, i, val):
        v = zeros(n)
        v[i[0]] += val[0]
        v[i[1]] += val[1]
        return v


class BFGS(Newton):
    
    def hessian(self, x_old, x, problem, H_prev):
        tol = 1e-5
        
        delta, gamma = self.get_gamma_delta(x, x_old, problem)
        
        if linalg.norm(gamma) < tol or linalg.norm(delta) < tol: #This is cheating, look in to if we can have it somewhere else
            return np.eye(len(x))
    
        first = (1 + gamma.T @ H_prev @ gamma / (delta.T @ gamma) ) * delta @ delta.T / (delta.T @ gamma)
        second = (delta @ gamma.T @ H_prev + H_prev @ gamma @ delta.T) / (delta.T @ gamma)
        H = H_prev + first - second
        return H
    
class GoodBroyden(Newton):
    
    def hessian(self, x, x_old, problem, H_prev):
        delta, gamma = self.get_gamma_delta(x, x_old, problem)
            
        H = H_prev + (delta - H_prev @ gamma) / (delta.T @ H_prev @ gamma) @ delta.T @ H_prev
        
        return H

class BadBroyden(Newton):
    
    def hessian(self, x, x_old, problem, H_prev):
        delta, gamma = self.get_gamma_delta(x, x_old, problem)
        
        H = H_prev + (delta - H_prev @ gamma)/(gamma.T @ gamma) @ gamma.T
        
        return H
        
        
        
        