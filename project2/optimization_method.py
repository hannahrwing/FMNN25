#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:52:01 2021

@author: nils
"""

from numpy import *
from matplotlib.pyplot import *
from scipy.optimize import minimize
import signal
import time

class OptimizationMethod:
        
    def __init__(self, exact_line_search = True):
        """
        Creates an optimization method which uses either
        exact or inexact line search.
        Parameters
        ----------
        exact_line_search : bool, optional
            Determines if exact or inexact line search is to be used. The default is True.
        Returns
        -------
        OptimizationMethod.
        """
        self.exact_line_search = exact_line_search
        
    def __call__(self, problem, x0):
        """
        Optimizes the function in the problem starting from
        the guessed point x0. 
        Returns the point for a local minima.
        Parameters
        ----------
        problem : OptimizationProblem
            The optimization problem that is to be solved.
        x0 : np.ndarray
            initial guess
        Returns
        -------
        x : np.ndarray
            point of optimum.
        """
        x, H = None, None
        for x, H in self.generate(problem, x0):
            pass
        return x
        
    def generate(self, problem, x0, max_num_steps = 1e5):
        """
        Returns a generator for the steps takes towards the minimum

        Parameters
        ----------
        problem : OptimizationProblem
        x0 : array
            DESCRIPTION.
        max_num_steps : int, optional
            DESCRIPTION. Max number of steps until algorithm stops.
            The default is 1e5.

        Yields
        ------
        TYPE
            point.
        H : TYPE
            Hessian matrix.

        """
        H = self.default_hessian(x0, problem.func)
        yield x0, H
        
        x, H = self.step(H, x0, problem)
        yield list(x), H
        
        x_old = x0
        tol1 = 1e-7
        tol2 = 1e-15
        num_steps = 0
        while linalg.norm(problem.gradient(x)) > tol1 and linalg.norm(x-x_old) > tol2:
            x_old = x
            x, H = self.step(H, x, problem)
            if num_steps > max_num_steps:
                yield list(x), H
                raise Exception("Maximum steps exceeded. No minimum found")
            num_steps += 1
            yield list(x), H
    
    
    def step(self):
        raise NotImplementedError()
        
    
class Newton(OptimizationMethod):
    
    def step(self, H, x, problem):
        """
        Uses the hessian matrix and the x-vector to take
        the next step towards optimizing the function.
        Does not matter whether the gradient is defined
        in the problem class or not.
        Returns the new x-vector x_new and the new hessian matrix.
        Parameters
        ----------
        H : np.ndarray
            Hessian from the previous step
        x : list
            previous point
        problem : OptimizationProblem
        Returns
        -------
        x_new : list
        H_new : np.ndarray
        """
        s = -H @ problem.gradient(x)
        if self.exact_line_search:
            alpha = self.exact_search(x, s, problem.func)
        else:
    
            alpha = self.inexact_search(x, s, problem.func)
        x_new = x + alpha*s
        H_new = self.hessian(x, x_new, problem, H)
        return x_new, H_new
    
    def default_hessian(self, x, f):
       """
        Calculates and returns the default hessian matrix by
        using finite differences.
        Parameters
        ----------
        x : np.ndarray
        f : function
        Returns
        -------
        np.ndarray
            Hessian computed by finite differance method.
        """
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
        """
        A help-function to calculate the deafult_hessian.
        Creates and returns an n-vectors with the values val
        in positions i.
        """
        v = zeros(n)
        v[i[0]] += val[0]
        v[i[1]] += val[1]
        return v
   
    def hessian(self):
        raise NotImplementedError()
        
    def exact_search(self, x, s, f):
        """
       Performs an exact search to return the most
       suitable alpha to take the next step.
       """
        return minimize(self.phi_func(x,s,f), 0).x
    
    def phi_func(self, x, s ,f):
        """
        Returns the phi function defined as f(x+alpha*s)
        """
        def phi(alpha):
            return f(x + alpha*s)
        return phi
    
    def inexact_search(self, x, s, f):
        """
        Performs an inexact search to return a suitable
        alpha to take the next step. This is executed
        utilizing the Powel-Wolfe conditions.
        """
        sigma = 1e-2
        rho = 0.9
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
        """
        Determines and returns the gamma and delta parameters
        which are used to calculate the hessians for the
        different optimization methods.
        Delta is the difference between two consecutive
        x-vectors.
        Gamma is the difference between two consecutive
        gradients.
        """
        delta = np.reshape(x - x_old, (len(x),1))
        gamma = np.reshape(np.array(problem.gradient(x)) - np.array(problem.gradient(x_old)), (len(x),1))
        return delta, gamma
        
    def derivative(self, f, x):
        """
        Differentiates the function f numerically in
        the value x for the x-vector component i.
        """
        h = 1e-7
        return (f(x+h) - f(x-h))/(2*h)


class ClassicNewton(Newton):
    
    
    def hessian(self, x_old, x, problem, H_prev = None):
        """
        Determines the hessian matrix used to
        take the next step in the algorithm.
        """
        return self.default_hessian(x, problem.func)
        

class BFGS(Newton):
    
    def hessian(self, x_old, x, problem, H_prev):
        """
        Calculates the hessian to take the next step for
        the BFGS method.
        """
        delta, gamma = self.get_gamma_delta(x, x_old, problem)
    
        first = (1 + gamma.T @ H_prev @ gamma / (delta.T @ gamma) ) * delta @ delta.T / (delta.T @ gamma)
        second = (delta @ gamma.T @ H_prev + H_prev @ gamma @ delta.T) / (delta.T @ gamma)
        H = H_prev + first - second
        return H
    
class GoodBroyden(Newton):
    
    def hessian(self, x, x_old, problem, H_prev):
        """
        Calculates the hessian to take the next step for
        the Good Broyden method.
        """

        delta, gamma = self.get_gamma_delta(x, x_old, problem)

        H = H_prev + (delta - H_prev @ gamma) / (delta.T @ H_prev @ gamma) @ delta.T @ H_prev
        return H
        
        
class DFP(Newton):
    
    def hessian(self, x_old, x, problem, H_prev):
        """
        Calculates the hessian to take the next step for
        the DFP method.
        """

        delta, gamma = self.get_gamma_delta(x, x_old, problem)

        
        first = delta @ delta.T / (delta.T @ gamma)
        second = H_prev @ gamma @ gamma.T @ H_prev / (gamma.T @ H_prev @ gamma)
        return H_prev + first - second 

class BadBroyden(Newton):
    
    def hessian(self, x, x_old, problem, H_prev):
        """
        Calculates the hessian to take the next step for
        the Bad Broyden method.
        """

        delta, gamma = self.get_gamma_delta(x, x_old, problem)
        H = H_prev + (delta - H_prev @ gamma)/(gamma.T @ gamma) @ gamma.T
        
        return H
        
    
class SymmetricBroyden(Newton):
    
    def hessian(self, x_old, x, problem, H_prev):
        """
        Calculates the hessian to take the next step for
        the Symmetric Broyden method.
        """
        delta, gamma = self.get_gamma_delta(x, x_old, problem)

        u = delta - H_prev @ gamma
        a = 1 / (u.T @ gamma)
        return H_prev  + a * u.T @ u