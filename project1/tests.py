from cubic_spline import CubicSpline, basis_function
from numpy import *
import unittest
from test_points_2 import control_points_2, grid_2
class Test(unittest.TestCase):
    
    def setUp(self):
        """
        Create spline object and basis functions.
        """
        self.spline = CubicSpline(grid_2, control_points_2)
        self.basis_functions = [basis_function(i,grid_2) for i in range(len(self.spline.control_points))]
    
    def test_unity(self):
        """
        Test if basis functions add up to 1. 
        """
        basis_functions = self.basis_functions
        expected = 1
        
        u_vec = linspace(0,1)
        for u in u_vec:
            result = 0
            for N in basis_functions:
                result += N(u)
            self.assertAlmostEqual(result, expected)
            
    def test_positive(self):
        """
        Test if basis functions positive. 
        """
        spline = self.spline
        basis_functions = self.basis_functions
        
        u_vec = linspace(0,1)
        for u in u_vec:
            for N in basis_functions:
                self.assertGreaterEqual(N(u), 0)
        
    def test_evaluate(self):
        """
        Test if De Boor algorithm and the basis functions generate same results. 
        """
        basis_functions = self.basis_functions
        spline = self.spline
        u_vec = linspace(spline.grid[2],spline.grid[-3])
        d = spline.control_points
        
        for u in u_vec:
            sum = [0,0]
            for i in range(len(basis_functions)):
                sum += array(d[i]) * basis_functions[i](u)
            self.assertAlmostEqual(sum[0], spline.blossom(u)[0])
            self.assertAlmostEqual(sum[1], spline.blossom(u)[1])
            
    def test_evaluate_point(self):
        """
        Test if test point accurate. 
        """
        spline = self.spline
        u = 0.2
        expected = array([-31.90219167, 6.47655833])
        result = spline.blossom(u)
        self.assertAlmostEqual(expected[0], result[0])
        self.assertAlmostEqual(expected[1], result[1])
    
if __name__ == '__main__':
    unittest.main()