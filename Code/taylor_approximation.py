"""
Fenics script to solve a Helmholtz problem in mixed-boundary condition for a three-dimensional domain
  The primal unknown is the pressure field, discretized with Lagrange P1-elements
"""
import numpy as np
import scipy.interpolate as interp

# Compute a Tyalor polynomial approximation of exp(x)
def compute(x, N):
    a = 1.
    s = 1.
    for i in np.arange(1, N):
        a *= x / i
        s += a
    return s

# Definition of unit tests
# to run the unit test just type:
#           py.test-3 -s -v taylor_approximation.py
def test_exp():
	# Parameters
	a = 0.1
	N = 5
	tol = 1. # error tolerance (%)
	# Compute
	pa = compute(a, N)
	# Compute relative error
	error_rel = np.abs(pa - np.exp(a)) / np.abs(np.exp(a))
	msg = '\n Relative error = %g' % error_rel ; print(msg)
	assert error_rel*100. < tol, msg

# Run the main code for a rotated cylinder
# to run the test just type:
#           python3 taylor_approximation.py
if __name__ == '__main__':
	# Parameters
	a = 0.5
	N = 10
	# Compute FE solution
	pa = compute(a, N)
	# Print outpur result
	print("Taylor approximation:", pa)
