"""
  Fenics script to solve a 2D Helmholtz problem in a bounded rectangular domain
  The primal unknown is the pressure field, discretized with piecewise linear elements
"""
import numpy as np
from dolfin import *
from mshr import *


rho_fluid = 1.21  # mass density [kg/m^3]
vel_fluid = 343.  # sound speed [m/s]
gamma_porous = 1.4 # Specific heat capacity ratio

Nele = 100; L0=0.; L1=1.
# Create mesh
mesh = IntervalMesh(Nele, L0, L1)

# Definition of the expression f(x)=x^2
f = Expression('pow(x[0],2)', degree=2)

# Define function space V = Lagrange 1st polynomials, DG = Piecewise constants
P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, P1)
DG = FiniteElement("DG", mesh.ufl_cell(), 0)
Q = FunctionSpace(mesh, DG)

# Define function u
#u = interpolate(f, V)
u = TrialFunction(V)

# Compute gradient of u
#grad_u = project(grad(u)[0], Q)
pressure = project(-rho_fluid * vel_fluid ** 2 / gamma_porous * grad(interpolate(u,V))[0], Q)

# Evaluate grad_u
print(pressure(0.5))

