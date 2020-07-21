"""
  Fenics script to solve a 2D Helmholtz problem in a bounded rectangular domain
  The primal unknown is the pressure field, discretized with piecewise linear elements
"""
import numpy as np
from dolfin import *
from mshr import *

# Computational domain
domain = Rectangle(Point(0.,0,),Point(1., 1.))

# Create mesh
mesh = generate_mesh(domain, 30)

# Definition of the expression f(x,y)=x*y
f = Expression('x[0]*x[1]', degree=2)

# Define function space Q = Lagrange 1st polynomials, DG = Piecewise constants
P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
Q = FunctionSpace(mesh, P1)
DG = FiniteElement("DG", mesh.ufl_cell(), 0)
V = FunctionSpace(mesh, DG * DG)

# Define function u
u = interpolate(f, Q)

# Compute gradient of u
grad_u = project(grad(u), V)

print(grad(u))
print(V)
print(grad_u)

# Evaluate grad_u
print(grad_u(0.5,0.5))

