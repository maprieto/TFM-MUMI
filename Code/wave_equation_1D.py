"""
Fenics script to solve a the wave equation with rigid wall conditions in a one-dimensional domain
  rho*d2u/dt2-rho*c^2*d2u/dx2=f in [T_init,T_end]x[L0,L1]
  u(t,L0)=u(t,L1)=0 in [T_init,T_end]
  u(0,x)=u0(x) in [L0,L1]
  du/dt(0)=v0(x) in [L0,L1]
  The primal unknown is the displacement field, discretized with Lagrange P1 elements in each component
  The time-discretization is given by an implicit Newmark scheme
"""
import numpy as np
from dolfin import *
from mshr import *
import matplotlib.pylab as plt
import sys
import os.path
parameters['linear_algebra_backend'] = 'PETSc'

# Definition of a progress bar
def drawProgressBar(time_, N_, barlen=50):
    sys.stdout.write("\r")
    progress = ""
    for i_ in range(barlen):
        if i_ < 1. * barlen * time_ / N_:
            progress += "="
        else:
            progress += " "
    sys.stdout.write("[%s] # Step = %i" % (progress, time_))
    sys.stdout.flush()

def computeError(approximateSol, exactSol):
    return np.abs(approximateSol.vector().get_local() - exactSol.vector().get_local()) #/ np.abs(exactSol.vector().get_local()) #Relative error. Problems dividing by zero

# Parameter values for the fluid
rho_fluid = 1.21 # mass density [kg/m^3]
vel_fluid = 343. # sound speed [m/s]

# Time interval
T_init = 0.0  # initial time
T_final = 2./vel_fluid # final time (time for only two reflections)

# Create mesh with Nele elements in the unit interval [L0,L1]
L0=0.; L1=1.; Nele=100
mesh = IntervalMesh(Nele, L0, L1)

# Define initial data
#u_at_0 = Expression('fabs(x[0]-a)<c ? b-pow(x[0]-a,2)/pow(c,2) : 0.',a=0.5*(L0+L1), b=1., c=(L1-L0)/20., degree=2)
u_at_0 = Expression('fabs(x[0]-a)<b-tol ? exp(-1./(1.-pow((x[0]-a)/b,2)))/exp(-1.) : 0.',a=0.5*(L0+L1), b=(L1-L0)/20, tol=1e-3, degree=6)
v_at_0 = Expression('0.', degree=1)

# Load term (right-hand side in the wave equation)
force = Expression('0.*t', t=0., degree=1)

# Define exact solution (D'Alambert solution using the images principle and only 2 reflections taken into account)
class ExactSolution(UserExpression):
    def __init__(self, t, u0, vel, L0, L1, **kwargs):
        self.t = t
        self.L0 = L0
        self.L1 = L1
        self.u0 = u0
        self.vel = vel
        super().__init__(**kwargs)    
    def eval(self, values, x):
        L0 = self.L0
        L1 = self.L1
        t = self.t
        vel = self.vel
        u0 = self.u0
        values[0] = 0.5 * (u0(x[0]+vel*t) + u0(x[0]-vel*t) \
                           - u0(-x[0]-vel*t+2.*L1) - u0(-x[0]+vel*t+2.*L0) \
                           + u0(x[0]-vel*t-2.*L0+2.*L1) + u0(x[0]+vel*t-2.*L1+2.*L0))
    def value_shape(self):
        return ()

uex = ExactSolution(t=0, u0=u_at_0, vel=vel_fluid, L0=L0, L1=L1, degree=2)

# Define boundary subdomains
rigid_boundary = CompiledSubDomain("on_boundary && (near(x[0], L0) || near(x[0], L1))", L0=L0, L1=L1)

# Initialize mesh function for boundary
boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundary_markers.set_all(0) # all faces (interior and exterior) are set to zero
rigid_boundary.mark(boundary_markers, 1) # rigid boundary

# Write to file the boundary markers (to check reference numbers)
vtk_boundaries = File("results/boundaries.pvd")
vtk_boundaries << boundary_markers

# Initialize mesh function for the physical domain
domain_markers = MeshFunction("size_t", mesh, mesh.topology().dim())
domain_markers.set_all(0) # all elements are set to zero

# Write to file the subdomain markers (to check reference numbers)
vtk_subdomains = File("results/subdomains.pvd")
vtk_subdomains << domain_markers

# Define new measures associated with each exterior boundaries
dx = Measure('dx', domain=mesh, subdomain_data=domain_markers)
ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

# Define function space (Lagrange 1st polynomials for each vector component)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
V = FunctionSpace(mesh, P1)

# Define source term for the real and the imaginary part and null boundary conditions
zero = Constant("0.0")

# Define trial and test functions for the vector functional space V
u = TrialFunction(V)
w = TestFunction(V)

# Define the part of the variational problem associated to stiffness (K)
a_stiff = rho_fluid * pow(vel_fluid, 2) * inner(grad(u), grad(w)) * dx(0)

# Define the part of the variational problem associated to inertia (M)
a_mass = rho_fluid * inner(u, w) * dx(0)

# Define the part of the variational problem associated to damping (C)
a_damping = zero * inner(u, w) * dx(0)

# Assemble the matrices
A_stiff = assemble(a_stiff)
A_mass = assemble(a_mass)
A_damping = assemble(a_damping)

# Define null Dirichlet boundary conditions in the whole boundary for Newmark matrices
bc = DirichletBC(V, zero, boundary_markers, 1)

# Time discretization setting for Newmark's scheme
beta = 0.25  # Newmark scheme coefficient
gamma = 0.5  # Newmark scheme coefficient

# Time marching quantities
#dt = mesh.hmax()/vel_fluid # time step
dt=5E-07
Nt = np.int((T_final-T_init) / dt) # number of time steps
t_vec = T_init + np.arange(0, Nt) * dt # time array

# Assemble the matrices associated to the Newmark method
# A is the effective stiffness matrix  A = K + a0 * M + a1 * C
A = A_stiff + (1. / (beta * dt ** 2)) * A_mass + gamma / (beta * dt) * A_damping

# Apply boundary conditions
bc.apply(A)
    
# Compute LU factorization and definition for re-using
solver = LUSolver(A)

# Initial conditions
u_0 = interpolate(u_at_0, V)
v_0 = interpolate(v_at_0, V)

# Displacement at time t=0
y0 = Vector(u_0.vector())
y1 = Vector(u_0.vector())
# Velocity at time t=0
v0 = Vector(v_0.vector())

# Init vectors (their actual values will not be used)
b_mass = Vector(u_0.vector())
b_damping= Vector(u_0.vector())

# Compute initial acceleration for the Newmark method
b_force = interpolate(force, V).vector()
A_stiff.mult(y0, b_mass)
A_damping.mult(v0, b_damping)
a0 = Vector(u_0.vector())
# Solve M * a0 + C * v0 + K * u0 = f to get a0
solve(A_mass, a0, b_force - b_mass - b_damping )

# Functions to storage the values
displacement = Function(V)
velocity = Function(V)
acceleration = Function(V)
error = Function(V)

displacement.rename("u", "u")
velocity.rename("v", "v")
acceleration.rename("a", "a")
error.rename("err", "err")

# Store acceleration, velocity and displacement in a function
displacement.vector().set_local(y0.get_local())
velocity.vector().set_local(v0.get_local())
acceleration.vector().set_local(a0.get_local())
solution = interpolate(uex, V)
solution.rename("uex","uex")
error.vector().set_local(computeError(displacement, solution))
sumOfError = error.vector().sum()

xdmf_file = XDMFFile('results/wave_equation.xdmf')
xdmf_file.parameters['rewrite_function_mesh'] = False
xdmf_file.parameters['functions_share_mesh'] = True

# Storage initial data
xdmf_file.write(displacement, 0.)
xdmf_file.write(velocity, 0.)
xdmf_file.write(acceleration, 0.)
xdmf_file.write(solution, 0.)
xdmf_file.write(error, 0.)

# Loop in time: Newmark method
for jt in range(Nt+1):
    # Update time in loads
    force.t=dt*(jt+1) 
    # Compute the loads at time t+dt
    b_rhs = interpolate(force, V).vector()
    # Compute right-hand side for Newmark
    # Calculate effective loads at time t + dt
    # R^(t+dt) + M (a0 U^t + a2 U.^t + a3 U..^t) + C (a1 U^t + a4 U.^t + a5 U..^t)
    A_mass.mult(1.0 / (beta * dt ** 2) * y0 + 1.0 / (beta * dt) * v0 + (1.0 / (2 * beta) - 1.0) * a0 , b_mass)
    A_damping.mult(gamma / (beta * dt) * y0 + (gamma / beta - 1.) * v0 + dt / 2. * (gamma / beta - 2.) * a0, b_damping)
    b_vec = b_rhs + b_mass + b_damping
    # Apply boundary conditions
    bc.apply(b_vec)
    # Solve y_1
    solver.solve(y1, b_vec)

    # Compute the linear combinations to obtain v_1 and a_1 (at dof vector level)
    a1 = 1.0 / (beta * dt ** 2) * (y1 - y0) - 1.0 / (beta * dt) * v0 - (1.0 / (2 * beta) - 1.0) * a0
    v1 = gamma / (beta * dt) * (y1 - y0) - (gamma / beta - 1) * v0 - (gamma / (2.0 * beta) - 1.0) * dt * a0

    # Update the dofs for the next time step
    y0.set_local(y1.get_local())
    v0.set_local(v1.get_local())
    a0.set_local(a1.get_local())

    # Update time bar
    drawProgressBar(jt+1, Nt)

    # Store acceleration, velocity and displacement in a function
    displacement.vector().set_local(y0.get_local())
    velocity.vector().set_local(v0.get_local())
    acceleration.vector().set_local(a0.get_local())

    # Compute the exact solution
    uex.t=dt*(jt+1)
    solution = interpolate(uex, V)
    solution.rename("uex","uex")

    # Compute the error between the exact solution and the aproximate solution
    error.vector().set_local(computeError(displacement, solution))
    sumOfError += error.vector().sum()

    # Write values only at some specific time steps (each Nplot time steps values are saved to file)
    Nplot = 1
    if jt%Nplot == 0:
        xdmf_file.write(displacement, dt*(jt+1))
        xdmf_file.write(velocity, dt*(jt+1))
        xdmf_file.write(acceleration, dt*(jt+1))
        xdmf_file.write(solution, dt*(jt+1))
        xdmf_file.write(error, dt*(jt+1))
xdmf_file.close()
print('\n\n Nele = %g' % Nele)
print(' dt = %g' % dt)
print(' Nt = %g' % Nt)
print(' Sum of absolute errors = %g' % sumOfError)
sys.stdout.write("\n")


