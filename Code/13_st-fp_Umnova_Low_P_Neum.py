import numpy as np
from dolfin import *
from mshr import *
import matplotlib.pylab as plt
import sys
import os.path
import csv
from scipy import stats
parameters['linear_algebra_backend'] = 'PETSc'

# ========== DEFINITIONS ===============================================================================================

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

# ========== PARAMETERS ================================================================================================

probIdStr = 'st-fp-UmnovaLow-P-N' # Identification of the problem for save files' names

# Parameter values for the fluid
rho_fluid = 1.21  # mass density [kg/m^3]
vel_fluid = 343.  # sound speed [m/s]

# Parameter values for the porous media
rho_porous = 1.5 # mass density [kg/m^3]
phi_porous = 0.36 #0.5 Porosity [0-1]
gamma_porous = 1.4 # Specific heat capacity ratio
sigma_porous = 27888. #100 Flux resistivity [N.s/m^4]
alfa_inf = 1.89 #1.5 Tortuosity

# Porous medium thickness
porous_thickness = 0.25

# Mesh limits (unit interval)
L0 = 0.; L1 = 1.1

# Material interfaces
Li0 = L1 - porous_thickness # Fluid-Porous boundary

# ========== SETUP =====================================================================================================

# Define porous subdomain
porous_domain = CompiledSubDomain("x[0] > Li0 - tol", Li0=Li0, tol=1e-10) # At further end

# Define boundary subdomains
speaker_boundary = CompiledSubDomain("on_boundary && (near(x[0], L0))", L0=L0)
interface = CompiledSubDomain("near(x[0], Li0)", Li0=Li0)
transparent_boundary = CompiledSubDomain("on_boundary && (near(x[0], L1))", L1=L1)

# ========== COMPUTATION ===============================================================================================

def compute(Nele, dt, T_init, T_final, savePrefix=''):

    # Set number of elements in mesh and create mesh
    mesh = IntervalMesh(Nele, L0, L1)

    u_at_0 = Expression('0.', degree=1)
    v_at_0 = Expression('0.', degree=1)

    # Initialize mesh function for boundary
    boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_markers.set_all(0)  # all faces (interior and exterior) are set to zero

    speaker_boundary.mark(boundary_markers, 1)      # speaker boundary
    interface.mark(boundary_markers, 2)             # coupling boundary
    transparent_boundary.mark(boundary_markers, 3)  # transparent boundary

    # Write to file the boundary markers (to check reference numbers)
    vtk_boundaries = File("results/boundaries.pvd")
    vtk_boundaries << boundary_markers

    # Initialize mesh function for the physical domain
    domain_markers = MeshFunction("size_t", mesh, mesh.topology().dim())
    domain_markers.set_all(1)  # all elements are set to zero (change to 1 to have all porous)
    porous_domain.mark(domain_markers, 1)  # porous domain

    # Write to file the subdomain markers (to check reference numbers)
    vtk_subdomains = File("results/subdomains.pvd")
    vtk_subdomains << domain_markers

    # Define new measures associated with each exterior boundaries
    dx = Measure('dx', domain=mesh, subdomain_data=domain_markers)
    ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
    dS = Measure('dS', domain=mesh, subdomain_data=boundary_markers)

    # Define function space (Lagrange 1st polynomials for each vector component)
    P1 = FiniteElement("CG", mesh.ufl_cell(), 1)
    V = FunctionSpace(mesh, P1)
    # THIS OUT
    # DG = FiniteElement("DG", mesh.ufl_cell(), 0)
    # Q = FunctionSpace(mesh, DG)

    # Define source term for the real and the imaginary part and null boundary conditions
    zero = Constant("0.0")
    #speakerOsc = Expression('a*(1-pow(w*(t-t0),2))*exp(-0.5*pow(w*(t-t0),2))', a=1, w=50, t=0., t0=0.1, degree=6)
    # MexHat = Expression('(1-pow(w*(t-t0),2))*exp(-0.5*pow(w*(t-t0),2))', w=50, t=0., t0=0.1, degree=6)

    # Define trial and test functions for the vector functional space V
    u = TrialFunction(V)
    w = TestFunction(V)

    # THIS IN
    # Define the metrices for the projection
    a_proj = inner(u, w) * dx
    A_proj = assemble(a_proj)
    solver_proj = PETScLUSolver(as_backend_type(A_proj), 'mumps')

    # Define coefficients for the variational problem
    C1 = gamma_porous * alfa_inf / vel_fluid**2
    C2 = phi_porous * gamma_porous * sigma_porous / (rho_fluid * vel_fluid**2)
    a = 50
    A0 = (2 * a * C1 + 3 * C2) / (2 * a * sqrt(C1 + C2 / a))
    A1 = C2 / (2*a**2*sqrt(C1 + C2/a))

    # Define the part of the variational problem associated to inertia (M)
    a_mass = rho_fluid * alfa_inf / phi_porous * inner(u, w) * dx(1) \
        + A1 * rho_fluid * vel_fluid**2 / (phi_porous * gamma_porous) * inner(u, w) * ds(3)

    # Define the part of the variational problem associated to damping (C)
    a_damping = sigma_porous * inner(u, w) * dx(1)\
        + A0 * rho_fluid * vel_fluid**2 / (phi_porous * gamma_porous)* inner(u, w) * ds(3)

    # Define the part of the variational problem associated to stiffness (K)
    a_stiff = rho_fluid * vel_fluid**2 / (phi_porous * gamma_porous) * inner(grad(u), grad(w)) * dx(1)

    # Load term (right-hand side in the wave equation)
    force = Expression('x[0] < tol ? 1/phi * (1-pow(w*(t-t0),2))*exp(-0.5*pow(w*(t-t0),2)): 0.', tol = 1e-10, phi=phi_porous, w=50, t=0., t0=0.1, degree=6)

    # Assemble the matrices
    A_stiff = assemble(a_stiff)
    A_mass = assemble(a_mass)
    A_damping = assemble(a_damping)

    # Define null Dirichlet boundary conditions in the whole boundary for Newmark matrices
    # bc_speaker = DirichletBC(V, speakerOsc, boundary_markers, 1)

    # Time discretization setting for Newmark's scheme
    beta = 0.25  # Newmark scheme coefficient
    gamma = 0.5  # Newmark scheme coefficient

    # Time marching quantities
    Nt = np.int((T_final - T_init) / dt)  # number of time steps
    t_vec = T_init + np.arange(0, Nt) * dt  # time array

    # Assemble the matrices associated to the Newmark method
    # A is the effective stiffness matrix  A = K + a0 * M + a1 * C
    A = A_stiff + (1. / (beta * dt ** 2)) * A_mass + gamma / (beta * dt) * A_damping

    # Apply boundary conditions
    # bc_speaker.apply(A)

    # Compute LU factorization and definition for re-using
    # solver = LUSolver(A)
    solver = PETScLUSolver(as_backend_type(A), 'mumps')

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
    b_damping = Vector(u_0.vector())

    # Compute initial acceleration for the Newmark method
    b_force = interpolate(force, V).vector()
    A_stiff.mult(y0, b_mass)
    A_damping.mult(v0, b_damping)
    a0 = Vector(u_0.vector())
    # Solve M * a0 + C * v0 + K * u0 = f to get a0
    solve(A_mass, a0, b_force - b_mass - b_damping)

    # Functions to storage the values
    displacement = Function(V)
    velocity = Function(V)
    acceleration = Function(V)

    # Functions for the pressure
    pressure = Function(V)
    press = Vector(u_0.vector())

    displacement.rename("u", "u")
    velocity.rename("v", "v")
    acceleration.rename("a", "a")

    # Store acceleration, velocity and displacement in a function
    displacement.vector().set_local(y0.get_local())
    velocity.vector().set_local(v0.get_local())
    acceleration.vector().set_local(a0.get_local())

    # THIS OUT
    # Compute pressure of the initial data
    # press = project(-rho_fluid * vel_fluid ** 2 / gamma_porous * grad(displacement)[0], Q)
    # pressure = interpolate(press,V)
    # pressure.rename("p", "p")
    # THIS IN
    b_proj = assemble(-rho_fluid * vel_fluid ** 2 / gamma_porous * grad(displacement)[0] * w * dx)
    solver_proj.solve(press, b_proj)
    pressure.vector().set_local(press.get_local())
    pressure.rename("p", "p")

    xdmfFileName = 'results/%s%s_N_%i_phi_%g_gam_%g_sig_%g.xdmf' % (savePrefix, probIdStr, Nele, phi_porous, gamma_porous, sigma_porous)

    xdmf_file = XDMFFile(xdmfFileName)
    xdmf_file.parameters['rewrite_function_mesh'] = False
    xdmf_file.parameters['functions_share_mesh'] = True

    # Storage initial data
    xdmf_file.write(displacement, 0.)
    xdmf_file.write(velocity, 0.)
    xdmf_file.write(acceleration, 0.)
    xdmf_file.write(pressure, 0.)

    # tP02= [0]; tP05= [0]; tP1= [0]; p_at_02 = [0]; p_at_05 = [0]; p_at_1 = [0];
    # setUp02 = True; setUp05 = True; setUp1 = True;
    csvFile = open('results/%s%s_Pressures_Nele_%g.csv' % (savePrefix, probIdStr, Nele), 'w', newline='')
    csvWriter = csv.writer(csvFile); csvWriter.writerow(["t", "p at 0.2", "p at 0.5", "p at 1"])

    # Loop in time: Newmark method -------------------------------------------------------------------------------------
    for jt in range(Nt + 1):
        # Update time in loads
        force.t = dt * (jt + 1)
        # Compute the loads at time t+dt
        b_rhs = interpolate(force, V).vector()
        # Compute right-hand side for Newmark
        # Calculate effective loads at time t + dt
        # R^(t+dt) + M (a0 U^t + a2 U.^t + a3 U..^t) + C (a1 U^t + a4 U.^t + a5 U..^t)
        A_mass.mult(1.0 / (beta * dt ** 2) * y0 + 1.0 / (beta * dt) * v0 + (1.0 / (2 * beta) - 1.0) * a0, b_mass)
        A_damping.mult(gamma / (beta * dt) * y0 + (gamma / beta - 1.) * v0 + dt / 2. * (gamma / beta - 2.) * a0,
                       b_damping)
        b_vec = b_rhs + b_mass + b_damping
        # Apply boundary conditions
        # bc_speaker.apply(b_vec)
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
        drawProgressBar(jt + 1, Nt)

        # Store acceleration, velocity and displacement in a function
        displacement.vector().set_local(y0.get_local())
        velocity.vector().set_local(v0.get_local())
        acceleration.vector().set_local(a0.get_local())

        # THIS OUT
        # press = project(-rho_fluid * vel_fluid ** 2 / gamma_porous * grad(displacement)[0], Q)
        # pressure = interpolate(press, V)
        # pressure.rename("p", "p")
        # THIS IN
        b_proj = assemble(-rho_fluid * vel_fluid ** 2 / gamma_porous * grad(displacement)[0] * w * dx)
        solver_proj.solve(press, b_proj)
        pressure.vector().set_local(press.get_local())
        pressure.rename("p", "p")

        # Write values only at some specific time steps (each Nplot time steps values are saved to file)
        Nplot = 1
        if jt % Nplot == 0:
            xdmf_file.write(displacement, dt * (jt + 1))
            xdmf_file.write(velocity, dt * (jt + 1))
            xdmf_file.write(acceleration, dt * (jt + 1))
            xdmf_file.write(pressure, dt * (jt + 1))

            # THIS OUT
            # csvWriter.writerow([dt * (jt + 1), press(0.2), press(0.5), press(1)])
            # THIS IN
            csvWriter.writerow([dt * (jt + 1), pressure(0.2), pressure(0.5), pressure(1)])

    # Ending -----------------------------------------------------------------------------------------------------------

    xdmf_file.close()
    csvFile.close()  # Close CSV file

    print('\n\n Nele = %g' % Nele)
    print(' dt = %g' % dt)
    sys.stdout.write("\n")
    return 1

# ========== MAIN ======================================================================================================

if __name__ == '__main__':

    # ========== PARAMETERS ==========

    # Time interval
    T_init = 0.0  # initial time
    T_final = 0.2#2 / vel_fluid  # final time (time for only two reflections)

    Nele = 50
    h = (L1-L0) / Nele
    dt = h / vel_fluid

    # ========== COMPUTATION ==========

    print('Test error: %g' % compute(Nele, dt, T_init, T_final, savePrefix='_guess_'))

