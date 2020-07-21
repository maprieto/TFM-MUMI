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

# Definition of error
def computeNormError(approximateSol, exactSol):
    return np.linalg.norm(approximateSol.vector() - exactSol.vector(), ord=2)

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
        #values[0] = 0.5 * (u0(x[0] + vel * t) + u0(x[0] - vel * t) \
        #                   - u0(-x[0] - vel * t + 2. * L1) - u0(-x[0] + vel * t + 2. * L0) \
        #                   + u0(x[0] - vel * t - 2. * L0 + 2. * L1) + u0(x[0] + vel * t - 2. * L1 + 2. * L0))

        values[0] = 0.5 * (u0(x[0] + vel * t) + u0(x[0] - vel * t) - u0(-x[0] + vel * t + 2. * L0))

    def value_shape(self):
        return ()

# ========== PARAMETERS ================================================================================================

# Parameter values for the fluid
rho_fluid = 1.21  # mass density [kg/m^3]
vel_fluid = 343.  # sound speed [m/s]

# Mesh limits (unit interval)
L0 = 0.; L1 = 1.

# Define initial data
#u_at_0 = Expression('fabs(x[0]-a)<c ? b-pow(x[0]-a,2)/pow(c,2) : 0.',a=0.5*(L0+L1), b=1., c=(L1-L0)/20., degree=2)
u_at_0 = Expression('fabs(x[0]-a)<b-tol ? exp(-1./(1.-pow((x[0]-a)/b,2)))/exp(-1.) : 0.', a=0.5 * (L0 + L1), b=(L1 - L0) / 20, tol=1e-3, degree=6)
v_at_0 = Expression('0.', degree=1)

# Load term (right-hand side in the wave equation)
force = Expression('0.*t', t=0., degree=1)

# ========== SETUP =====================================================================================================

# Define boundary subdomains
rigid_boundary = CompiledSubDomain("on_boundary && (near(x[0], L0))", L0=L0, L1=L1)
transparent_boundary = CompiledSubDomain("on_boundary && (near(x[0], L1))", L0=L0, L1=L1)

# Initialize exact solution
uex = ExactSolution(t=0, u0=u_at_0, vel=vel_fluid, L0=L0, L1=L1, degree=2)

# ========== COMPUTATION ===============================================================================================

def compute(Nele, dt, T_init, T_final):#, filename):

    #global u_at_0
    #global uex

    # Set number of elements in mesh and create mesh
    mesh = IntervalMesh(Nele, L0, L1)

    # Initialize mesh function for boundary
    boundary_markers = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundary_markers.set_all(0)  # all faces (interior and exterior) are set to zero
    rigid_boundary.mark(boundary_markers, 1)  # rigid boundary
    transparent_boundary.mark(boundary_markers, 2)  # rigid boundary

    # Write to file the boundary markers (to check reference numbers)
    vtk_boundaries = File("results/boundaries.pvd")
    vtk_boundaries << boundary_markers

    # Initialize mesh function for the physical domain
    domain_markers = MeshFunction("size_t", mesh, mesh.topology().dim())
    domain_markers.set_all(0)  # all elements are set to zero

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
    a_damping = vel_fluid * inner(u, w) * ds(2)

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
    Nt = np.int((T_final - T_init) / dt)  # number of time steps
    t_vec = T_init + np.arange(0, Nt) * dt  # time array

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

    displacement.rename("u", "u")
    velocity.rename("v", "v")
    acceleration.rename("a", "a")

    # Store acceleration, velocity and displacement in a function
    displacement.vector().set_local(y0.get_local())
    velocity.vector().set_local(v0.get_local())
    acceleration.vector().set_local(a0.get_local())
    solution = interpolate(uex, V)
    solution.rename("uex", "uex")

    #filename = 'results/wave_equation_Nele_%i_dt_%f.xdmf' % (Nele, dt)
    filename = 'results/_test.xdmf'

    xdmf_file = XDMFFile(filename)
    xdmf_file.parameters['rewrite_function_mesh'] = False
    xdmf_file.parameters['functions_share_mesh'] = True

    # Storage initial data
    xdmf_file.write(displacement, 0.)
    xdmf_file.write(velocity, 0.)
    xdmf_file.write(acceleration, 0.)
    xdmf_file.write(solution, 0.)

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
        drawProgressBar(jt + 1, Nt)

        # Store acceleration, velocity and displacement in a function
        displacement.vector().set_local(y0.get_local())
        velocity.vector().set_local(v0.get_local())
        acceleration.vector().set_local(a0.get_local())

        # Compute the exact solution
        uex.t = dt * (jt + 1)
        solution = interpolate(uex, V)
        solution.rename("uex", "uex")

        # Write values only at some specific time steps (each Nplot time steps values are saved to file)
        Nplot = 1
        if jt % Nplot == 0:
            xdmf_file.write(displacement, dt * (jt + 1))
            xdmf_file.write(velocity, dt * (jt + 1))
            xdmf_file.write(acceleration, dt * (jt + 1))
            xdmf_file.write(solution, dt * (jt + 1))

    # Ending -----------------------------------------------------------------------------------------------------------

    xdmf_file.close()

    finalNormError = computeNormError(displacement, solution)

    print('\n\n Nele = %g' % Nele)
    print(' dt = %g' % dt)
    print(' Nt = %g' % Nt)
    print(' Final norm2 Error = %g' % finalNormError)
    sys.stdout.write("\n")
    return finalNormError

# ========== UNIT TESTS ================================================================================================
# to run the unit test just type:
#           py.test-3 -s -v wave_equation_1D-v2.py

def test_order_h_dt():

    # ---------- PARAMETERS --------------------------------------------------------------------------------------------

    # Time interval
    T_init = 0.0  # initial time
    #T_final = 2. / vel_fluid  # final time (time for only two reflections)
    T_final = 0.3 / vel_fluid
    tol = 1. # error tolerance (%)

    # ---------- SETUP -------------------------------------------------------------------------------------------------

    # Open CSV file to store errors
    csvFile = open('results/errorLog.csv', 'w', newline='')
    csvWriter = csv.writer(csvFile)
    csvWriter.writerow(["h", "dt", "error"])

    errorLogDt = []; dtLogDt = []
    errorLogH = []; hLogH = []
    errorLogComb = []; hLogComb = []; dtLogComb = []

    # ---------- COMPUTATION -------------------------------------------------------------------------------------------

    # Fixing h - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    Nele = 250
    h = (L1 - L0) / Nele
    baseDt = h/vel_fluid

    #for pert in [-10, -5, -2, 0, 2, 5, 10]:  # perturbation in dt in percentage
        #dt = baseDt * (1 + pert / 100)
    for dt in [1E-05, 5E-06, 2.5E-06, 1.25E-06]:  # perturbation in dt in percentage

        error = compute(Nele, dt, T_init, T_final)

        csvWriter.writerow([h, dt, error])
        errorLogDt = np.append(errorLogDt, error)
        dtLogDt = np.append(dtLogDt, dt)

    # Fixing dt - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    dt = 1.E-05
    baseNele = (L1- L0) / (dt * vel_fluid)

    #for pert in [-10, -5, -2, 0, 2, 5, 10]:#[-5, -2, 0, 2, 5]:  # perturbation in Nele in percentage
        #Nele = int( round( baseNele * (1 + pert/100) ) )
    for Nele in [50, 100, 200, 400, 800]:
        h = (L1 - L0) / Nele
        error = compute(Nele, dt, T_init, T_final)

        csvWriter.writerow([h, dt, error])
        errorLogH = np.append(errorLogH, error)
        hLogH = np.append(hLogH, h)

    # Combined - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    for Nele in [50, 100, 200, 400, 800]:
        h = (L1 - L0) / Nele
        dt = h / vel_fluid
        error = compute(Nele, dt, T_init, T_final)

        csvWriter.writerow([h, dt, error])
        errorLogComb = np.append(errorLogComb, error)
        hLogComb = np.append(hLogComb, h)
        dtLogComb = np.append(dtLogComb, dt)


    csvFile.close() # Close CSV file

    # ---------- GRAPHS ------------------------------------------------------------------------------------------------

    plt.subplots_adjust(wspace=0.2, hspace=1)

    # LINEAR DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # Linear regression
    slopeDt, interceptDt, r_value, p_value, std_err = stats.linregress(dtLogDt, errorLogDt)
    print("error v. dt: slope: %f    intercept: %f" % (slopeDt, interceptDt))

    slopeH, interceptH, r_value, p_value, std_err = stats.linregress(hLogH, errorLogH)
    print("error v. h: slope: %f    intercept: %f" % (slopeH, interceptH))

    projLogComb = np.sqrt(hLogComb**2 + dtLogComb**2)
    slopeComb, interceptComb, r_value, p_value, std_err = stats.linregress(projLogComb, errorLogComb)
    print("error v. h: slope: %f    intercept: %f" % (slopeComb, interceptComb))

    # Setup plot
    figLin = plt.figure(1)

    # Error v. dt
    axDt = figLin.add_subplot(3, 1, 1)
    axDt.plot(dtLogDt, errorLogDt, marker = "o")
    axDt.plot(dtLogDt, interceptDt + slopeDt * dtLogDt, color = 'red')
    axDt.set(xlabel='dt', ylabel='error', title='Error varying dt (h = 4E-03)')

    # Error v. h
    axH = figLin.add_subplot(3, 1, 2)
    axH.plot(hLogH, errorLogH, marker = "o")
    axH.plot(hLogH, interceptH + slopeH * hLogH, color = 'red')
    axH.set(xlabel='h', ylabel='error', title='Error varying h (dt = 1E-05)')

    # Combined
    axComb = figLin.add_subplot(3, 1, 3)
    axComb.plot(projLogComb, errorLogComb, marker="o")
    axComb.plot(projLogComb, interceptComb + slopeComb * projLogComb, color='red')
    axComb.set(xlabel='projection of dt and h over the line of slope v_fluid', ylabel='error', title='Error varying dt and h simultaneously')

    # Save
    figLin.savefig("figures/errorLinear.png")

    # LOG DATA - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    plt.subplots_adjust(wspace=0.2, hspace=1)

    errorLogDt = np.log(errorLogDt); dtLogDt = np.log(dtLogDt)
    errorLogH = np.log(errorLogH); hLogH = np.log(hLogH)
    errorLogComb = np.log(errorLogComb); projLogComb = np.log(projLogComb)

    # Linear regression
    slopeDt, interceptDt, r_value, p_value, std_err = stats.linregress(dtLogDt, errorLogDt)
    print("log(error) v. log(dt): slope: %f    intercept: %f" % (slopeDt, interceptDt))

    slopeH, interceptH, r_value, p_value, std_err = stats.linregress(hLogH, errorLogH)
    print("log(error) v. log(h): slope: %f    intercept: %f" % (slopeH, interceptH))

    slopeComb, interceptComb, r_value, p_value, std_err = stats.linregress(projLogComb, errorLogComb)
    print("error v. h: slope: %f    intercept: %f" % (slopeComb, interceptComb))

    # Setup plot
    figLog = plt.figure(2)

    # Error v. dt
    axDt = figLog.add_subplot(3, 1, 1)
    axDt.plot(dtLogDt, errorLogDt, marker = "o")
    axDt.plot(dtLogDt, interceptDt + slopeDt * dtLogDt, color = 'red')
    axDt.set(xlabel='log(dt)', ylabel='log(error)', title='Error varying dt (h = 4E-03)')

    # Error v. h
    axH = figLog.add_subplot(3, 1, 2)
    axH.plot(hLogH, errorLogH, marker = "o")
    axH.plot(hLogH, interceptH + slopeH * hLogH, color = 'red')
    axH.set(xlabel='log(h)', ylabel='log(error)', title='Error varying h (dt = 1E-05)')

    # Combined
    axComb = figLog.add_subplot(3, 1, 3)
    axComb.plot(projLogComb, errorLogComb, marker="o")
    axComb.plot(projLogComb, interceptComb + slopeComb * projLogComb, color='red')
    axComb.set(xlabel='projection of dt and h over the line of slope v_fluid', ylabel='error', title='Error varying dt and h simultaneously')


    # Save
    figLog.savefig("figures/errorLog.png")

    # ---------- TEST --------------------------------------------------------------------------------------------------

    msg = '\n Slope varying log(dt) = %g' % slopeDt
    print(msg)
    #assert abs(slopeDt - 2) < tol, msg

    msg = '\n Slope varying log(h) = %g' % slopeH
    print(msg)
    #assert abs(slopeH - 2) < tol, msg

# ========== MAIN ======================================================================================================

if __name__ == '__main__':

    # ========== PARAMETERS ==========

    #global u_at_0
    #global uex

    # Time interval
    T_init = 0.0  # initial time
    T_final = 1.6 / vel_fluid  # final time (time for only two reflections)

    Nele = 100
    h = (L1-L0) / Nele
    dt = h / vel_fluid



    # ========== COMPUTATION ==========
    #u_at_0 = Expression('fabs(x[0]-a)<c ? b-pow(x[0]-a,2)/pow(c,2) : 0.',a=0.5*(L0+L1), b=1., c=(L1-L0)/20., degree=2)
    #uex = ExactSolution(t=0, u0=u_at_0, vel=vel_fluid, L0=L0, L1=L1, degree=2)
    compute(Nele, dt, T_init, T_final)#, 'results/_initialConditionsComparisson-1.xdmf')


    #u_at_0 = Expression('fabs(x[0]-a)<b-tol ? exp(-1./(1.-pow((x[0]-a)/b,2)))/exp(-1.) : 0.', a=0.5 * (L0 + L1), b=(L1 - L0) / 20, tol=1e-3, degree=6)
    #uex = ExactSolution(t=0, u0=u_at_0, vel=vel_fluid, L0=L0, L1=L1, degree=2)
    #compute(Nele, dt, T_init, T_final, 'results/_initialConditionsComparisson-2.xdmf')

    #u_at_0 = Expression('fabs(x[0]-a)<b-tol ? 1 : 0.', a=0.5 * (L0 + L1), b=(L1 - L0) / 20, tol=1e-3, degree=1)
    #uex = ExactSolution(t=0, u0=u_at_0, vel=vel_fluid, L0=L0, L1=L1, degree=2)
    #compute(Nele, dt, T_init, T_final, 'results/_initialConditionsComparisson-3.xdmf')

    # Print output result
