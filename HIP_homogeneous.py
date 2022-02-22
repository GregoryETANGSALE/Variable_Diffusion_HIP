"""

Hybridizable Interior Penalty method for Solving 
Homogeneous Diffusion problem

div(grad(u)) = f    in Omega
           u = 0    on Gamma

f     : Source term

Copyright ©
Author: Grégory ETANGSALE, University of La Réunion

"""

##################################################################

# Import Netgen/NGSolve and Python modules

from ngsolve import *
from netgen.geom2d import unit_square
import math

ngsglobals.msg_level = 1

##################################################################

# Mesh generation

geometry = SplineGeometry()
mesh = Mesh(unit_square.GenerateMesh(maxh=1/2, quad_dominated=True))

##################################################################

# Exact solution / Source term

ue = x*y*(1-x)*(1-y)*exp(-x**2-y**2)

dux = (1-2*x)*exp(-x**2-y**2)*y*(1-y)-2*x*ue
duy = (1-2*y)*exp(-x**2-y**2)*x*(1-x)-2*y*ue

d2ux = (-2-2*x*(1-2*x))*exp(-x**2-y**2)*y*(1-y)-2*ue-2*x*dux
d2uy = (-2-2*y*(1-2*y))*exp(-x**2-y**2)*x*(1-x)-2*ue-2*y*duy

f = -(d2ux + d2uy)

##################################################################

# Numerical parameters of the H-IP method

order = 1                       # polynomial degree
B = 1
gamma = 2*(order+1)*(order+2)   # constant for stabilization function
epsilon = 0                     # variant of the H-IP formulation

condense = True                 # static condensation 

##################################################################

# Main functions

def fes_space(mesh):
    V = L2(mesh, order=order, discontinuous=True)
    Vhat = FacetFESpace(mesh, order=order, dirichlet="left|right|top|bottom", discontinuous=True)
    fes = FESpace([V,Vhat], dgjumps=True)

    print ("vdofs:    ", fes.Range(0))
    print ("vhatdofs: ", fes.Range(1))

    return fes 



def Assembling(fes):
    # Primal HDG : Create the associated discrete variables u & uhat
    u, uhat = fes.TrialFunction()
    v, vhat = fes.TestFunction()

    n = specialcf.normal(mesh.dim)      # unit normal
    h = specialcf.mesh_size             # element size
    
    # Creating the bilinear form associated to our primal HDG method:
    a = BilinearForm(fes, eliminate_internal=condense)

    # 1. Interior terms
    a_int = grad(u)*grad(v) 
    # 2. Boundary terms : residual
    a_fct = (grad(u)*n)*(vhat-v)+epsilon*(grad(v)*n)*(uhat-u)
    a_sta = gamma/(h**B)*(u-uhat)*(v-vhat)
    
    a += SymbolicBFI( a_int )
    a += SymbolicBFI( a_fct + a_sta , element_boundary=True )

    # Creating the linear form associated to our primal HDG method:
    l = LinearForm(fes)
    l += SymbolicLFI(f*v)

    # Preconditonner
    c = Preconditioner(type="direct", bf=a, inverse="umfpack")

    # gfu : total vector of dof [u,uhat]
    gfu = GridFunction(fes)

    return a,l,c,gfu


def compute_L2_error(uh):
    # CalcL2Error computes the L2 error of the discrete variable ||u-uh||_X=sqrt(sum(||u-uh||^2_A))
    return sqrt( Integrate((uh-ue)**2, mesh, order=2*order ) )


##################################################################

# Solve linear system

def SOLVE_DIFFUSION():

    fes = fes_space(mesh)

    [a,l,c,gfu]=Assembling(fes)

    start_timer = time.time()

    a.Assemble()
    l.Assemble()

    # boundary condition   
    gfu.components[1].Set(ue, definedon=mesh.Boundaries("left|right|top|bottom"))

    # SOLVE LINEAR SYSTEM
    BVP(bf=a, lf=l, gf=gfu, pre=c, maxsteps=3, prec=1.0e-16).Do()

    end_timer = time.time()

    Draw (gfu.components[0], mesh, "u-primal-HDG")

    error_u = compute_L2_error(gfu.components[0])
    store.append([fes.ndof, mesh.ne, error_u,end_timer-start_timer])

    return gfu


##################################################################

# Main programm

store = []
N = 5       # Number of mesh refinement

for i in range(N):
    Draw(ue, mesh, "exact_solution")
    gfu = SOLVE_DIFFUSION()
    if i == N-1:
        pass
    else:
        mesh.Refine()


# Display L2-norm estimate (history of convergence)

i = 1
print (" Cells      L2_error     r     CPU_time")
print("%6d & %2.1e & - & %s  \\\ " % \
      (store[0][1], store[0][2], store[0][3]))
while i < len(store) :
    rate_L2   = log(store[i-1][2]/store[i][2])/log(2)
    print("%6d & %2.1e & %2.1f & %s \\\ " % \
          (store[i][1], store[i][2], rate_L2, store[i][3]))
    i =  i+1


