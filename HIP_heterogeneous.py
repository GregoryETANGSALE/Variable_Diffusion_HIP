"""

Hybridizable Interior Penalty method for Solving 
Heterogeneous and Anisotropic Diffusion problem

div(kappa * grad(u)) = f    in Omega
           u = 0    on Gamma

kappa : Homogeneous dispersion tensor
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

def part1(geom):
    coord = [ (0,0), (0.5,0), (0.5,0.5), (0,0.5) ]
    nums1 = [geom.AppendPoint(*p) for p in coord]
    lines = [(nums1[0], nums1[1], "gammaD" , 1, 0),
             (nums1[1], nums1[2], "interface", 1, 2),
             (nums1[2], nums1[3], "interface", 1, 4),
             (nums1[3], nums1[0], "gammaD" , 1, 0)]

    for p0,p1,bc,left,right in lines:
        geom.Append( ["line", p0, p1], bc=bc, leftdomain=left, rightdomain=right,maxh=hmax)

    geom.SetMaterial(1, "domain1")
    return (geom,nums1)

def part2(geom,nums1):
    coord = [ (1,0), (1,0.5) ]
    nums2 = [geom.AppendPoint(*p) for p in coord]
    lines = [(nums1[1], nums2[0], "gammaD", 2,0),
             (nums2[0], nums2[1], "gammaD", 2,0),
             (nums2[1], nums1[2], "interface", 2,3)]

    for p0,p1,bc,left,right in lines:
        geom.Append( ["line", p0, p1], bc=bc, leftdomain=left, rightdomain=right,maxh=hmax)

    geom.SetMaterial(2, "domain2")
    return (geom,nums2)

def part3(geom,nums1,nums2):
    coord = [ (1,1), (0.5,1) ]
    nums3 = [geom.AppendPoint(*p) for p in coord]
    lines = [(nums2[1], nums3[0], "gammaD", 3,0),
             (nums3[0], nums3[1], "gammaD", 3,0),
             (nums3[1], nums1[2], "interface", 3,4)]

    for p0,p1,bc,left,right in lines:
        geom.Append( ["line", p0, p1], bc=bc, leftdomain=left, rightdomain=right,maxh=hmax)

    geom.SetMaterial(3, "domain3")
    return (geom,nums3)

def part4(geom,nums1,nums3):
    coord = [ (0,1) ]
    nums4 = [geom.AppendPoint(*p) for p in coord]
    lines = [(nums3[1], nums4[0], "gammaD", 4,0),
             (nums4[0], nums1[3], "gammaD", 4,0)]

    for p0,p1,bc,left,right in lines:
        geom.Append( ["line", p0, p1], bc=bc, leftdomain=left, rightdomain=right,maxh=hmax)

    geom.SetMaterial(4, "domain4")
    return (geom)


geo = SplineGeometry()
geo,nums1 = part1(geo)
geo,nums2 = part2(geo,nums1)
geo,nums3 = part3(geo,nums1,nums2)
geo = part4(geo,nums1,nums3)

mesh = Mesh(geo.GenerateMesh(maxh=1/2,quad_dominated=True))

##################################################################

# Create the diffusion tensor inside each regions
lbd = 1e6

kappa1 = (1,0,0,lbd)
kappa2 = (1/lbd,0,0,1)

kappar = { "domain1" : kappa1, "domain2" : kappa2, "domain3" : kappa1, "domain4" : kappa2 }
kappa_coef = [ kappar[mat] for mat in mesh.GetMaterials() ]
print ("kappa_coef=\n", kappa_coef)
print("")
kappa = CoefficientFunction(kappa_coef,dims=(2,2))

# Exact solution / Source term

ue   = sin(pi*x)*sin(pi*y)
f = kappa[0]*pi**2*sin(pi*x)*sin(pi*y) + kappa[3]*pi**2*sin(pi*x)*sin(pi*y)

##################################################################

# Numerical parameters of the H-IP method

order = 1                       # polynomial degree
gamma = 2*(order+1)*(order+2)   # constant for stabilization function
epsilon = 1                     # variant of the H-IP formulation

condense = True                 # static condensation 

##################################################################

# Main functions

def fes_space(mesh):
    V = L2(mesh, order=order, discontinuous=True)
    Vhat = FacetFESpace(mesh, order=order, dirichlet="gammaD", discontinuous=True)
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

    Kn = InnerProduct( n, CoefficientFunction(kappa*n,dims=(2,1)) )  # normal permeability
    
    # Creating the bilinear form associated to our primal HDG method:
    a = BilinearForm(fes, eliminate_internal=condense)

    # 1. Interior terms
    a_int = kappa*grad(u)*grad(v) 
    # 2. Boundary terms : residual
    a_fct = (kappa*grad(u)*n)*(vhat-v)+epsilon*(kappa*grad(v)*n)*(uhat-u)
    a_sta = (gamma*Kn/h)*(u-uhat)*(v-vhat)
    
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
    gfu.components[1].Set(ue, definedon=mesh.Boundaries("gammaD"))

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


