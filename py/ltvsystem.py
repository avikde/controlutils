import autograd.numpy as np
import scipy.sparse as sparse
from . import csc

"""
What to use over the horizon during the MPC update (see docstring of update) ---
trajMode --- 

`GIVEN_POINT_OR_TRAJ`: if a single goal point is given in x0, use linearization at that point, or if a trajectory is given in x0, use the linearization at each knot point

`ITERATE_TRAJ`: apply the current or provided input u0 recursively to produce a trajectory. Warning: this can produce bad trajectories for complicated dynamics

`PREV_SOL_TRAJ`: Use the x1, ..., xN from the previous run

`SQP` i.e. sequential QP: take the first solution, get a traj, and then linearize around that and rinse repeat (before actually taking a control action in the sim)
"""
GIVEN_POINT_OR_TRAJ = 0
ITERATE_TRAJ = 1
PREV_SOL_TRAJ = 2
SQP = 4

class LTVSystem:
    """This deals with the (LTV) dynamics and constraints"""

    def __init__(self, model):
        self.m = model

    def init(self, nx, nu, N, x0, xmin, xmax, umin, umax, polyBlocks=None):
        """Return A, l, u"""
        self.nx = nx
        self.nu = nu
        self.N = N
        self.polyBlocks = polyBlocks

        # Create the CSC A matrix manually.
        # conA = CondensedA(self.nx, self.nu, N, polyBlocks=polyBlocks)
        conA = csc.init(nx, nu, N, polyBlocks=polyBlocks)
        self.A = sparse.csc_matrix((conA.data, conA.indices, conA.indptr))
        
        # - input and state constraints
        lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
        uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])
        
        leq = np.hstack([-x0, np.zeros(N*self.nx)])
        ueq = leq
        self.l = np.hstack([leq, lineq])
        self.u = np.hstack([ueq, uineq])
        if polyBlocks is not None:
            # Add corresponding RHS elements for polyhedron membership
            # Only C x <= d type constraints
            Nctotal = self.A.shape[0] - len(self.l)
            self.l = np.hstack((self.l, np.full(Nctotal, -np.inf)))
            self.u = np.hstack((self.u, np.full(Nctotal, np.inf)))
        
        return self.A, self.l, self.u
    
    def updateStateConstraint(self, ti, xidx, u=None, l=None):
        '''Update a state constraint

        The constraint added is l <= x(ti)[xidx] <= u
        '''
        # get to the ineq constraints
        ioffs = self.nx * (self.N + 1 + ti)
        if u is not None:
            self.u[ioffs + xidx] = u
        if l is not None:
            self.l[ioffs + xidx] = l
    
    def updatePolyhedronConstraint(self, ti, pbi, Ci, di):
        '''Update the polyhedron membership constraint for time i

        The constraint added is Ci projection_pbi( x(ti) ) <= di

        pbi = index into polyBlocks provided during init

        To "remove" the constraint, just set di = np.full(*, np.inf)
        '''
        # Only C x <= d type constraints, so only change A and u
        ioffs = csc.updatePolyBlock(self.A, self.nx, self.nu, self.N, ti, self.polyBlocks, pbi, Ci)
        # update u, but not l (which stays at -inf)
        assert Ci.shape[0] == len(di)
        assert Ci.shape[0] == self.polyBlocks[pbi][1]
        assert Ci.shape[1] == self.polyBlocks[pbi][2]
        self.u[ioffs : ioffs + len(di)] = di


if __name__ == "__main__":
    print("Testing LTVSystem")
    from models.pendulums import Pendulum
    # import pen

    model = Pendulum()
    ltvs = LTVSystem(model)




