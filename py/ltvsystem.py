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
        
        self.xtraj = np.zeros((self.N, self.nx))
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

    def updateTrajectory(self, x0, u0, trajMode):
        """Returns a (N,nx)-shaped traj
        
        trajMode should be one of the constants in the group up top.

        x0 --- either a (nx,) vector (current state) or an (N,nx) array with a given horizon (used with `trajMode=GIVEN_POINT_OR_TRAJ`). NOTE: when a horizon is provided, x0[0,:] must still have the current state.

        u0 --- input to linearize around (does not matter in a lot of cases if the linearization does not depend on the input).
        """
        # Update the initial state
        xlin = x0[0,:] if len(x0.shape) > 1 else x0
        ulin = u0[0,:] if len(u0.shape) > 1 else u0
        # Assumes that when a trajectory is provided, the first is the initial condition (see docstring)
        self.l[:self.nx] = -xlin
        self.u[:self.nx] = -xlin
        
        # First dynamics
        dyn = self.m.getLinearDynamics(xlin, ulin)
        Ad, Bd = dyn[0:2]
        bAffine = len(dyn) > 2
        
        for ti in range(self.N):
            # Dynamics update in the equality constraint (also set xlin) --
            # Get linearization point if it is provided
            if trajMode == GIVEN_POINT_OR_TRAJ:
                if len(x0.shape) > 1:
                    xlin = x0[ti, :]
                if len(u0.shape) > 1:
                    ulin = u0[ti, :]
            # Get linearization point by iterating a provided input
            elif trajMode in [ITERATE_TRAJ] and ti > 0:
                # update the point at which the next linearization will happen
                xlin = self.m.dynamics(xlin, ulin)
            elif trajMode == PREV_SOL_TRAJ:
                # use the previous solution shifted by 1 timestep (till the end)
                # NOTE: uses the end point twice (probably matters least then anyway)
                prevSolIdx = self.nx * min(ti+1, self.N)  # index into prevSol
                xlin = self.prevSol[prevSolIdx : prevSolIdx + self.nx]
                
            if ti > 0 and (trajMode == GIVEN_POINT_OR_TRAJ and len(x0.shape) > 1) or trajMode in [ITERATE_TRAJ, PREV_SOL_TRAJ]:
                # if a trajectory is provided or projected, need to get the newest linearization
                dyn = self.m.getLinearDynamics(xlin, ulin)
                Ad, Bd = dyn[0:2]
            
            # Place new Ad, Bd in Aeq
            csc.updateDynamics(self.A, self.N, ti, Ad=Ad, Bd=Bd)
            # Update RHS of constraint
            fd = dyn[2] if bAffine else np.zeros(self.nx)
            self.l[self.nx * (ti+1) : self.nx * (ti+2)] = -fd
            self.u[self.nx * (ti+1) : self.nx * (ti+2)] = -fd
            # /dynamics update --
            self.xtraj[ti, :] = xlin

        return self.xtraj

if __name__ == "__main__":
    print("Testing LTVSystem")
    from models.pendulums import Pendulum
    # import pen

    model = Pendulum()
    ltvs = LTVSystem(model)




