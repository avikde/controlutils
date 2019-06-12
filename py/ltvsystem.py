import autograd.numpy as np
import scipy.sparse as sparse
import osqp
import time
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

class LTVDirTran:
    """This deals with the (LTV) dynamics, constraints, and trajectory"""

    def __init__(self, model):
        self.m = model
        self.tqpsolve = np.nan
        self.niter = np.nan

    def init(self, nx, nu, N, x0, polyBlocks=None, useModelLimits=True):
        """Return A, l, u"""
        self.nx = nx
        self.nu = nu
        self.N = N
        self.polyBlocks = polyBlocks

        # Constraints
        if hasattr(self.m, 'limits') and useModelLimits:
            self.umin, self.umax, self.xmin, self.xmax = self.m.limits
        else:
            # unconstrained
            self.umin = np.full(nu, -np.inf)
            self.xmin = np.full(nx, -np.inf)
            self.umax = -self.umin
            self.xmax = -self.xmin

        # Create the CSC A matrix manually.
        # conA = CondensedA(self.nx, self.nu, N, polyBlocks=polyBlocks)
        conA = csc.init(nx, nu, N, polyBlocks=polyBlocks)
        self.A = sparse.csc_matrix((conA.data, conA.indices, conA.indptr))
        
        # - input and state constraints
        lineq = np.hstack([np.kron(np.ones(N+1), self.xmin), np.kron(np.ones(N), self.umin)])
        uineq = np.hstack([np.kron(np.ones(N+1), self.xmax), np.kron(np.ones(N), self.umax)])
        
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


class LTVSolver(LTVDirTran):
    """Combine LTV dynamics with the solver"""
    
    # Parameters that could be modified --
    # If returned u violated umin/umax by a fraction (this is useful for systems with small input magnitudes and large tolerance)
    MAX_ULIM_VIOL_FRAC = None
    # /parameters

    def initSolver(self, P, q, **settings):
        # Create an OSQP object
        self.prob = osqp.OSQP()
        # Full so that it is not made sparse. prob.update() cannot change the sparsity structure
        # Setup workspace
        self.prob.setup(P, q, self.A, self.l, self.u, warm_start=True, **settings)#, eps_abs=1e-05, eps_rel=1e-05
        

    def debugResult(self, res):
        # Debugging infeasible
        if res.info.status == 'primal infeasible':
            # NOTE: when infeasible, OSQP appears to return a np.ndarray vector in res.x, but each element is "None"
            v = res.prim_inf_cert
            # Try to figure out which constraints are being violated
            PRIM_INFEAS_VIOL_THRESH = 1e-2
            v[np.abs(v) < PRIM_INFEAS_VIOL_THRESH] = 0
            print('Constraints corresponding to infeasibility:')
            numxdyn = (self.N + 1) * self.nx
            numxcon = numxdyn + (self.N + 1) * self.nx
            for iviol in np.flatnonzero(v):
                if iviol < numxdyn:
                    print('Dynamics ti=', int(iviol/self.nx))
                elif iviol < numxcon:
                    print('Constraint x', int((iviol - numxdyn)/self.nx), ',', (iviol - numxdyn) % self.nx)
                else:
                    print('Constraint u', int((iviol - numxcon)/self.nu), ',', (iviol - numxcon) % self.nu)
                    
        elif res.info.status == 'solved inaccurate':
            # The inaccurate statuses define when the optimality, primal infeasibility or dual infeasibility conditions are satisfied with tolerances 10 times larger than the ones set.
            # https://osqp.org/docs/interfaces/status_values.html
            print('Try increasing max_iter to see if it can be solved more accurately')
    
    def solve(self, Pdata, q):
        
        # Update
        t0 = time.perf_counter()
        self.prob.update(l=self.l, u=self.u, q=q, Ax=self.A.data, Px=Pdata)
        # print(A.data.shape)

        # Solve
        res = self.prob.solve()
        # measure OSQP time
        self.tqpsolve = time.perf_counter() - t0
        self.niter = res.info.iter

        # Check solver status
        if res.info.status != 'solved':
            print('Current y,u:', x0, u0)
            self.debugResult(res)
            raise ValueError(res.info.status)
        else:
            # Heuristics to detect "bad" solutions
            if self.MAX_ULIM_VIOL_FRAC is not None:
                # Check for constraint violations based on a % of what was requested to see if there are tolerance issues
                uHorizon = res.x[(self.N+1)*self.nx:]
                # pick some thresholds to consider violation. This accounts for sign of umin/umax
                umaxV = self.umax + np.absolute(self.umax) * self.MAX_ULIM_VIOL_FRAC
                uminV = self.umin - np.absolute(self.umin) * self.MAX_ULIM_VIOL_FRAC

                if np.any(np.amax(uHorizon) > umaxV) or np.any(np.amin(uHorizon) < uminV):
                    # 
                    Ax = self.A @ res.x
                    # Try to adaptively refine precision to avoid this happening again. also see https://github.com/oxfordcontrol/osqp/issues/125
                    worstViolation = max(np.amax(Ax - self.u), np.amax(self.l - Ax))
                    newEps = 1e-2 * worstViolation  # tolerance is related to eps like this https://osqp.org/docs/solver/index.html#convergence
                    print('[mpc] warning: violated ulim ratio. Worst violation: ', worstViolation, 'new eps =', newEps)
                    self.prob.update_settings(eps_rel=newEps, eps_abs=newEps)

        return res.x

if __name__ == "__main__":
    print("Testing LTVSystem")
    from models.pendulums import Pendulum
    # import pen

    model = Pendulum()
    ltvs = LTVSystem(model)




