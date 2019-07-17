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

# Options for costMode
FINAL = 0
TRAJ = 1

# Prototype for quadratic objective functions
class QOFTrajectoryError:
    def __init__(self, wx, wu, kdamping=0):
        # store kdamping as a vector
        if isinstance(kdamping, list):
            self.kdamping = kdamping
        else:
            self.kdamping = np.full(len(wu), kdamping)
        
        # Objective function
        self.Q = sparse.diags(wx)
        self.QN = self.Q
        self.R = sparse.diags(wu)

        self.firstTime = True

    def getPq(self, xtraj, xr=None, ur=None, trajMode=ITERATE_TRAJ, costMode=TRAJ, u0=None, udamp=None, ugoalCost=False):
        """Must return P (Hessian in sparse form), q (vector Jacobian)"""
        N = xtraj.shape[0]
        nx = self.Q.shape[0]
        nu = self.R.shape[0]

        # FIXME: combine these
        if self.firstTime:
            # - quadratic objective
            self.P = sparse.block_diag([sparse.kron(sparse.eye(N), self.Q), self.QN, self.R + sparse.diags(self.kdamping), sparse.kron(sparse.eye(N - 1), self.R)]).tocsc()
            # After eliminating zero elements, since P is diagonal, the sparse format is particularly simple. If P is np x np, ...
            self.P.eliminate_zeros()
            szP = self.P.shape[0]
            # the data array just corresponds to the diagonal elements
            assert (self.P.indices == range(szP)).all()
            assert (self.P.indptr == range(szP + 1)).all() 
            # print(P.toarray(), P.data)

            # make up single goal xr for now - will get updated. NOTE: q is not sparse so we can update all of it
            xr = np.zeros(nx)
            self.q = np.hstack([np.kron(np.ones(N), -self.Q @ xr), -self.QN @ xr, np.zeros(N*nu)])  # -uprev * damping goes in the u0 slot

            self.firstTime = False
        else:
            # Single point goal goes in cost (replaced below)
            if ur is None:
                ur = np.zeros(nu)
            self.q = np.hstack([np.kron(np.ones(N), -self.Q @ xr), -self.QN @ xr, np.kron(np.ones(N), -self.R @ ur)])
            if ugoalCost and u0 is not None:
                uoffs = (N+1)*nx  # offset into q where u0, u1, etc. terms appear
                for ii in range(u0.shape[0]):
                    self.q[uoffs + ii*nu:uoffs + (ii+1)*nu] = -self.R @ u0[ii,:]

            for ti in range(N):
                # FIXME: need to look through and make sense of these modes
                if costMode == TRAJ and (trajMode == GIVEN_POINT_OR_TRAJ) or trajMode in [ITERATE_TRAJ, PREV_SOL_TRAJ]:
                    # cost along each point
                    self.q[nx * ti:nx * (ti + 1)] = -self.Q @ xtraj[ti, :nx]

            # add "damping" by penalizing changes from uprev to u0
            self.q[-N*nu:-(N-1)*nu] -= np.multiply(self.kdamping, udamp)
        return self.P, self.q
        
    def updateWeights(self, wx=None, wu=None):
        if wx is not None:
            # diagonal elements of P
            self.P.data[:(self.N + 1)*self.nx] = np.tile(wx, self.N + 1)
        if wu is not None:
            self.P.data[-self.N*self.nu:] = np.hstack((self.kdamping + np.array(wu), np.tile(wu, self.N - 1)))
            self.R = sparse.diags(wu)


class LTVDirTran:
    """This deals with the (LTV) dynamics, constraints, and trajectory
    
    NOTE: Dimensions ---
    Ad = nx*nx, Bd = nx*nu
    Ax = (N+1)*nx * (N+1)*nx
    Bu = (N+1)*nx * N*nu
    Aeq = (N+1)*nx * ((N+1)*nx+N*nu)
    x = [y0, y1, ..., yN, u1, ..., uN] of size (N+1)*nx + N*nu
    After solving, apply u1 (MPC)
    """

    def __init__(self, model):
        self.m = model
        self.tqpsolve = np.nan
        self.niter = np.nan

    def initConstraints(self, nx, nu, N, useModelLimits=True, **kwargs):
        """Return A, l, u"""
        self.nx = nx
        self.nu = nu
        self.N = N
        self.polyBlocks = kwargs.get('polyBlocks', None)
        periodic = kwargs.get('periodic', False)
        stateLim = kwargs.get('stateLim', True)
        inputLim = kwargs.get('inputLim', True)

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
        conA = csc.init(nx, nu, N, **kwargs)
        self.A = sparse.csc_matrix((conA.data, conA.indices, conA.indptr))
        
        x0 = np.zeros(nx) # will get updated
        leq = np.hstack([-x0, np.zeros(N*self.nx)])

        if periodic:
            # -x0 + xN = 0
            leq = np.hstack((leq, np.zeros(self.nx)))

        ueq = leq

        # - input and state constraints
        if stateLim or inputLim:
            if stateLim and inputLim:
                lineq = np.hstack([np.kron(np.ones(N+1), self.xmin), np.kron(np.ones(N), self.umin)])
                uineq = np.hstack([np.kron(np.ones(N+1), self.xmax), np.kron(np.ones(N), self.umax)])
            elif stateLim:
                lineq = np.kron(np.ones(N+1), self.xmin)
                uineq = np.kron(np.ones(N+1), self.xmax)
            elif inputLim:
                lineq = np.kron(np.ones(N), self.umin)
                uineq = np.kron(np.ones(N), self.umax)
            self.l = np.hstack([leq, lineq])
            self.u = np.hstack([ueq, uineq])
        else:
            self.l = leq
            self.u = ueq
        
        if self.polyBlocks is not None:
            # Add corresponding RHS elements for polyhedron membership
            # Only C x <= d type constraints
            Nctotal = self.A.shape[0] - len(self.l)
            self.l = np.hstack((self.l, np.full(Nctotal, -np.inf)))
            self.u = np.hstack((self.u, np.full(Nctotal, np.inf)))
        
        # Array to store traj in x and u
        self.xtraj = np.zeros((self.N, self.nx + self.nu))  # +1 to store the last x
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

    def updateTrajectory(self, x0, u0, trajMode, *args):
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
        dyn = self.m.getLinearDynamics(xlin, ulin, *args)
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
                dyn = self.m.getLinearDynamics(xlin, ulin, *args)
                Ad, Bd = dyn[0:2]
            
            # Place new Ad, Bd in Aeq
            numUpdated = csc.updateDynamics(self.A, self.N, ti, Ad=Ad, Bd=Bd)
            # Debugging: make sure both A, B were updated
            assert numUpdated == self.nx * (self.nx + self.nu)
            # Update RHS of constraint
            fd = dyn[2] if bAffine else np.zeros(self.nx)
            self.l[self.nx * (ti+1) : self.nx * (ti+2)] = -fd
            self.u[self.nx * (ti+1) : self.nx * (ti+2)] = -fd
            # /dynamics update --
            self.xtraj[ti, :] = np.hstack((xlin, ulin))

        return self.xtraj

    def initObjective(self, qof, **kwargs):
        self.qof = qof
        # Call once with the zero initial traj to set things up
        self.P, self.q = self.qof.getPq(self.xtraj, **kwargs)

    def updateObjective(self, **kwargs):
        # Call once with the zero initial traj to set things up
        self.P, self.q = self.qof.getPq(self.xtraj, **kwargs)
        

class LTVSolver(LTVDirTran):
    """Combine LTV dynamics with the solver"""
    
    # Parameters that could be modified --
    # If returned u violated umin/umax by a fraction (this is useful for systems with small input magnitudes and large tolerance)
    MAX_ULIM_VIOL_FRAC = None
    # /parameters

    def initSolver(self, **settings):
        # Create an OSQP object
        self.prob = osqp.OSQP()
        # Full so that it is not made sparse. prob.update() cannot change the sparsity structure
        # Setup workspace
        self.prob.setup(self.P, self.q, self.A, self.l, self.u, warm_start=True, **settings)#, eps_abs=1e-05, eps_rel=1e-05
        

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
    
    def solve(self, throwOnError=True):
        """throwOnError = true => Will throw if status is not solved"""
        # Update
        t0 = time.perf_counter()
        self.prob.update(l=self.l, u=self.u, q=self.q, Ax=self.A.data, Px=self.P.data)
        # print(A.data.shape)

        # Solve
        res = self.prob.solve()
        # measure OSQP time
        self.tqpsolve = time.perf_counter() - t0
        self.niter = res.info.iter

        # Check solver status
        if res.info.status != 'solved' and throwOnError:
            # print('Current y,u:', x0, u0)
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

        if throwOnError:
            return res.x
        else:
            return res.x, res

if __name__ == "__main__":
    print("Testing LTVSystem")
    from models.pendulums import Pendulum
    # import pen

    model = Pendulum()
    ltvs = LTVSystem(model)




