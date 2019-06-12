import autograd.numpy as np
import scipy.sparse as sparse # for testing
import osqp
import sys
import time
try:
    from . import csc, ltvsystem
except:
    import csc, ltvsystem

'''
NOTE: Dimensions ---
Ad = nx*nx, Bd = nx*nu
Ax = (N+1)*nx * (N+1)*nx
Bu = (N+1)*nx * N*nu
Aeq = (N+1)*nx * ((N+1)*nx+N*nu)
x = [y0, y1, ..., yN, u1, ..., uN] of size (N+1)*nx + N*nu
After solving, apply u1 (MPC)
# print(P.shape)
# print(q.shape)
# print(A.shape)
# print(l.shape)
# print(leq.shape)
# print(lineq.shape)
'''

# CONSTANTS ---

# Options for costMode
FINAL = 0
TRAJ = 1
# /CONSTANTS --


class LTVMPC:
    '''Interface that the provided "model" must provide:
    - getLinearDynamics(y, u)
    - dynamics(y, u) - if ITERATE_TRAJ is selected
    '''

    def __init__(self, model, N, wx, wu, kdamping=0, polyBlocks=None, **settings):
        '''
        model = model class with dynamics(), 
        property `limits`
        getLinearDynamics() (TBD)

        N = prediction horizon

        wx,wu = initial weight on states,inputs (update with updateWeights)

        kdamping = (diagonal) weight for (u-uprev)^T @ Kdamping @ (u-uprev) term added to objective. Either a scalar can be provided, Kdamping = kdamping*eye(nu), otherwise Kdamping = diag(kdamping)
        
        polyBlocks: see csc.init for help on specifying polyBlocks
        '''
        self.tqpsolve = np.nan
        self.niter = np.nan
        self.ltvsys = ltvsystem.LTVSystem(model)
        self.N = N
        # store dims
        self.nx = len(wx)
        self.nu = len(wu)

        # store kdamping as a vector
        if isinstance(kdamping, list):
            self.kdamping = kdamping
        else:
            self.kdamping = np.full(self.nu, kdamping)

        # Create an OSQP object
        self.prob = osqp.OSQP()
        
        # QP state: x = (y(0),y(1),...,y(N),u(0),...,u(N-1))
        # Dynamics and constraints
        x0 = np.zeros(self.nx) # Initial state will get updated
        self.ltvsys.init(self.nx, self.nu, N, x0, polyBlocks=polyBlocks)
        
        # Objective function
        self.Q = sparse.diags(wx)
        self.QN = self.Q
        self.R = sparse.diags(wu)

        # - quadratic objective
        self.P = sparse.block_diag([sparse.kron(sparse.eye(self.N), self.Q), self.QN, self.R + sparse.diags(self.kdamping), sparse.kron(sparse.eye(N - 1), self.R)]).tocsc()
        # After eliminating zero elements, since P is diagonal, the sparse format is particularly simple. If P is np x np, ...
        self.P.eliminate_zeros()
        szP = self.P.shape[0]
        # the data array just corresponds to the diagonal elements
        assert (self.P.indices == range(szP)).all()
        assert (self.P.indptr == range(szP + 1)).all() 
        # print(P.toarray(), P.data)

        # make up single goal xr for now - will get updated. NOTE: q is not sparse so we can update all of it
        xr = np.zeros_like(x0)
        q = np.hstack([np.kron(np.ones(N), -self.Q @ xr), -self.QN @ xr, np.zeros(N*self.nu)])  # -uprev * damping goes in the u0 slot
        
        # Full so that it is not made sparse. prob.update() cannot change the sparsity structure
        # Setup workspace
        self.prob.setup(self.P, q, self.ltvsys.A, self.ltvsys.l, self.ltvsys.u, warm_start=True, **settings)#, eps_abs=1e-05, eps_rel=1e-05

        # Variables to store the previous result in
        self.ctrl = np.zeros(self.nu)
        self.prevSol = None

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

    def updateWeights(self, wx=None, wu=None):
        if wx is not None:
            # diagonal elements of P
            self.P.data[:(self.N + 1)*self.nx] = np.tile(wx, self.N + 1)
        if wu is not None:
            self.P.data[-self.N*self.nu:] = np.hstack((self.kdamping + np.array(wu), np.tile(wu, self.N - 1)))
            self.R = sparse.diags(wu)
    
    def _sanitizeTrajAndCostModes(self, trajMode, costMode, x0):
        '''Test if these modes all make sense'''

        if trajMode == ltvsystem.SQP:
            raise 'Not implemented'
        
        # if this is the first time, cannot use the previous solution
        if self.prevSol is None and trajMode == ltvsystem.PREV_SOL_TRAJ:
            trajMode = ltvsystem.GIVEN_POINT_OR_TRAJ

        # If there is no trajectory, then the cost can only take the final point
        if trajMode == ltvsystem.GIVEN_POINT_OR_TRAJ and len(x0.shape) == 1 and costMode == TRAJ:
            costMode = FINAL
            
        return trajMode, costMode


    def update(self, x0, xr, u0=None, ur=None, trajMode=ltvsystem.GIVEN_POINT_OR_TRAJ, costMode=TRAJ, ugoalCost=False):
        '''
        xr --- goal state (placed in the linear term of the objective)

        ur --- input to make cost around
        
        costMode --- if FINAL, only xr (at the end of the horizon) contributes to the cost. If TRAJ, each point in the provided or constructed trajectory (see trajMode) contributes.

        ugoalCost --- if True (and u0 is not None), u0 is incorporated into the linear cost term q

        Returns: the next control input u0
        '''
        trajMode, costMode = self._sanitizeTrajAndCostModes(trajMode, costMode, x0)
        # If you want u0 to affect the u cost, it must be provided
        if u0 is None:
            ugoalCost = False
            u0 = self.ctrl  # use the previous control input
        
        xtraj = self.ltvsys.updateTrajectory(x0, u0, trajMode)
        
        # Single point goal goes in cost (replaced below)
        if ur is None:
            ur = np.zeros(self.nu)
        q = np.hstack([np.kron(np.ones(self.N), -self.Q @ xr), -self.QN @ xr, np.kron(np.ones(self.N), -self.R @ ur)])
        if ugoalCost:
            uoffs = (self.N+1)*self.nx  # offset into q where u0, u1, etc. terms appear
            if len(u0.shape) > 1:
                # print('hihihi', u0[0,:])
                for ii in range(u0.shape[0]):
                    q[uoffs + ii*self.nu:uoffs + (ii+1)*self.nu] = -self.R @ u0[ii,:]
            else:
                q[uoffs:uoffs + self.nu] = -self.R @ u0

        for ti in range(self.N):
            # Objective update in q --
            if costMode == TRAJ and (trajMode == ltvsystem.GIVEN_POINT_OR_TRAJ and len(x0.shape) > 1) or trajMode in [ltvsystem.ITERATE_TRAJ, ltvsystem.PREV_SOL_TRAJ]:
                # cost along each point
                q[self.nx * ti:self.nx * (ti + 1)] = -self.Q @ xtraj[ti, :]
            # /objective update

        # add "damping" by penalizing changes from uprev to u0
        q[-self.N*self.nu:-(self.N-1)*self.nu] -= np.multiply(self.kdamping, self.ctrl)
            
        # Update
        t0 = time.perf_counter()
        self.prob.update(l=self.ltvsys.l, u=self.ltvsys.u, q=q, Ax=self.ltvsys.A.data, Px=self.P.data)
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
            if self.ltvsys.MAX_ULIM_VIOL_FRAC is not None:
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

        # Apply first control input to the plant, and store
        self.ctrl = res.x[-self.N*self.nu:-(self.N-1)*self.nu]
        self.prevSol = res.x

        return self.ctrl


if __name__ == "__main__":
    print("Testing csc stuff...")
    testConA = False

    #
    nx = 3
    nu = 2
    N = 4
    polyBlocks = [[0,3,1],[1,4,2]]  # see csc.init for help on specifying polyBlocks
    # create instance
    if testConA:
        conA = csc.CondensedA(nx, nu, N, val=1, polyBlocks=polyBlocks)
    # wider display
    np.set_printoptions(edgeitems=30, linewidth=100000, precision=2, threshold=2)
    
    # Create dense and then use scipy.sparse
    A = csc.init(nx, nu, N, polyBlocks=polyBlocks)

    # Tests ---
    if testConA:
        assert conA.shape == A.shape
        assert conA.nnz == A.nnz
        assert (conA.indptr == A.indptr).all()
        assert (conA.indices == A.indices).all()
        # both were filled with ones
        assert (conA.data == A.data).all()

    #  Usage: update blocks on a sparse, and a CondensedA, and compare to a newly created sparse
    Aeq = A[:(N+1)*nx, :]
    Aineq = A[(N+1)*nx:, :]
    Ad2 = np.full((nx, nx), 123)
    Bd2 = np.full((nx, nu), -456)
    Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad2)
    Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd2)
    Aeq = sparse.hstack([Ax, Bu])
    A2 = sparse.vstack([Aeq, Aineq]).tocsc()
    for ti in range(N):
        csc.updateDynamics(A, N, ti, Ad=Ad2, Bd=Bd2)
        if polyBlocks is not None:
            csc.updatePolyBlock(A, nx, nu, N, ti, polyBlocks, 1, np.full((4,2), 3 + ti))
        if testConA:
            csc.updateDynamics(conA, N, ti, Ad=Ad2, Bd=Bd2)
    # Can update the Nth polyblock
    csc.updatePolyBlock(A, nx, nu, N, N, polyBlocks, 0, np.full((3,1), 123))
    # test update
    if testConA:
        assert (conA.data == A2.data).all()
    if polyBlocks is None:
        assert (A.data == A2.data).all()

    # print(conA.indices)
    # print(A)
    # print(conA.data)
    # cscUpdateElem(conA, 6, 0, 123)
    # print(conA.data)

    # print(A.data)
    # print(conA.shape)
    # print(conA.indptr.shape)
    # # cscUpdateDynamics(A, N, 9, Bd=np.full((nx, nu), 123))
    # print(A.toarray())

    print("All passed")
