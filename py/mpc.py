import autograd.numpy as np
import scipy.sparse as sparse # for testing
import sys, time
try:
    from . import csc, ltvsystem
except:
    import csc, ltvsystem


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
        self.ltvsys = ltvsystem.LTVSolver(model)
        
        nx = len(wx)
        nu = len(wu)
        
        # QP state: x = (y(0),y(1),...,y(N),u(0),...,u(N-1))
        # Dynamics and constraints
        self.ltvsys.initConstraints(nx, nu, N, polyBlocks=polyBlocks)
        self.ltvsys.initObjective(ltvsystem.QOFTrajectoryError(wx, wu, kdamping))
        self.ltvsys.initSolver(**settings)

        # Variables to store the previous result in
        self.ctrl = np.zeros(nu)
        self.prevSol = None
    
    def _sanitizeTrajAndCostModes(self, trajMode, costMode, x0):
        '''Test if these modes all make sense'''

        if trajMode == ltvsystem.SQP:
            raise 'Not implemented'
        
        # if this is the first time, cannot use the previous solution
        if self.prevSol is None and trajMode == ltvsystem.PREV_SOL_TRAJ:
            trajMode = ltvsystem.GIVEN_POINT_OR_TRAJ

        # If there is no trajectory, then the cost can only take the final point
        if trajMode == ltvsystem.GIVEN_POINT_OR_TRAJ and len(x0.shape) == 1 and costMode == ltvsystem.TRAJ:
            costMode = ltvsystem.FINAL
            
        return trajMode, costMode


    def update(self, x0, xr, *args, u0=None, ur=None, trajMode=ltvsystem.GIVEN_POINT_OR_TRAJ, costMode=ltvsystem.TRAJ, ugoalCost=False):
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
        
        t0 = time.perf_counter()
        xtraj = self.ltvsys.updateTrajectory(x0, u0, trajMode, *args)
        t1 = time.perf_counter()
        self.ltvsys.updateObjective(xr=xr, ur=ur, trajMode=trajMode, costMode=costMode, u0=u0, udamp=self.ctrl, ugoalCost=ugoalCost)
        t2 = time.perf_counter()
        self.prevSol = self.ltvsys.solve()

        self.timings = {'updateTrajectory': (t1-t0)*1e3, 'updateObjective': (t2-t1)*1e3, 'qpsolve': (self.ltvsys.tqpsolve)*1e3}
        # Apply first control input to the plant, and store
        self.ctrl = self.prevSol[-self.ltvsys.N*self.ltvsys.nu:-(self.ltvsys.N-1)*self.ltvsys.nu]

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
