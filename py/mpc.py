import numpy as np
import scipy.sparse as sparse # for testing
import osqp
import sys
try:
	from . import csc
except:
	import csc

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

# What to use over the horizon during the MPC update (see docstring of update) ---
GIVEN_POINT_OR_TRAJ = 0
ITERATE_TRAJ = 1
PREV_SOL_TRAJ = 2
SQP = 4

# Options for costMode
FINAL = 0
TRAJ = 1
# /CONSTANTS --


class LTVMPC:
	'''Interface that the provided "model" must provide:
	- getLinearDynamics(y, u)
	- dynamics(y, u) - if ITERATE_TRAJ is selected
	'''
	
	# Parameters that could be modified --
	# If returned u violated umin/umax by a fraction (this is useful for systems with small input magnitudes and large tolerance)
	MAX_ULIM_VIOL_FRAC = None
	# /parameters

	def __init__(self, model, N, wx, wu, kdamping=0, polyBlocks=None, **settings):
		'''
		model = model class with dynamics(), getLinearDynamics(), getLimits()
		N = prediction horizon
		wx,wu = initial weight on states,inputs (update with updateWeights)
		kdamping = weight for kdamping * || u - uprev ||^2 term added to objective
		polyBlocks: see csc.init for help on specifying polyBlocks
		'''
		self.m = model
		self.N = N
		self.kdamping = kdamping
		self.polyBlocks = polyBlocks

		# Constraints
		umin, umax, xmin, xmax = model.getLimits()

		# Create an OSQP object
		self.prob = osqp.OSQP()
		
		# Objective function
		self.Q = sparse.diags(wx)
		self.QN = self.Q
		R = sparse.diags(wu)

		assert len(wx) == self.m.nx
		assert len(wu) == self.m.nu

		# Cast MPC problem to a QP: x = (x(0),x(1),...,x(N),u(0),...,u(N-1))
		# - quadratic objective
		self.P = sparse.block_diag([sparse.kron(sparse.eye(self.N), self.Q), self.QN, R + sparse.eye(self.m.nu) * kdamping, sparse.kron(sparse.eye(N - 1), R)]).tocsc()
		# After eliminating zero elements, since P is diagonal, the sparse format is particularly simple. If P is np x np, ...
		self.P.eliminate_zeros()
		szP = self.P.shape[0]
		# the data array just corresponds to the diagonal elements
		assert((self.P.indices == range(szP)).all())
		assert((self.P.indptr == range(szP + 1)).all())
		# print(P.toarray(), P.data)
		# - input and state constraints
		lineq = np.hstack([np.kron(np.ones(N+1), xmin), np.kron(np.ones(N), umin)])
		uineq = np.hstack([np.kron(np.ones(N+1), xmax), np.kron(np.ones(N), umax)])

		# Create the CSC A matrix manually.
		# conA = CondensedA(self.m.nx, self.m.nu, N, polyBlocks=polyBlocks)
		conA = csc.init(self.m.nx, self.m.nu, N, polyBlocks=polyBlocks)
		self.A = sparse.csc_matrix((conA.data, conA.indices, conA.indptr))
		# make up single goal xr for now - will get updated. NOTE: q is not sparse so we can update all of it
		xr = np.zeros(self.m.nx)
		x0 = np.zeros_like(xr)
		q = np.hstack([np.kron(np.ones(N), -self.Q.dot(xr)), -self.QN.dot(xr), np.zeros(N*self.m.nu)])  # -uprev * damping goes in the u0 slot
		# Initial state will get updated
		leq = np.hstack([-x0, np.zeros(self.N*self.m.nx)])
		ueq = leq
		self.l = np.hstack([leq, lineq])
		self.u = np.hstack([ueq, uineq])
		if polyBlocks is not None:
			# Add corresponding RHS elements for polyhedron membership
			# Only C x <= d type constraints
			Nctotal = self.A.shape[0] - len(self.l)
			self.l = np.hstack((self.l, np.full(Nctotal, -np.inf)))
			self.u = np.hstack((self.u, np.full(Nctotal, np.inf)))
		
		# Full so that it is not made sparse. prob.update() cannot change the sparsity structure
		# Setup workspace
		self.prob.setup(self.P, q, self.A, self.l, self.u, warm_start=True, **settings)#, eps_abs=1e-05, eps_rel=1e-05

		# Variables to store the previous result in
		self.ctrl = np.zeros(self.m.nu)
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
			numxdyn = (self.N + 1) * self.m.nx
			numxcon = numxdyn + (self.N + 1) * self.m.nx
			for iviol in np.flatnonzero(v):
				if iviol < numxdyn:
					print('Dynamics ti=', int(iviol/self.m.nx))
				elif iviol < numxcon:
					print('Constraint x', int((iviol - numxdyn)/self.m.nx), ',', (iviol - numxdyn) % self.m.nx)
				else:
					print('Constraint u', int((iviol - numxcon)/self.m.nu), ',', (iviol - numxcon) % self.m.nu)
		# print(res.x.shape)
		# # print((Ax - self.l)[-self.N*self.m.nu:-(self.N-1)*self.m.nu])
		# # print((self.u - Ax)[-self.N*self.m.nu:-(self.N-1)*self.m.nu])
		# print(np.min(Ax - self.l))
		# print(np.min(self.u - Ax))
		# print("HELLO")
	
	def updatePolyhedronConstraint(self, ti, pbi, Ci, di):
		'''Update the polyhedron membership constraint for time i
		The constraint added is Ci projection_pbi( x(ti) ) <= di
		pbi = index into polyBlocks provided during init
		To "remove" the constraint, just set di = np.full(*, np.inf)
		'''
		# Only C x <= d type constraints, so only change A and u
		ioffs = csc.updatePolyBlock(self.A, self.m.nx, self.m.nu, self.N, ti, self.polyBlocks, pbi, Ci)
		# update u, but not l (which stays at -inf)
		assert(Ci.shape[0] == len(di))
		self.u[ioffs : ioffs + len(di)] = di

	def updateWeights(self, wx=None, wu=None):
		if wx is not None:
			# diagonal elements of P
			self.P.data[:(self.N + 1)*self.m.nx] = np.tile(wx, self.N + 1)
		if wu is not None:
			self.P.data[-self.N*self.m.nu:] = np.hstack((np.full(self.m.nu, self.kdamping) + np.array(wu), np.tile(wu, self.N - 1)))
	
	def _sanitizeTrajAndCostModes(self, trajMode, costMode, x0):
		'''Test if these modes all make sense'''

		if trajMode == SQP:
			raise 'Not implemented'
		
		# if this is the first time, cannot use the previous solution
		if self.prevSol is None and trajMode == PREV_SOL_TRAJ:
			trajMode = GIVEN_POINT_OR_TRAJ

		# If there is no trajectory, then the cost can only take the final point
		if trajMode == GIVEN_POINT_OR_TRAJ and len(x0.shape) == 1 and costMode == TRAJ:
			costMode = FINAL
			
		return trajMode, costMode


	def update(self, x0, xr, u0=None, trajMode=GIVEN_POINT_OR_TRAJ, costMode=TRAJ):
		'''trajMode should be one of the constants in the group up top.

		x0 --- either a (nx,) vector (current state) or an (N,nx) array with a given horizon (used with `trajMode=GIVEN_POINT_OR_TRAJ`). NOTE: when a horizon is provided, x0[0,:] must still have the current state.

		xr --- goal state (placed in the linear term of the objective)

		u0 --- input to linearize around (does not matter in a lot of cases if the linearization does not depend on the input).

		trajMode --- 

		`GIVEN_POINT_OR_TRAJ`: if a single goal point is given in x0, use linearization at that point, or if a trajectory is given in x0, use the linearization at each knot point

		`ITERATE_TRAJ`: apply the current or provided input u0 recursively to produce a trajectory. Warning: this can produce bad trajectories for complicated dynamics

		`PREV_SOL_TRAJ`: Use the x1, ..., xN from the previous run

		`SQP` i.e. sequential QP: take the first solution, get a traj, and then linearize around that and rinse repeat (before actually taking a control action in the sim)
		
		costMode --- if FINAL, only xr (at the end of the horizon) contributes to the cost. If TRAJ, each point in the provided or constructed trajectory (see trajMode) contributes.

		Returns: the next control input u0
		'''
		trajMode, costMode = self._sanitizeTrajAndCostModes(trajMode, costMode, x0)

		if u0 is None:
			u0 = self.ctrl  # use the previous control input
		# Update the initial state
		xlin = x0[0,:] if len(x0.shape) > 1 else x0
		ulin = u0[0,:] if len(u0.shape) > 1 else u0
		# Assumes that when a trajectory is provided, the first is the initial condition (see docstring)
		self.l[:self.m.nx] = -xlin
		self.u[:self.m.nx] = -xlin
		
		# Single point goal goes in cost (replaced below)
		q = np.hstack([np.kron(np.ones(self.N), -self.Q.dot(xr)), -self.QN.dot(xr), np.zeros(self.N*self.m.nu)])
		
		# First dynamics
		dyn = self.m.getLinearDynamics(xlin, ulin)
		Ad, Bd = dyn[0:2]
		bAffine = len(dyn) > 2
			
		for ti in range(self.N):
			# Dynamics update in the equality constraint (also set xlin) --
			# Get linearization point if it is provided
			if trajMode == GIVEN_POINT_OR_TRAJ and len(x0.shape) > 1:
				xlin = x0[i, :]
				ulin = u0[i, :]
			# Get linearization point by iterating a provided input
			elif trajMode in [ITERATE_TRAJ] and ti > 0:
				# update the point at which the next linearization will happen
				xlin = self.m.dynamics(xlin, ulin)
			elif trajMode == PREV_SOL_TRAJ:
				# use the previous solution shifted by 1 timestep (till the end)
				# NOTE: uses the end point twice (probably matters least then anyway)
				prevSolIdx = self.m.nx * min(ti+1, self.N)  # index into prevSol
				xlin = self.prevSol[prevSolIdx : prevSolIdx + self.m.nx]
				
			if ti > 0 and (trajMode == GIVEN_POINT_OR_TRAJ and len(x0.shape) > 1) or trajMode in [ITERATE_TRAJ, PREV_SOL_TRAJ]:
				# if a trajectory is provided or projected, need to get the newest linearization
				dyn = self.m.getLinearDynamics(xlin, ulin)
				Ad, Bd = dyn[0:2]
			
			# Place new Ad, Bd in Aeq
			csc.updateDynamics(self.A, self.N, ti, Ad=Ad, Bd=Bd)
			# Update RHS of constraint
			fd = dyn[2] if bAffine else np.zeros(self.m.nx)
			self.l[self.m.nx * (ti+1) : self.m.nx * (ti+2)] = -fd
			self.u[self.m.nx * (ti+1) : self.m.nx * (ti+2)] = -fd
			# /dynamics update --

			# Objective update in q --
			if costMode == TRAJ and (trajMode == GIVEN_POINT_OR_TRAJ and len(x0.shape) > 1) or trajMode in [ITERATE_TRAJ, PREV_SOL_TRAJ]:
				# cost along each point
				q[self.m.nx * ti:self.m.nx * (ti + 1)] = -self.Q @ xlin
			# /objective update

		# add "damping" by penalizing changes from uprev to u0
		q[-self.N*self.m.nu:-(self.N-1)*self.m.nu] = -self.kdamping * self.ctrl
			
		# Update
		self.prob.update(l=self.l, u=self.u, q=q, Ax=self.A.data, Px=self.P.data)
		# print(A.data.shape)

		# Solve
		res = self.prob.solve()

		# Check solver status
		if res.info.status != 'solved':
			print('Current y,u:', x0, u0)
			self.debugResult(res)
			raise ValueError(res.info.status)

		if self.MAX_ULIM_VIOL_FRAC is not None:
			# Check for constraint violations based on a % of what was requested to see if there are tolerance issues
			uHorizon = res.x[(self.N+1)*self.m.nx:]
			umin, umax, _, _ = self.m.getLimits()
			# FIXME: change based on signs of umin/umax?
			if np.any(np.amax(uHorizon) > (1 + self.MAX_ULIM_VIOL_FRAC) * umax) or np.any(np.amin(uHorizon) < (1 + self.MAX_ULIM_VIOL_FRAC) * umin):
				print('Warning: u viol')

		# Apply first control input to the plant, and store
		self.ctrl = res.x[-self.N*self.m.nu:-(self.N-1)*self.m.nu]
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
		assert(conA.shape == A.shape)
		assert(conA.nnz == A.nnz)
		assert((conA.indptr == A.indptr).all())
		assert((conA.indices == A.indices).all())
		# both were filled with ones
		assert((conA.data == A.data).all())

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
		assert((conA.data == A2.data).all())
	if polyBlocks is None:
		assert((A.data == A2.data).all())

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
