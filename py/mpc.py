import numpy as np
import scipy.sparse as sparse # for testing
import osqp
import sys

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

# What to use over the horizon during the MPC update ---
# If a single goal point is given, use dynamics at that, or if a trajectory is given
# use the linearization at each knot point
GIVEN_POINT_OR_TRAJ = 0
# ITERATE_TRAJ_LIN = 1
ITERATE_TRAJ = 1
# sequential QP: take the first solution, get a traj, and then linearize around that and rinse repeat (before actually taking a control action in the sim)
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

	def __init__(self, model, N, wx, wu, kdamping=0, polyBlocks=[], **settings):
		'''
		model = model class with dynamics(), getLinearDynamics(), getLimits()
		N = prediction horizon
		wx,wu = initial weight on states,inputs (update with updateWeights)
		kdamping = weight for kdamping * || u - uprev ||^2 term added to objective
		polyBlocks = blocks of x with polyhedron constraints (vs. the default of Identity)
		'''
		self.m = model
		self.N = N
		self.kdamping = kdamping

		# Constraints
		umin, umax, xmin, xmax = model.getLimits()

		# Create an OSQP object
		self.prob = osqp.OSQP()
		
		# Objective function
		self.Q = sparse.diags(wx)
		self.QN = self.Q
		R = sparse.diags(wu)

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
		conA = CondensedA(self.m.nx, self.m.nu, N, polyBlocks=polyBlocks)
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
		
		# Full so that it is not made sparse. prob.update() cannot change the sparsity structure
		# Setup workspace
		self.prob.setup(self.P, q, self.A, self.l, self.u, warm_start=True, **settings)#, eps_abs=1e-05, eps_rel=1e-05
		self.ctrl = np.zeros(self.m.nu)

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

	def updateWeights(self, wx=None, wu=None):
		if wx is not None:
			# diagonal elements of P
			self.P.data[:(self.N + 1)*self.m.nx] = np.tile(wx, self.N + 1)
		if wu is not None:
			self.P.data[-self.N*self.m.nu:] = np.hstack((np.full(self.m.nu, self.kdamping) + np.array(wu), np.tile(wu, self.N - 1)))

	def update(self, x0, u0, xr, wx=None, wu=None, trajMode=GIVEN_POINT_OR_TRAJ, costMode=TRAJ):
		'''trajMode should be one of the constants in the group up top.
		'''
		
		if trajMode == SQP:
			raise 'Not implemented'
		
		# If there is no trajectory, then the cost can only take the final point
		if trajMode == GIVEN_POINT_OR_TRAJ and len(x0.shape) == 1 and costMode == TRAJ:
			costMode = FINAL

		self.updateWeights(wx, wu)  # they can be none

		# Update the initial state
		xlin = x0[0,:] if len(x0.shape) > 1 else x0
		ulin = u0[0,:] if len(u0.shape) > 1 else u0
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
				
			if ti > 0 and (trajMode == GIVEN_POINT_OR_TRAJ and len(x0.shape) > 1) or trajMode in [ITERATE_TRAJ]:
				# if a trajectory is provided or projected, need to get the newest linearization
				dyn = self.m.getLinearDynamics(xlin, ulin)
				Ad, Bd = dyn[0:2]
			
			# Place new Ad, Bd in Aeq
			cscUpdateDynamics(self.A, self.N, ti, Ad=Ad, Bd=Bd)
			# Update RHS of constraint
			fd = dyn[2] if bAffine else np.zeros(self.m.nx)
			self.l[self.m.nx * (ti+1) : self.m.nx * (ti+2)] = -fd
			self.u[self.m.nx * (ti+1) : self.m.nx * (ti+2)] = -fd
			# /dynamics update --

			# Objective update in q --
			if costMode == TRAJ:
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

		# Apply first control input to the plant
		self.ctrl = res.x[-self.N*self.m.nu:-(self.N-1)*self.m.nu]

		# if self.ctrl[0] < -1e-6:


		return self.ctrl


'''
These are all for the condensed A matrix
CSC matrix
three NumPy arrays: indices, indptr, data
In OSQP: sparse structure cannot be changed, i.e. can only change data
Ax_new: Vector of new elements in A->x
Ax_new_idx: Index mapping new elements to positions in A->x
A->x must be the data vector
Assuming that indices, indptr can't be changed
'''

class CondensedA:

	def __init__(self, nx, nu, N, polyBlocks=[], val=0):
		# Pass in nx=#states, nu=#inputs, and N=prediction horizon
		self.nx = nx
		self.nu = nu
		self.N = N

		# create the indptr and index arrays for sparse CSC format
		self.data = np.zeros((self.nnz))
		self.indices = np.zeros_like(self.data, dtype=np.int32)
		numcols = self.shape[1]
		self.indptr = np.zeros((numcols + 1), dtype=np.int32)

		# NOTE: in the code below, i refers to row index, j to col index
		# similar for bj for block col index
		
		# populate indptr
		si = 0
		for j in range(numcols):
			if j < self.N * self.nx:
				si += nx + 2
			elif j < (self.N + 1) * self.nx:
				si += 2
			else:
				si += self.nx + 1
			self.indptr[j+1] = si
		
		# populate row indices
		self._offs = -1 # internal val to count through the data vector
		for j in range(self.shape[1]):
			# In each column, the Aeq blocks come first
			if j < (self.N + 1) * self.nx:
				# Leftmost cols
				bj = int(np.floor(j / self.nx)) # block col index - goes from [0,N+1)
				# the block diagonal -I
				self.addNZ(j, j, -1)
					# The sub-block-diagonal Ad under the -I -- only in the left column block
				if j < self.N * self.nx:
					for i in range(self.nx * (bj + 1), self.nx * (bj + 2)):
						self.addNZ(j, i, val)
			else:
				# Right block column
				bj = int(np.floor((j - (self.N + 1) * self.nx) / self.nu))# block col index for the right - goes from [0,N)
				# The Bd blocks
				for i in range(self.nx * (bj + 1), self.nx * (bj + 2)):
					self.addNZ(j, i, val)

			# The identity in Aineq
			if len(polyBlocks) > 0:
				# we are at column j
				# self.addNZ(j, (self.N + 1) * self.nx + j, 1)
				raise NotImplementedError
			else:
				self.addNZ(j, (self.N + 1) * self.nx + j, 1)


			# offs = self.matrixIdxToOffset(i, j)
			# if offs >= 0:
			# 	self.indices[offs] = j

		# sparse structure is now fixed, but data can be updated
	
	def addNZ(self, j, i, val):
		self._offs += 1
		self.indices[self._offs] = i # row index
		self.data[self._offs] = val

	# def blockToMatrixIdx(self, ib, jb, i, j):
	# 	# Return matrix indices from the block index (ib, jb) and element index within the block (i, j)
	# 	if ib < 2*(N+1):
	# 		rowi = self.nx * ib + i
	# 	else:
	# 		rowi = 2 * (self.N+1) * self.nx + (ib - 2 * (self.N+1)) * self.nu + i

	# 	if jb < (N+1):
	# 		rowi = self.nx * jb + j
	# 	else:
	# 		rowi = (self.N+1) * self.nx + (jb - (self.N+1)) * self.nu + j
		
	# 	return rowi, coli

	def get_nnz(self):
		# For testing
		return self.N * self.nx * (self.nx + 2) + self.nx * 2 + self.N * self.nu * (self.nx + 1)
	def get_shape(self):
		# For testing
		return (2 * (self.N + 1) * self.nx + self.N * self.nu), (self.N + 1) * self.nx + self.N * self.nu

	nnz = property(get_nnz, None, None, "Number of nonzero entries")
	shape = property(get_shape, None, None, "Number of nonzero entries")

# These work for sparse as well as CondensedA

def cscUpdateElem(obj, i, j, val):
	'''Will update an element; do nothing if that entry is zero
	Works on scipy sparse as well as CondensedA
	'''

	# indptr has #cols elements, and the entry is the index into data/indices for the first element of that column
	offs = obj.indptr[j]
	# 
	if j < obj.shape[1] - 1:
		maxOffsThisCol = obj.indptr[j+1]
	else:
		maxOffsThisCol = len(obj.indices)
	while offs < maxOffsThisCol:
		if obj.indices[offs] == i:
			obj.data[offs] = val
			return
		offs += 1

def cscUpdateDynamics(obj, N, ti, Ad=None, Bd=None):
	'''Pass a block to update, and a matrix to go in that block, and it will update all the elements in that block.
	At ti=0, this updates the block equation for x1 = Ad x0 + Bd u0 [+ fd].
	'''

	assert(ti < N)
	
	if Ad is not None:
		nx = Ad.shape[0]
		for i in range(nx):
			for j in range(nx):
				cscUpdateElem(obj, nx * (ti + 1) + i, nx * ti + j, Ad[i,j])

	if Bd is not None:
		nx, nu = Bd.shape
		for i in range(nx):
			for j in range(nu):
				cscUpdateElem(obj, nx * (ti + 1) + i, (N + 1) * nx + nu * ti + j, Bd[i,j])
		

if __name__ == "__main__":
	print("Testing CondensedA...")

	#
	nx = 3
	nu = 2
	N = 4
	# create instance
	conA = CondensedA(nx, nu, N, val=1, polyBlocks=[[4,5],[6,7]])
	# wider display
	# np.set_printoptions(edgeitems=30, linewidth=100000)
	
	# Create dense and then use scipy.sparse
	Ad = np.ones((nx, nx))
	Bd = np.ones((nx, nu))
	# Ad, Bd = getLin(x0, ctrl, dt)
	Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
	Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
	Aeq = sparse.hstack([Ax, Bu])
	Aineq = sparse.block_diag((sparse.kron(sparse.eye(N+1), sparse.eye(nx)), sparse.eye(N*nu)))
	A = sparse.vstack([Aeq, Aineq]).tocsc()

	# Tests ---

	assert(conA.shape == A.shape)
	assert(conA.nnz == A.nnz)
	assert((conA.indptr == A.indptr).all())
	assert((conA.indices == A.indices).all())
	# both were filled with ones
	assert((conA.data == A.data).all())

	#  Usage: update blocks on a sparse, and a CondensedA, and compare to a newly created sparse
	Ad2 = np.full((nx, nx), 123)
	Bd2 = np.full((nx, nu), -456)
	Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad2)
	Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd2)
	Aeq = sparse.hstack([Ax, Bu])
	A2 = sparse.vstack([Aeq, Aineq]).tocsc()
	for ti in range(N):
		cscUpdateDynamics(A, N, ti, Ad=Ad2, Bd=Bd2)
		cscUpdateDynamics(conA, N, ti, Ad=Ad2, Bd=Bd2)
	# test update
	assert((conA.data == A2.data).all())
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
