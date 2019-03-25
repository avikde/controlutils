import numpy as np
import scipy.sparse as sparse

# These work for sparse as well as CondensedA

def init(nx, nu, N, polyBlocks=None):
	'''Return scipy sparse.
	If polyBlocks is specified, additional polyhedron membership constraints are added at the bottom.
	The size of these needs to be fixed to maintain the sparsity structure (i.e. Nc-sized polyhedron).
	polyBlocks = [polyBlock1, ...], where
	polyBlock = [xstart, Nc, Ncx], where
	Apoly @ x[xstart : xstart+Ncx] <= upoly,
	with both sides having Nc rows (# constraints)
	'''
	Ad = np.ones((nx, nx))
	Bd = np.ones((nx, nu))
	# Ad, Bd = getLin(x0, ctrl, dt)
	Ax = sparse.kron(sparse.eye(N+1),-sparse.eye(nx)) + sparse.kron(sparse.eye(N+1, k=-1), Ad)
	Bu = sparse.kron(sparse.vstack([sparse.csc_matrix((1, N)), sparse.eye(N)]), Bd)
	Aeq = sparse.hstack([Ax, Bu])
	Aineq = sparse.block_diag((sparse.kron(sparse.eye(N+1), np.eye(nx)), sparse.eye(N*nu)))
	A = sparse.vstack([Aeq, Aineq])
	# Add additional constraints for polyhedra
	if polyBlocks is not None:
		Cpolyx = np.zeros((0,nx))  # will stack rows below for each constraint
		for polyBlock in polyBlocks:
			# First get the LHS matrix corresponding to a single xk
			xstart, Nc, Ncx = polyBlock
			Cconstraint = np.tile(np.hstack([np.zeros(xstart), np.ones(Ncx), np.zeros(nx-(Ncx + xstart))]), (Nc,1))
			Cpolyx = np.vstack((Cpolyx, Cconstraint))
		# A = sparse.vstack((A,
		# 	np.tile( np.hstack([np.zeros(xstart), np.ones(Ncx), np.zeros(nx-(Ncx + xstart))]), (Nc, 1))
		# ))
		print(Cpolyx)
		raise NotImplementedError
		# Aineqx = []
		# for polyBlock in polyBlocks:
		# 	Aineqx.append(np.full((polyBlock, polyBlock), 1))
		# Aineqx = sparse.block_diag(Aineqx)

	else:
		return A.tocsc()


def updateElem(obj, i, j, val):
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

def updateDynamics(obj, N, ti, Ad=None, Bd=None):
	'''Pass a block to update, and a matrix to go in that block, and it will update all the elements in that block.
	At ti=0, this updates the block equation for x1 = Ad x0 + Bd u0 [+ fd].
	'''

	assert(ti < N)
	
	if Ad is not None:
		nx = Ad.shape[0]
		for i in range(nx):
			for j in range(nx):
				updateElem(obj, nx * (ti + 1) + i, nx * ti + j, Ad[i,j])

	if Bd is not None:
		nx, nu = Bd.shape
		for i in range(nx):
			for j in range(nu):
				updateElem(obj, nx * (ti + 1) + i, (N + 1) * nx + nu * ti + j, Bd[i,j])
	
def updatePolyBlock(obj, N, ti, nx, xstart, Cdi):
	'''Updates a block in the lower left (Aineq) with a matrix denoting a polyhedron.
	xstart can be a scalar (both row and col start same) or tuple
	Cdi must be a matrix. If updating a 1x1 block, supply np.full((1,1), val)
	'''
	if isinstance(xstart, list):
		xstarti, xstartj = xstart[0], xstart[1]
	else:
		xstarti = xstartj = xstart
	# Cdi = np.array(Cdi)
	for i in range(xstarti, xstarti + Cdi.shape[0]):
		for j in range(xstartj, xstartj + Cdi.shape[1]):
			updateElem(obj, (N + 1 + ti) * nx + i, nx * ti + j, Cdi[i-xstarti,j-xstartj])
