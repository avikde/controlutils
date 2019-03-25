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
			Cpolyx = sparse.vstack((Cpolyx, Cconstraint))
		A = sparse.vstack((A,
			# left block corresponds to x0,...,xN, right block to u0,...,u(N-1)
			sparse.hstack((sparse.kron(sparse.eye(N+1), Cpolyx), np.zeros(((N+1)*Cpolyx.shape[0], N*nu))))
		))

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

	assert ti < N
	
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
	
def updatePolyBlock(obj, nx, nu, N, ti, polyBlocks, pbi, Cdi):
	'''Updates a block in the lower left (Aineq) with a matrix denoting a polyhedron.
	polyBlocks = Provide the same polyBlocks used in csc.init
	pbi = index into polyBlocks list
	Cdi = (Nc,Ncx)-shaped matrix to use to replace the existing block
	'''
	assert ti <= N
	assert pbi < len(polyBlocks), "index too big for polyBlocks list"
	NcPerTi = 0
	NcBeforei = 0
	# Get information needed about all the constraints
	for ii in range(len(polyBlocks)):
		xstart, Nc, Ncx = polyBlocks[ii]
		NcPerTi += Nc
		if ii < pbi:
			NcBeforei += Nc
		elif ii == pbi:
			xstartj = xstart
			Nci = Nc
			Ncxi = Ncx

	# Lots of error checking
	assert Cdi.shape[0] == Nci, "Cdi shape is wrong"
	assert Cdi.shape[1] == Ncxi, "Cdi shape is wrong"

	ioffs = 2*(N+1)*nx + N*nu  # stanard Aeq,Aineq size of condensed A before polyBlocks
	ioffs += ti * NcPerTi + NcBeforei
	joffs = nx * ti + xstartj
	
	for i in range(Nci):
		for j in range(Ncxi):
			updateElem(obj, ioffs + i, joffs + j, Cdi[i, j])
	
	return ioffs  # to be helpful
