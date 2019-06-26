import autograd.numpy as np
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
    Works on scipy sparse as well as CondensedA.
    Returns 0 if nothing was updated (element was 0 in the sparse matrix), or 1 if updated successfully.
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
            return 1
        offs += 1
    return 0

def updateDynamics(obj, N, ti, Ad=None, Bd=None):
    '''Pass a block to update, and a matrix to go in that block, and it will update all the elements in that block.
    At ti=0, this updates the block equation x1 = A0 x0 + B0 u0 [+ c0], 
    at ti=N-1, x[N] = A[N-1] x[N-1] + B[N-1] u[N-1] [+ c[N-1]]
    (recall X = (x0, ..., x[N], u0, ..., u[N-1]))
    Note that since this only deals with the constrain "A" matrix, the affine part ci must be updated separately.
    '''
    assert ti < N
    retval = 0
    if Ad is not None:
        nx = Ad.shape[0]
        for i in range(nx):
            for j in range(nx):
                retval += updateElem(obj, nx * (ti + 1) + i, nx * ti + j, Ad[i,j])

    if Bd is not None:
        nx, nu = Bd.shape
        for i in range(nx):
            for j in range(nu):
                retval += updateElem(obj, nx * (ti + 1) + i, (N + 1) * nx + nu * ti + j, Bd[i,j])
    
    return retval
    
def updatePolyBlock(obj, nx, nu, N, ti, polyBlocks, pbi, Cdi):
    '''Updates a block in the lower left (Aineq) with a matrix denoting a polyhedron.
    polyBlocks = Provide the same polyBlocks used in csc.init
    pbi = index into polyBlocks list
    Cdi = (Nc,Ncx)-shaped matrix to use to replace the existing block
    '''
    if polyBlocks is None:
        return
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

    def __init__(self, nx, nu, N, polyBlocks=None, val=0):
        # Pass in nx=#states, nu=#inputs, and N=prediction horizon
        self.nx = nx
        self.nu = nu
        self.N = N
        if polyBlocks is not None:
            if np.sum(polyBlocks) != nx:
                raise ValueError('Sum of elements of polyBlocks must be = nx')
            # each element >= 1
            # all elements integers

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
            if polyBlocks is not None and j < (self.N + 1) * self.nx:
                # we are at column j
                bj = int(np.floor(j / self.nx)) # block col index - goes from [0,N+1)
                iWithinX = j - bj * self.nx
                i0 = 0
                raise NotImplementedError
                for polyBlock in polyBlocks:
                    for ii in range(polyBlock):
                        # self.addNZ(j, i0, 1)
                        i0 += 1
                # 	if iWithinX >= polyBlock[0] and iWithinX < polyBlock[1]:
                # 		# rows
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
