import autograd.numpy as np
import scipy.sparse as sparse
from . import csc

class LTVSystem:
    """This deals with the (LTV) dynamics and constraints"""

    def __init__(self, model):
        self.m = model

    def init(self, nx, nu, N, x0, xmin, xmax, umin, umax, polyBlocks=None):
        """Return A, l, u"""
        self.nx = nx
        self.nu = nu
        self.N = N

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
        
        return self.A, self.l, self.u


if __name__ == "__main__":
    print("Testing LTVSystem")
    from models.pendulums import Pendulum
    # import pen

    model = Pendulum()
    ltvs = LTVSystem(model)




