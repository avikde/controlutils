"""
Base class for models
"""
from abc import abstractmethod
import autograd.numpy as np
from autograd import jacobian

class Model:
    # can be set
    dt = 0.1

    @abstractmethod
    def dydt(self, y, u, *args):
        """Continuous dynamics ydot = f(y, u)"""
        raise NotImplementedError()

    def _autoLinJac(self, y0, u0, *args):
        # check inputs
        y0 = np.asarray(y0)
        u0 = np.asarray(u0)
        dfdy = jacobian(lambda y: self.dydt(y, u0, *args))
        dfdu = jacobian(lambda u: self.dydt(y0, u, *args))
        return dfdy, dfdu
    
    def autoLin(self, y0, u0, *args):
        """Use autograd to get linearization at the provided point.
        Returns A, B, c, where c is the affine component.
        """
        dfdy, dfdu = self._autoLinJac(y0, u0, *args)
        A = dfdy(y0)
        B = dfdu(u0)
        # add affine term so that linear dynamics can be projected forward
        return A, B, self.dydt(y0, u0, *args) - A @ y0 - B @ u0
        
    
    def autoDLin(self, y0, u0, *args):
        """Linearization of discretized system with euler integration"""
        dfdy, dfdu, ydoterr = self.autoLin(y0, u0, *args)
        return np.eye(len(y0)) + self.dt * dfdy, self.dt * dfdu, self.dt * ydoterr


    def getLinearDynamics(self, y, u, *args):
        """Implementation that can be overriden"""
        return self.autoDLin(y, u, *args)
    
    def modelInfo(self, opt, traj):
        # number of knot points
        N = ((len(traj) - (1 if opt['vart'] else 0)) // (self.ny + self.nu)) - 1
        δt = traj[-1] if opt['vart'] else opt['fixedδt']
        yk = lambda k : traj[k*self.ny : (k+1)*self.ny]
        uk = lambda k : traj[(N+1)*self.ny + k*self.nu : (N+1)*self.ny + (k+1)*self.nu]
        return N, δt, yk, uk
    
