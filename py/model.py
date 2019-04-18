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
    def dynamics(self, y, u):
        """Continuous dynamics ydot = f(y, u)"""
        raise NotImplementedError()
    
    def autoDLin(self, y0, u0):
        """Use autograd to get discretized linearization at the provided point.

        Returns A, B, c, where c is the affine component.
        """
        # check inputs
        y0 = np.asarray(y0)
        u0 = np.asarray(u0)

        dfdy = jacobian(lambda y: self.dynamics(y, u0))(y0)
        dfdu = jacobian(lambda u: self.dynamics(y0, u))(u0)

        # add affine term so that linear dynamics can be projected forward
        c = self.dt * (self.dynamics(y0, u0) - dfdy @ y0 - dfdu @ u0)
        A = np.eye(len(y0)) + self.dt * dfdy
        B = self.dt * dfdu

        return A, B, c

    def getLinearDynamics(self, y, u):
        """Implementation that can be overriden"""
        return self.autoDLin(y, u)
