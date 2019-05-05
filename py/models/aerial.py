"""
Flying models
"""
import autograd.numpy as np
from ..model import Model

g = 9.81

class Quadrotor2D(Model):
    """config q is SE(2). From [Underactuated book, 3.3.1]"""
    m = 0.5
    r = 0.1
    ib = 0.001

    def dynamics(self, y, u):
        nq = 3
        u1 = u[0]
        u2 = u[1]
        ydot = np.hstack((y[nq:], 
        np.array([
            -(u1 + u2) / self.m * np.sin(y[2]),
            (u1 + u2) / self.m * np.cos(y[2]) - g,
            self.r * (u1 - u2) / self.ib
            ])
        ))
        return ydot

    def kinematics(self, y):
        """Returns position of the end of each link"""
        p1 = self.l * np.array([-np.sin(y[0]), -np.cos(y[0])])
        return p1
