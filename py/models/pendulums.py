"""
Pendulum models for testing controls stuff
"""
import autograd.numpy as np
from ..model import Model

# common parameters
g = 9.81

class Pendulum(Model):
    """Single pendulum"""
    l = 1.0
    kd = 0.1

    def dynamics(self, y, u):
        return np.array([y[1], -g * np.sin(y[0]) / self.l - self.kd * y[1] + u[0]])

    def kinematics(self, y):
        """Returns position of the end of each link"""
        p1 = self.l * np.array([-np.sin(y[0]), -np.cos(y[0])])
        return p1


class DoublePendulum(Model):
    """Class for model"""
    l1 = 1.0
    l2 = 1.0
    m1 = 1.0
    m2 = 1.0

    def __init__(self, nu=2):
        """If nu=2 assumes double pendulum, and if nu=1, acrobot"""
        self.nu = nu

    def _getLimits(self):
        if self.nu == 2:
            umin = np.array([-100, -100])
        else:
            umin = np.array([-100])
        umax = -umin
        xmin = np.full(4, -np.inf)
        xmax = -xmin
        return umin, umax, xmin, xmax
    limits = property(_getLimits)
    
    def dynamics(self, y, u):
        # unpack
        if self.nu > 1:
            # double pendulum
            tau1 = u[0]
            tau2 = u[1]
        else:
            # acrobot
            tau1 = 0.0
            tau2 = u[0] 
        l1 = self.l1
        l2 = self.l2
        m1 = self.m1
        m2 = self.m2
        dq1 = y[2]
        dq2 = y[3]
        s1 = np.sin(y[0])
        c1 = np.cos(y[0])
        s2 = np.sin(y[1])
        c2 = np.cos(y[1])
        s12 = np.sin(y[0] + y[1])
        dq12 = dq1**2
        dq22 = dq2**2
        l22 = l2**2
        l12 = l1**2
        m22 = m2**2

        # from mathematica
        y2dot = np.array([(-(g*l1*l2*m1*s1) - g*l1*l2*m2*s1 + c2*g*l1*l2*m2*s12 + 
      dq12*l1*l2*(c2*l1 + l2)*m2*s2 + 2*dq1*dq2*l1*l22*m2*s2 + 
      dq22*l1*l22*m2*s2 + l2*tau1 - c2*l1*tau2 - l2*tau2)/
    (l1**2*l2*(m1 + m2 - c2**2*m2)), 
        -((-(g*l1*l22*m1*m2*s1) - g*l1*l22*m22*s1 + 
        c2**2*g*l1*l22*m22*s1 + c1*g*l12*l2*m1*m2*s2 + 
        dq12*l1*l2*m2*(2*c2*l1*l2*m2 + l22*m2 + l12*(m1 + m2))*s2 + 
        c1*g*l12*l2*m22*s2 + c1*c2*g*l1*l22*m22*s2 + 
        2*dq1*dq2*l1*(c2*l1 + l2)*l22*m22*s2 + 
        dq22*l1*(c2*l1 + l2)*l22*m22*s2 + c2*l1*l2*m2*tau1 + l22*m2*tau1 - 
        l12*m1*tau2 - l12*m2*tau2 - 2*c2*l1*l2*m2*tau2 - l22*m2*tau2)/
      (l1**2*l2**2*m2*(m1 + m2 - c2**2*m2)))
        ])

        return np.hstack((y[2:], y2dot))

    def kinematics(self, y):
        """Returns position of the end of each link"""
        p1 = self.l1 * np.array([-np.sin(y[0]), -np.cos(y[0])])
        p2 = p1 + self.l2 * np.array([-np.sin(y[0] + y[1]), -np.cos(y[0] + y[1])])
        return p1, p2

class PendulumFlyWheel(Model):
    """Equations from Pratt (2006)"""
    m = 1.0
    g = 9.81
    J = 1e-3

    def dynamics(self, y, u):
        qdot = y[3:]
        fk = u[0]
        tauh = u[1]
        l = np.sqrt(y[0]**2 + y[1]**2)
        tha = np.arctan2(y[0], y[1])
        return np.hstack((qdot,
            np.array([1/self.m * (fk * np.sin(tha) - tauh/l * np.cos(tha)),
            -self.g + 1/self.m * (fk * np.cos(tha) + tauh/l * np.sin(tha)),
            tauh/self.J ])
        ))

    def kinematics(self, y):
        """Returns position of the end of each link"""
        p1 = self.l * y[0:2]
        return p1

class LIP(Model):
    """Equations from Pratt (2006)"""
    m = 1.0
    g = 9.81
    J = 1e-3
    z0 = 0.1

    def dynamics(self, y, u):
        qdot = y[2:]
        tauh = u[0]
        return np.hstack((qdot,
            np.array([self.g/self.z0 * y[0] - 1./(self.m * self.z0) * tauh,
            tauh/self.J ])
        ))

    def kinematics(self, y):
        """Returns position of the end of each link"""
        p1 = self.l * np.array([y[0], self.z0])
        return p1
