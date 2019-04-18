"""
Pendulum models for testing controls stuff
"""
import autograd.numpy as np

# common parameters
g = 9.81

class Pendulum:
    """Single pendulum"""
    l = 1.0
    kd = 0.1

    def dynamics(self, y, u):
        return np.array([y[1], -g * np.sin(y[0]) / self.l - self.kd * y[1] + u])

class DoublePendulum:
    """Class for model"""
    l1 = 1.0
    l2 = 1.0
    m1 = 1.0
    m2 = 1.0
    
    def dynamics(self, y, u):
        # unpack
        if len(u) > 1:
            # double pendulum
            tau1 = u[0]
            tau2 = u[1]
        else:
            # acrobot
            tau1 = 0
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
        y2dot = np.array([(g*l1*l2*m1*s1 + g*l1*l2*m2*s1 - c2*g*l1*l2*m2*s12 + 
        dq12*l1*l2*(c2*l1 + l2)*m2*s2 + 2*dq1*dq2*l1*l22*m2*s2 + 
        dq22*l1*l22*m2*s2 + l2*tau1 - c2*l1*tau2 - l2*tau2)/
        (l1**2*l2*(m1 + m2 - c2**2*m2)), 
        -((g*l1*l22*m1*m2*s1 + g*l1*l22*m22*s1 - c2**2*g*l1*l22*m22*s1 - 
            c1*g*l12*l2*m1*m2*s2 + 
            dq12*l1*l2*m2*(2*c2*l1*l2*m2 + l22*m2 + l12*(m1 + m2))*s2 - 
            c1*g*l12*l2*m22*s2 - c1*c2*g*l1*l22*m22*s2 + 
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
