"""Features added to other solvers for convenience"""
import autograd.numpy as np
from scipy.integrate import solve_ivp

from . import mpc

class DumbOdeResult:
    """Data is just arrays, but can call like scipy.integrate.OdeResult"""
    def __init__(self, ny, nu):
        self.t = np.zeros(0)
        self.y = np.zeros((ny, 0))
        self.t_events = None
    
    def append(self, sol, firstIndex=0):
        """Append data from an OdeSolution
        
        firstIndex --- set to 1 to leave out duplicate times
        """
        self.t = np.hstack((self.t, sol.t[firstIndex:]))
        self.y = np.hstack((self.y, sol.y[:, firstIndex:]))


def solve_ivp_dmpc(model, t_span, y0, dt, mpcdt, mpcgoal, N, wx, wu):
    """Similar to solve_ivp, but runs a discretized MPC"""

    # Instantiate MPC
    model.dt = mpcdt
    dpmpc = mpc.LTVMPC(model, N, wx, wu, verbose=False, polish=False, scaling=0, eps_rel=1e-2, eps_abs=1e-2, kdamping=0)

    # simulate
    # need to have discrete changes to the MPC
    def discretizationEvent(t, y):
        return t - tev[-1] - mpcdt
    discretizationEvent.direction = 1
    discretizationEvent.terminal = True

    # initial states
    tev = np.array([t_span[0]])
    uev = np.zeros((len(wu), 1))
    yev = np.zeros((len(wx), 1))
    yev[:, 0] = y0
    uev[:, 0] = dpmpc.update(y0, mpcgoal)
    ret = DumbOdeResult(ny=len(wx), nu=len(wx))

    while True:
        sol = solve_ivp(lambda t, y: model.dynamics(y, uev[:, -1]), [tev[-1], t_span[-1]], yev[:, -1], dense_output=True, events=discretizationEvent, t_eval=np.arange(tev[-1], t_span[-1] - 1e-3, dt))

        if sol.status == 1:
            # save data
            ret.append(sol)
            # events
            tev = np.hstack((tev, sol.t_events[0][0]))
            yev = np.hstack((yev, sol.sol(tev[-1])[:, np.newaxis]))
            uev = np.hstack((uev, dpmpc.update(yev[:, -1], mpcgoal)[:, np.newaxis]))
            # continuous for plotting
            # should we go again?
            if t_span[-1] - tev[-1] < dt:
                break

        if sol.status == 0:
            break

    # Return similar to solve_ivp
    return ret

