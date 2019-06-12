import autograd.numpy as np
import scipy.sparse as sparse
import csc

class LTVSystem:
    """This deals with the (LTV) dynamics and constraints"""

    def __init__(self, model):
        pass


if __name__ == "__main__":
    print("Testing LTVSystem")
    from models.pendulums import Pendulum
    # import pen

    model = Pendulum()
    ltvs = LTVSystem(model)




