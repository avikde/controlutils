import autograd.numpy as np
import scipy.sparse as sparse

class LTVSystem:
    def __init__(self, model):
        pass


if __name__ == "__main__":
    print("Testing LTVSystem")
    from models.pendulums import Pendulum
    # import pen

    model = Pendulum()
    ltvs = LTVSystem(model)




