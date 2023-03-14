class SGD_Params:
    def __init__(self):
        pass


class Adam_Params:
    def __init__(
        self,
        eta: float = 0.001,
        eta_decay: float = 0.00001,
        epsilon: float = 1e-7,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
    ):
        self.eta = eta
        self.eta_decay = eta_decay
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2
