
class SimpleDiscrete:
    def __init__(self, A, Bu, Bv, x0, x_t, v):
        """
        Initialize the environment with system matrices A, B and initial state x0.
        A: State transition matrix (scalar)
        B: Control input matrix (scalar)
        x0: Initial state (scalar). 
        """
        self.A = A
        self.Bu = Bu
        self.Bv = Bv

        self.x0 = x0
        self.x = x0
        self.x_t= x_t
        self.v= v

        self.t= 0


    def step(self, u):
        """
        Apply control input u and update the state.
        u: Control input (scalar)
        Returns the new state.
        """
        self.x = self.A * self.x + self.Bu * u + self.Bv * self.v[self.t]
        self.t += 1
        return self.x


class MultiDiscrete:
    def __init__(self, A, Bu, Bv, x0, x_t, v):
        """
        Initialize the environment with system matrices A, B and initial state x0.
        A: State transition matrix (numpy array)
        B: Control input matrix (numpy array)
        x0: Initial state (numpy array)
        x_t: Target state (numpy array)
        v: Disturbance vector (numpy array)
        """
        self.A = A
        self.Bu = Bu
        self.Bv = Bv

        self.x0 = x0
        self.x = x0
        self.x_t= x_t
        self.v= v

        self.t= 0


    def step(self, u):
        """
        Apply control input u and update the state.
        u: Control input (numpy array)
        Returns the new state.
        """
        v_t= self.v[self.t].reshape(self.Bv.shape[1],1)
        self.x = self.A @ self.x + self.Bu @ u + self.Bv @ v_t
        self.t += 1
        return self.x