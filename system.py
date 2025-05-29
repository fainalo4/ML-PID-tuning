import numpy as np

class SimpleDiscrete:
    def __init__(self, A, Bu, Bv, x0, dt):
        """
        Initialize the environment with system matrices A, B and initial state x0.
        A: State transition matrix (numpy array)
        B: Control input matrix (numpy array)
        x0: Initial state (numpy array). If None, initializes to zeros.
        """
        self.A = A
        self.Bu = Bu
        self.Bv = Bv

        self.x0 = x0
        self.x = x0
        self.dt= dt

    def step(self, u, v):
        """
        Apply control input u and update the state.
        u: Control input (numpy array or scalar)
        Returns the new state.
        """
        self.x = self.A * self.x + self.Bu * u + self.Bv * v
        return self.x
