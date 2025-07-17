import numpy as np

class Discrete:
    def __init__(self, A, Bu, Bv, x0, x_t, v,
                 obs_index, act_index):
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

        self.observability= obs_index
        self.actuability= act_index


    def step(self, u):
        """
        Apply control input u and update the state.
        u: Control input (numpy array)
        Returns the new state.
        """

        c_u= self.actuate(u)
        v_t= self.v[:,self.t].reshape(self.Bv.shape[1],1)

        self.x = self.A @ self.x + self.Bu @ c_u + self.Bv @ v_t
        self.t += 1

        return self.observe(self.x)
    
    def observe(self,x):
        return x[self.observability]
    
    def actuate(self,u):
        complete_u= np.zeros([self.Bu.shape[1],1])
        complete_u[self.actuability]= u
        return complete_u


class Continuous(Discrete):
    def __init__(self, A, Bu, Bv, x0, x_t, v,
                 obs_index, act_index,
                 dt):
        """
        Initialize the continuous environment with system matrices A, B and initial state x0.
        A: State transition matrix (numpy array)
        Bu: Control input matrix (numpy array)
        Bv: Disturbance matrix (numpy array)
        x0: Initial state (numpy array)
        x_t: Target state (numpy array)
        v: Disturbance vector (numpy array)
        """
        super().__init__(A, Bu, Bv, x0, x_t, v, obs_index, act_index)

        self.dt= dt

    def step(self, u):
        """
        Apply control input u and update the state in continuous time.
        u: Control input (numpy array)
        Returns the new state.
        """
        c_u= self.actuate(u)
        v_t= self.v[:,self.t].reshape(self.Bv.shape[1], 1)

        # Continuous time state update
        self.x = self.x + (self.A @ self.x + self.Bu @ c_u + self.Bv @ v_t) * self.dt
        self.t += 1

        return self.observe(self.x)