import numpy as np
import gymnasium as gym
import jax.numpy as jnp
import jax
from system import *

class Env():
    def __init__(self, system : SimpleDiscrete, states_size= 1, actions_size= 1, T= 96):
        
        self.system= system

        self.done = False
        self.system.t= 0

        self.states_size= states_size
        self.actions_size= actions_size
        self.T= T

        self.reward= 0

    def reset(self):
        self.system.x = self.system.x0
        self.done = False
        self.system.t= 0
        return self.system.x
 
    def step(self,u):

        x= self.system.x
        x_p=  self.system.step(u)

        reward= self.rewards(x,u)

        if self.system.t >=self.T: self.done= True
        if self.system.x < 0 or self.system.x > 30: 
            self.done= True
            reward+= -1000

        return x_p, reward , self.done
    
    def rewards(self, x, u):
        return -(self.system.x_t - x)**2 - 0.01*u**2  