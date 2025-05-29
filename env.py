import numpy as np
import gymnasium as gym
import jax.numpy as jnp
import jax
from system import *

class Env():
    def __init__(self, system : SimpleDiscrete, states_size= 1):
        self.system= system

        self.done = False
        self.system.t= 0

        self.states_size= states_size
        self.actions_size= 2

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

        if self.system.t >=96: self.done= True

        return x_p, reward , self.done
    
    def rewards(self, x, u):
        return -(self.system.x_t - x)**2 - u**2  