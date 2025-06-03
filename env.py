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
        # self.error_integral= 0

    def reset(self):
        self.system.x = self.system.x0
        self.done = False
        self.system.t= 0
        # self.error_integral= 0

        return self.system.x
 
    def step(self,u):

        x_p=  self.system.step(u)

        obs= self.observation(x_p)
        reward= self.rewards(x_p,u)

        if self.system.t >=self.T: self.done= True
        if x_p < 0 or x_p > 30: 
            self.done= True
            reward+= -1000

        return obs, reward , self.done
    
    def rewards(self, x, u):
        return -(self.system.x_t - x)**2 - 0.01*u**2  
    
    def observation(self,x):
        return x

    # def observation_1(self):
    #     self.error= self.system.x_t - self.system.x
    #     self.error_integral+= self.error
    #     return jnp.array([self.error, self.error_integral])