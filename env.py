import jax.numpy as jnp
import jax
from system import *
import haiku as hk

verbose= False

KEY= jax.random.PRNGKey(4)
KEYS= hk.PRNGSequence(KEY)

class Env():
    def __init__(self, system : SimpleDiscrete, T= 96):
        
        self.system= system

        self.done = False
        self.system.t= 0

        self.T= T

        self.reward= 0
        self.error_integral= 0

        # TODO: add env state variable

    def reset(self):
        if verbose: print("Initial state: ", self.system.x0)

        self.system.x = self.system.x0 + jax.random.normal(next(KEYS))
        self.system.v = self.random_disturbance()
        
        if verbose: print("Initial state after noise: ", self.system.x)
        if verbose: print("Disturbance: \n ", self.system.v)
        
        self.done = False
        self.system.t= 0
        self.error_integral= 0

        obs= self.observation(self.system.x)

        return obs
 
    def step(self,u):

        x_p=  self.system.step(u)

        obs= self.observation(x_p)
        reward= self.rewards(x_p,u)

        if self.system.t >= self.T: 
            self.done= True
        if x_p < 0 or x_p > 30: 
            self.done= True
            reward+= -1000

        return obs, reward , self.done
    
    def rewards(self, x, u):
        return -(self.system.x_t - x)**2 - 0.01*u**2  
    
    def observation(self, x):
        self.error= self.system.x_t - x
        self.error_integral+= self.error
        return jnp.array([self.error, self.error_integral])
    

    def random_disturbance(self):
        v0 = jax.random.uniform(next(KEYS), minval= 15, maxval=20) 
        
        v=[v0]
        for t in range(self.T-1):
            v.append(v[t] + jax.random.normal(next(KEYS), shape=()) * 0.5)

        return v
