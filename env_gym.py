import jax.numpy as jnp
import jax
import haiku as hk

import gymnasium as gym
from gymnasium import spaces

from system import *

verbose= False

KEY= jax.random.PRNGKey(4)
KEYS= hk.PRNGSequence(KEY)

class Env(gym.Env):
    def __init__(self, system, o_dim, a_dim, umin, umax, T= 96):

        self.render_mode= None

        self.observation_space= spaces.Box(-jnp.inf, jnp.inf, shape=(2*o_dim,1))
        
        self.action_space= spaces.Box(umin,umax, shape=(a_dim,1))
        # in case it can be changed after init

        self.system= system

        self.terminated = False
        self.truncated= False

        self.system.t= 0
        self.T= T

        self.reward= 0
        self.error_integral= 0

        # TODO: add env state variable

    def reset(self, seed= None):

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.system.x = self.system.x0 + jax.random.uniform(next(KEYS), 
                                                            shape= self.system.x0.shape,
                                                            minval=-5, maxval=5)
        self.system.v = self.random_disturbance()
        
        self.terminated = False
        self.truncated= False

        self.system.t= 0
        self.error_integral= 0

        obs= self.observation(self.system.x)

        return obs, {}
 
    def step(self,u):

        x_p=  self.system.step(u)

        obs= self.observation(x_p)
        reward= self.rewards(x_p,u)

        if self.system.t >= self.T: 
            self.terminated= True
        if (x_p < 0).any() or (x_p > 30).any(): 
            self.truncated= True
            reward+= -1000

        if verbose: print('x', x_p, 'o:', obs,'a', u, 'r:', reward )

        return obs, reward, self.terminated, self.truncated, {}
    
    def rewards(self, x, u):
        r_vec= -(self.system.x_t - x)**2 - 0.0001*u**2
        return float(jnp.sum(r_vec))
    
    def observation(self, x):
        self.error= self.system.x_t - x
        self.error_integral+= self.error
        return jnp.concatenate([self.error, self.error_integral], axis=0)
    

    def random_disturbance(self):
        v_dim= self.system.Bv.shape[1]
        v0 = jax.random.uniform(next(KEYS), shape=(v_dim,1), minval= 15, maxval=20) 
        v=[v0]
        for t in range(self.T-1):
            v.append(v[t] + jax.random.normal(next(KEYS), shape=(v_dim,1)) * 0.5)
        return v


    def reset_for_test(self,v):

        self.system.x = self.system.x0
        self.system.v = v
        
        self.terminated = False
        self.truncated= False

        self.system.t= 0
        self.error_integral= 0

        obs= self.observation(self.system.x)

        return obs, {}