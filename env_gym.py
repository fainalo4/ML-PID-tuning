import numpy as np
import gymnasium as gym
from gymnasium import spaces

from system import *

verbose= False


class Env(gym.Env):
    def __init__(self, system, o_dim, a_dim, umin, umax, T= 96):

        self.render_mode= None

        self.observation_space= spaces.Box(-np.inf, np.inf, shape=(2*o_dim,1))
        self.action_space= spaces.Box(np.float32(umin), np.float32(umax), shape=(a_dim,1))

        self.system= system
        self.system.t= 0
        self.T= T
        self.error_integral= 0

        self.terminated = False
        self.truncated= False
        self.reward= 0

        # TODO: add env state variable

    def reset(self, seed= None, v= None):

        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # SYSTEM
        self.system.x = self.system.x0 + np.random.uniform(low=-5, high=5,
                                                           size= self.system.x0.shape)
        self.system.v = self.random_disturbance() if v is None else v
        self.system.t= 0
        self.error_integral= 0

        # ENV
        self.terminated = False
        self.truncated= False
        obs= self.observation(self.system.observe(self.system.x))

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
        '''
        Compute average reward over observations
        '''
        r_vec= (self.system.x_t.T - x)**2 + 0.0001*u**2
        dim= x.shape[0]
        return float(-np.sum(r_vec)/dim)
        
    def observation(self, x):
        self.error= self.system.x_t - x
        self.error_integral+= self.error
        obs= np.concatenate([self.error, self.error_integral], axis=1)
        return obs.flatten().reshape(obs.size,1)
    

    def random_disturbance(self):
        v_dim= self.system.Bv.shape[1]
        v0 = np.random.uniform(low= 15, high=20, size=(v_dim,1)) 
        v=v0
        for t in range(self.T-1):
            v= np.concatenate([v, v[:,t].reshape(v_dim,1) + np.random.normal(size=(v_dim,1)) * 0.5], axis=1)
        return v
