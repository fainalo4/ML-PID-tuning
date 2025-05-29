import numpy as np
import gymnasium as gym
import jax.numpy as jnp
import jax
from system import *

class Env():
    def __init__(self, system : SimpleDiscrete):
        self.system= system
        self.state = 0
        self.done = False
        self.time_step= 0

        self.states_size= states_size
        self.actions_size= 2

        self.rewards=  np.zeros([self.states_size,self.actions_size])
        self.rewards[0,1]= 0.5
        self.rewards[self.states_size-1,0]= 1

        self.state_trans= np.zeros([self.states_size, self.states_size, self.actions_size])
        # a1
        for row in range(self.states_size):
            for col in range(self.states_size):
                if row==col:
                    self.state_trans[row,col,0]= 0.05
                elif row==col-1:
                    self.state_trans[row,col,0]= 0.9
                elif row==col+1:
                    self.state_trans[row,col,0]= 0.05
        self.state_trans[0,0,0]= 0.1
        self.state_trans[-1,-1,0]= 0.95
        # a2
        for row in range(self.states_size):
            for col in range(self.states_size):
                self.state_trans[0,0,1]= 1
                if row==col+1:
                    self.state_trans[row,col,1]= 1

    def reset(self):
        self.state = 0
        self.done = False
        self.time_step= 0
        return self.state
 
    def step(self,action):
        s= self.state
        a= action
        new_state=  self.system.step(action, self.system)
        

        reward= self.rewards[s,a]

        self.time_step+= 1
        if self.time_step>=100: self.done= True

        return new_state, reward , self.done
    
# # Test the environment
# env= RiverSwim()
# s= env.reset()
# print("Initial state:", s)
# a= np.random.randint(env.actions_size)
# print("Action taken:", a)
# s, r, d=  env.step(a)
# print("New state:", s)
# print("Reward received:", r)
# print("Done:", d)


class FrozenLake():
    def __init__(self):
        self.name= 'FrozenLake'
        self.env= gym.make('FrozenLake-v1', is_slippery=False)
        self.state = self.env.reset()[0]
        self.done = False
        self.actions_size = self.env.action_space.n
        self.states_size = self.env.observation_space.n

    def reset(self):
        self.state = self.env.reset()[0]
        self.done = False
        return self.state
 
    def step(self,action):
        s= self.state
        a= action
        s1, r, self.done, _, _ = self.env.step(a)
        # Modify the reward with a small incentive to keep moving
        if jnp.array_equal(s1, s):
            r = -0.01

        return s1, r , self.done
    
# # Test the environment
# env= FrozenLake()
# s= env.reset()
# print("Initial state:", s)
# a= np.random.randint(env.actions_size)
# print("Action taken:", a)
# s, r, d=  env.step(a)
# print("New state:", s)
# print("Reward received:", r)
# print("Done:", d)

class CartPole():
    def __init__(self):
        self.name= 'CartPole'
        self.env= gym.make('CartPole-v1')
        self.state = self.env.reset()[0]
        self.done = False
        self.actions_size = self.env.action_space.n
        self.states_size = self.env.observation_space.shape[0]

        self.gamma= 0.99

    def reset(self):
        self.state = self.env.reset()[0]
        self.done = False
        return self.state
 
    def step(self,action):
        a= action

        s1, r, term, trunc, _ = self.env.step(a)
        self.done= term or trunc
        if self.done: r= 0
        if trunc: r= 1/(1-self.gamma)

        return s1, r , self.done
    
# # Test the environment
# env= CartPole()
# s= env.reset()
# print("Initial state:", s)
# a= np.random.randint(env.actions_size)
# print("Action taken:", a)
# s, r, d=  env.step(a)
# print("New state:", s)
# print("Reward received:", r)
# print("Done:", d)

# Function that returns a vector containing all parameters
def tree_ravel(pytree):
    return jnp.concatenate([jnp.ravel(leaf) for leaf in jax.tree_util.tree_leaves(pytree)])