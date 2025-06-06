import jax.numpy as jnp
from tqdm import tqdm

verbose= False

class Reinforce():
    def __init__(self, env, policy, params, gamma, alpha):
        
        self.env = env
        self.policy = policy
        self.params_p= params
        self.gamma = gamma
        self.alpha = alpha
        self.history_reward= []


    def train(self, num_episodes):
        #TODO: make alpha decaying

        for episode in tqdm(range(num_episodes), desc="episodes"):

            states, actions, rewards= trajectory(self)
            
            self.history_reward.append( sum(rewards) )

            self.update_params(states, actions, rewards)

    def update_params(self, states, actions, rewards):
        for t in range(len(rewards)):

            if verbose: print("t: ", t)

            G = sum([ self.gamma**(k-t) * rewards[k] for k in range(t, len(rewards))])
            hot_s= jnp.array(states[t], dtype=jnp.float32)
            step= self.alpha * self.gamma**t * G

            self.params_p = self.policy.update_policy(hot_s, 
                                                      actions[t],
                                                      self.params_p,
                                                      step)

class Reinforce_w_baseline():
    def __init__(self, env, policy, params_p, value, params_v, gamma, alpha_p, alpha_v):
        
        self.env = env
        self.policy = policy
        self.params_p= params_p
        self.value= value
        self.params_v= params_v
        self.gamma = gamma
        self.alpha_p = alpha_p
        self.alpha_v = alpha_v
        self.history_reward= []

    def train(self, num_episodes):
        for episode in tqdm(range(num_episodes), desc="episodes"):

            states, actions, rewards= trajectory(self)

            self.history_reward.append( sum(rewards) )

            self.update_params(states, actions, rewards)

    def update_params(self, states, actions, rewards):
        for t in range(len(rewards)):
            if verbose: print("t: ", t)

            G = sum([ self.gamma**(k-t) * rewards[k] for k in range(t, len(rewards))])

            hot_s= jnp.array(states[t], dtype=jnp.float32)

            td_error= G - self.value.sample_value(self.params_v, hot_s)
            step_v= self.alpha_v * td_error
            self.params_v = self.value.update_value(hot_s,
                                                    self.params_v,
                                                    step_v)

            step_p= self.alpha_p * self.gamma**t * td_error
            self.params_p = self.policy.update_policy(hot_s, 
                                                      actions[t],
                                                      self.params_p,
                                                      step_p)

class AC():
    def __init__(self, env, policy, params_p, value, params_v, gamma, alpha_p, alpha_v):
        
        self.env = env
        self.policy = policy
        self.params_p= params_p
        self.value= value
        self.params_v= params_v
        self.gamma = gamma
        self.alpha_p = alpha_p
        self.alpha_v = alpha_v
        self.history_reward= []

    def train(self, num_episodes):
        for episode in tqdm(range(num_episodes), desc="episodes"):

            states, actions, rewards= trajectory(self)

            self.history_reward.append( sum(rewards) )

            self.update_params(states, actions, rewards)

    def update_params(self, states, actions, rewards):
        for t in range(len(rewards[:-1])):
            if verbose: print("t: ", t)

            hot_s= jnp.array(states[t], dtype=jnp.float32)
            hot_s1= jnp.array(states[t+1], dtype=jnp.float32)

            td_error= rewards[t] \
                        + self.gamma* self.value.sample_value(self.params_v, hot_s1) \
                        - self.value.sample_value(self.params_v, hot_s)
            
            step_v= self.alpha_v * td_error
            self.params_v = self.value.update_value(hot_s,
                                                    self.params_v,
                                                    step_v)
            
            step_p= self.alpha_p * self.gamma**t * td_error
            self.params_p = self.policy.update_policy(hot_s, 
                                                      actions[t],
                                                      self.params_p,
                                                      step_p)
            
            


def trajectory(self):
    """
    Sample trajectories from the environment using the given policy.
    Returns a list of (state, action, reward) tuples for each episode.
    """
    self.env.state = self.env.reset()
    rewards = []
    states = []
    actions = []

    done = False
    t=0 
    while not done:
        t+= 1
        if verbose: print("t: ", t)

        action = self.policy.sample_action(self.params_p, self.env.state)
        next_state, reward, done = self.env.step(action)
        if verbose: print("state: ", self.env.state, "action: ", action, "reward: ", reward)

        states.append(self.env.state)
        actions.append(action)
        rewards.append(reward)

        self.env.state= next_state
    
    return states, actions, rewards