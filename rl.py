import numpy as np
import jax
import jax.numpy as jnp
import scipy 
from tqdm import tqdm

verbose= False

class DiscreteActionPolicy():
    def __init__(self, env, nn, action_set):
        self.env = env
        self.nn = nn
        self.gradient_function= jax.grad(self.log_function)
        self.action_set = action_set
        
    def sample_action(self, params, state):
        action_probs = np.asarray(self.nn(params,state)).astype('float64')

        # normalize action_probs to sum to 1 (numpy numerical stability)
        if np.abs(np.sum(action_probs)-1.) < 1e-9:
            pass
        else:
            if verbose: print("normalizing action_probs, diff: ", np.sum(action_probs)-1.)
            action_probs = action_probs/np.sum(action_probs)

        if verbose: print("action_probs: ", action_probs)

        return np.random.choice(self.action_set, p=action_probs)
    
    def log_function(self, params, state, action):
        action_probs = self.nn(params,state)
        return jnp.log(action_probs[action])
    
class ContinuousActionPolicy():
    def __init__(self, env, nn, umin, umax):
        self.env = env
        self.nn = nn
        self.gradient_function= jax.grad(self.log_function)

        self.umin = umin
        self.umax = umax
        
    def sample_action(self, params, state):
        action_params = self.nn(params,state)

        mean = action_params[0]
        std = jnp.exp(action_params[1])
        if verbose: print("mu, std: ", mean, std)
        action= scipy.stats.truncnorm.rvs(
                                a=(self.umin-mean)/std, 
                                b=(self.umax-mean)/std, 
                                loc=mean, 
                                scale=std)

        # action = scipy.stats.norm.rvs(loc=mean, scale=std)

        return action
    
    def log_function(self, params, state, action):
        action_params = self.nn(params,state)

        mean = action_params[0]
        std = jnp.exp(action_params[1])

        log_p_action= jax.scipy.stats.truncnorm.logpdf(
                                x= np.array([action]),
                                a=(self.umin-mean)/std, 
                                b=(self.umax-mean)/std, 
                                loc= mean, 
                                scale= std)
        
        # log_p_action= jax.scipy.stats.norm.logpdf(
        #                         x= np.array([action]),
        #                         loc=mean, 
        #                         scale= std)
        
        return jnp.asarray(log_p_action)[0]
    
class Value():
    def __init__(self, env, nn):
        self.env = env
        self.nn = nn
        self.gradient_function= jax.grad(self.sample_value)
        
    def sample_value(self, params, state):
        return self.nn(params,state)[0]


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

            grad_log_p= self.policy.gradient_function(self.params_p,
                                                      hot_s,
                                                      actions[t])
            
            if verbose: print("grad_log_p: ", grad_log_p, "step: ", self.alpha * self.gamma**t * G)

            self.params_p = update(self.params_p,
                                 grad_log_p,
                                 self.alpha * self.gamma**t * G)
            
            if verbose: print("params_p: ", self.params_p)

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

            grad_v= self.value.gradient_function(self.params_v,
                                                 hot_s)
            if verbose: print('grad_v: ', grad_v, "step: ", self.alpha_v * td_error)

            self.params_v = update(self.params_v,
                                   grad_v,
                                   self.alpha_v * td_error) 

            grad_log_p= self.policy.gradient_function(self.params_p,
                                                      hot_s, 
                                                      actions[t])
            if verbose: print("grad_log_p: ", grad_log_p, "step: ", self.alpha_p * self.gamma**t * G)

            self.params_p = update(self.params_p,
                                 grad_log_p,
                                 self.alpha_p * self.gamma**t * td_error)

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
            if verbose: print("TD: ", td_error)

            grad_v= self.value.gradient_function(self.params_v,
                                                 hot_s)
            if verbose: print('grad_v: ', grad_v, "step: ", self.alpha_v * td_error)

            self.params_v = update(self.params_v,
                                   grad_v,
                                   self.alpha_v * td_error) 

            grad_log_p= self.policy.gradient_function(self.params_p,
                                                      hot_s, 
                                                      actions[t])
            if verbose: print("grad_log_p: ", grad_log_p, "step: ", self.alpha_p * self.gamma**t * td_error)

            self.params_p = update(self.params_p,
                                 grad_log_p,
                                 self.alpha_p * self.gamma**t * td_error)
            
            

def update(params, grad, step):
    return jax.tree_util.tree_map(lambda p, g: p + step* g, params, grad)


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