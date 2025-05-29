import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm

verbose= True

class DiscreteActionPolicy():
    def __init__(self, env, nn):
        self.env = env
        self.nn = nn
        self.gradient_function= jax.grad(self.log_function)
        
    def sample_action(self, params, state):
        action_probs = self.nn(params,state)
        # normalize action_probs to sum to 1 (numpy numerical stability)
        action_probs= np.array(action_probs)
        action_probs= action_probs/np.sum(action_probs)
        if verbose: print("action_probs: ", action_probs)

        return np.random.choice(self.env.actions_size, p=action_probs)
    
    def log_function(self, params, state, action):
        action_probs = self.nn(params,state)
        return jnp.log(action_probs[action])
    
class ContinuousActionPolicy():
    def __init__(self, env, nn):
        self.env = env
        self.nn = nn
        self.gradient_function= jax.grad(self.sample_action)
        
    def sample_action(self, params, state):
        return self.nn(params,state)[0]
    
    def log_function(self, params, state, action):
        action_probs = self.nn(params,state)
        return jnp.log(action_probs[action])
    
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
        for episode in tqdm(range(num_episodes), desc="episodes"):
            self.env.state = self.env.reset()
            rewards = []
            states = []
            actions = []

            done = False
            while not done:
                
                # TODO: remove one-hot encoding variable name
                hot_s= jnp.array([self.env.state], dtype=jnp.float32)
                
                action = self.policy.sample_action(self.params_p, hot_s)
                next_state, reward, done = self.env.step(action)
                if verbose: print("state: ", self.env.state, "action: ", action, "reward: ", reward)

                states.append(self.env.state)
                actions.append(action)
                rewards.append(reward)

                self.env.state= next_state
            
            self.history_reward.append( sum(rewards) )

            self.update_policy(states, actions, rewards)

    def update_policy(self, states, actions, rewards):
        for t in range(len(rewards)):
            G = sum([ self.gamma**(k-t) * rewards[k] for k in range(t, len(rewards))])
            
            hot_s= hot_s= jnp.array([states[t]], dtype=jnp.float32)

            # grad_log_p= self.policy.gradient_function(self.params_p,
            #                                           hot_s, 
            #                                           actions[t])

            grad_log_p= self.policy.gradient_function(self.params_p,
                                                      hot_s)

            self.params_p = update(self.params_p,
                                 grad_log_p,
                                 self.alpha * self.gamma**t * G)


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
            self.env.state = self.env.reset()
            rewards = []
            states = []
            actions = []

            done = False
            while not done:

                hot_s= jax.nn.one_hot(self.env.state,num_classes= self.env.states_size)
                if self.env.name=='CartPole': hot_s= self.env.state

                action = self.policy.sample_action(self.params_p, hot_s)
                next_state, reward, done = self.env.step(action)
                if verbose: print("state: ", self.env.state, "action: ", action, "reward: ", reward)

                states.append(self.env.state)
                actions.append(action)
                rewards.append(reward)

                self.env.state= next_state

            self.history_reward.append( sum(rewards) )

            self.update_stuff(states, actions, rewards)

    def update_stuff(self, states, actions, rewards):
        for t in range(len(rewards)):
            G = sum([ self.gamma**(k-t) * rewards[k] for k in range(t, len(rewards))])

            hot_s= jax.nn.one_hot(states[t],num_classes= self.env.states_size)
            if self.env.name=='CartPole': hot_s= states[t]

            td_error= G - self.value.sample_value(self.params_v, hot_s)

            grad_v= self.value.gradient_function(self.params_v,
                                                 hot_s)
            self.params_v = update(self.params_v,
                                   grad_v,
                                   self.alpha_v * td_error) 

            grad_log_p= self.policy.gradient_function(self.params_p,
                                                      hot_s, 
                                                      actions[t])
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
            self.env.state = self.env.reset()
            rewards = []
            states = []
            actions = []

            done = False
            while not done:

                hot_s= jax.nn.one_hot(self.env.state,num_classes= self.env.states_size)
                if self.env.name=='CartPole': hot_s= self.env.state

                action = self.policy.sample_action(self.params_p, hot_s)
                next_state, reward, done = self.env.step(action)
                if verbose: print("state: ", self.env.state, "action: ", action, "reward: ", reward)

                states.append(self.env.state)
                states.append(next_state)
                actions.append(action)
                rewards.append(reward)

                self.env.state= next_state

            self.history_reward.append( sum(rewards) )

            self.update_stuff(states, actions, rewards)

    def update_stuff(self, states, actions, rewards):
        for t in range(len(rewards)):
            hot_s= jax.nn.one_hot(states[t],num_classes= self.env.states_size)
            hot_s1= jax.nn.one_hot(states[t+1],num_classes= self.env.states_size)
            if self.env.name=='CartPole': 
                hot_s= states[t]
                hot_s1= states[t+1]

            if verbose: print('s ', states[t], 's1 ', states[t+1], 'a ', actions[t], 'r ', rewards[t])
                
            td_error= rewards[t] \
                        + self.gamma* self.value.sample_value(self.params_v, hot_s1) \
                        - self.value.sample_value(self.params_v, hot_s)
            if verbose: print("TD: ", td_error)

            grad_v= self.value.gradient_function(self.params_v,
                                                 hot_s)
            self.params_v = update(self.params_v,
                                   grad_v,
                                   self.alpha_v * td_error) 

            grad_log_p= self.policy.gradient_function(self.params_p,
                                                      hot_s, 
                                                      actions[t])
            self.params_p = update(self.params_p,
                                 grad_log_p,
                                 self.alpha_p * self.gamma**t * td_error)
            
            if verbose: print("grad_log_p: ", grad_log_p)
            if verbose: print("grad_v: ", grad_v)
            

def update(params, grad, step):
    return jax.tree_util.tree_map(lambda p, g: p + step* g, params, grad)