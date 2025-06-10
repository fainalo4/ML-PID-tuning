import numpy as np
import jax
import jax.numpy as jnp
import scipy

verbose= False


def update(params, grad, step):
    return jax.tree_util.tree_map(lambda p, g: p + step* g, params, grad)

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

    def update_policy(self,s,a,params,step):
            
        grad_log_p= self.gradient_function(params,s,a)

        # check if there are NaN values in the gradient
        if jnp.isnan(grad_log_p).any():
            raise ValueError("NaN values in policy gradient")
        
        if verbose: print("grad_log_p: ", grad_log_p, "step: ", step)

        new_params = update(params, grad_log_p, step)
        
        if verbose: print("params_p: ", params)

        return new_params
    
class ContinuousActionPolicy_Prob():
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
        return action
        
    def log_function(self, params, state, action):
        action_params = self.nn(params,state)

        mean = action_params[0]
        std = int(jnp.exp(action_params[1]))

        log_p_action= jax.scipy.stats.truncnorm.logpdf(
                                x= jnp.asarray(action),
                                a=(self.umin-mean)/std, 
                                b=(self.umax-mean)/std, 
                                loc= mean, 
                                scale= std)
        return jnp.asarray(log_p_action)
    
    def update_policy(self,s,a,params,step):
            
        grad_log_p= self.gradient_function(params,s,a)

        # check if there are NaN values in the gradient
        if jnp.isnan(grad_log_p).any():
            raise ValueError("NaN values in policy gradient")
        
        if verbose: print("grad_log_p: ", grad_log_p, "step: ", step)

        new_params = update(params, grad_log_p, step)
        
        if verbose: print("params_p: ", params)

        return new_params
    

class ContinuousActionPolicy_Deter():
    def __init__(self, env, nn, umin, umax):
        self.env = env
        self.nn = nn
        self.gradient_function= jax.grad(self.sample_action)

        self.umin = umin
        self.umax = umax
        
    def sample_action(self, params, state):
        return self.nn(params,state)[0]
    
    def update_policy(self,s,a,params,step):
            
        grad_log_p= self.gradient_function(params,s)

        # check if there are NaN values in the gradient
        if jnp.isnan(grad_log_p).any():
            raise ValueError("NaN values in policy gradient")
        
        if verbose: print("grad_log_p: ", grad_log_p, "step: ", step)

        new_params = update(params, grad_log_p, step)
        
        if verbose: print("params_p: ", params)

        return new_params
    
    
class Value():
    def __init__(self, env, nn):
        self.env = env
        self.nn = nn
        self.gradient_function= jax.grad(self.sample_value)
        
    def sample_value(self, params, state):
        v= self.nn(params,state)
        return jnp.reshape(v,())
    
    def update_value(self, s, params, step):
        grad_v= self.gradient_function(params, s)

        # check if there are NaN values in the gradient
        if jnp.isnan(grad_v).any():
            raise ValueError("NaN values in value gradient")
        
        if verbose: print('grad_v: ', grad_v, "step: ", step)

        new_params = update(params, grad_v, step)
        
        if verbose: print("params_v: ", params)

        return new_params