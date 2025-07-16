import numpy as np

EPISODE_REWARDS= []

def reward_callback(_locals, _globals):
    # This callback is called at each step, but we want to save rewards at the end of each episode
    if len(_locals['infos']) > 0 and 'episode' in _locals['infos'][0]:
        EPISODE_REWARDS.append(_locals['infos'][0]['episode']['r'])

    # This part performs early stopping
    output= True
    sb3_w= 100
    if len(EPISODE_REWARDS) > sb3_w:
        avg_reward= np.mean(EPISODE_REWARDS[-sb3_w::])
        if avg_reward > -50: output= False

    return output


def pid_trajectory(env, v, x0, controller):
    """
    Sample trajectories from the environment using the given controller.
    Returns a list of (state, action, reward) tuples for each episode.
    """
    verbose= False

    _,_ = env.reset(v=v, x0=x0)
    x= env.system.observe(env.system.x)
    controller.reset()

    rewards = []
    states = []
    actions = []
    sys_states= env.system.x

    done = False
    t=0 
    while not done:
        t+= 1
        if verbose: print("t: ", t)

        action = controller.compute(x)
        _, reward, term, trunc, _ = env.step(action)
        if verbose: print("x: ", x, "action: ", action, "reward: ", reward)

        states.append(x)
        actions.append(action)
        rewards.append(reward)
        sys_states= np.concatenate([sys_states, env.system.x], axis=1)

        x= env.system.observe(env.system.x)
        done= term or trunc
    
    return states, actions, rewards, sys_states


def trajectory(env, v, controller):
    """
    Sample trajectories from the environment using the given controller.
    Returns a list of (state, action, reward) tuples for each episode.
    """
    verbose= True

    state,_ = env.reset_for_test(v)

    rewards = []
    states = []
    actions = []

    done = False
    t=0 
    while not done:
        t+= 1
        if verbose: print("t: ", t)

        action = controller.compute(state)
        next_state, reward, term, trunc, _ = env.step(action)
        if verbose: print("state: ", state, "action: ", action, "reward: ", reward)

        states.append(state)
        actions.append(action)
        rewards.append(reward)

        state= next_state
        done= term or trunc
        if done: states.append(next_state)
    
    return states, actions, rewards