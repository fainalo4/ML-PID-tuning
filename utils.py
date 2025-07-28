import numpy as np
import matplotlib.pyplot as plt

EPISODE_REWARDS= []

def reward_callback(_locals, _globals):
    info= _locals['infos']
    # This callback is called at each step, but we want to save rewards at the end of each episode
    if len(info) > 0 and 'episode' in info[0]:
        r= info[0]['episode']['r']
        EPISODE_REWARDS.append(r)

    # This part performs early stopping
    output= True
    sb3_w= 10
    if len(EPISODE_REWARDS) > sb3_w:
        avg_reward= np.mean(EPISODE_REWARDS[-sb3_w::])
        if avg_reward > -1e5: output= False

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


def rl_trajectory(env, v, x0, controller):
    """
    Sample trajectories from the environment using the given RL controller.
    Returns a list of (state, action, reward) tuples for each episode.
    """
    verbose= False

    obs_0, _ = env.reset(v=v, x0=x0)
    state= obs_0 

    rewards = []
    states = []
    actions = []
    sys_states= env.system.x
    sys_obs= env.system.observe(env.system.x)

    done = False
    t=0 
    while not done:
        t+= 1
        if verbose: print("t: ", t)

        action = controller.predict(state)[0]

        new_state, reward, term, trunc, _ = env.step(action)
        if verbose: print("state: ", state, "new_ state: ", new_state, "action: ", action, "reward: ", reward)

        states.append(new_state)
        actions.append(action[0][0])
        rewards.append(reward)
        sys_states= np.concatenate([sys_states, env.system.x], axis=1)
        sys_obs= np.concatenate([sys_obs, env.system.observe(env.system.x)])

        state= new_state
        done= term or trunc
    
    return states, actions, rewards, sys_states, sys_obs


def plot_test(time, states, x_t, actions, v_test, umin, umax):
    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel('time (hr)')
    ax1.set_ylabel('T [°C]', color=color)
    ax1.plot(time, states, color=color, label='x')
    ax1.plot(time, v_test, color=color, alpha=0.5, label='v')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim((0, 30))
    ax1.grid()

    plt.axhline(y=x_t[0], color=color, linestyle='--', label='x target')

    ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis

    color = 'tab:orange'
    ax2.set_ylabel('Q [kW]', color=color)  # we already handled the x-label with ax1
    ax2.plot(time, actions, color=color, label='u')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim((umin[0][0], umax[0][0]))

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.legend(bbox_to_anchor=(0.3, 0.4))
    plt.show()