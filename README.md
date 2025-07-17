# RLinear

Another environment for RL agents.    
This time, the simplest possible.   
The idea is to wrap generic dynamic LTI systems to compare PID and RL controllers.    

## Repo structure
- *custom_sb3* : a series of classes to use SB3 algorithm [https://stable-baselines3.readthedocs.io/en/master/#] with custom inner function approximations
    - `FA.py`: function approximations and action distribution classes
    - `policy.py`: extractor and policy classes

- *env* : folder for environment definition
    - `env_gym.py`: wrapper class to use Gymnasium functionalities [https://gymnasium.farama.org/api/env/]
    - `system.py`: Ax+Bu+Cv system defintion

- *my-rl* : homemade RL algorithm implementations for:
    - REINFORCE
    - REINFORCE with baseline
    - Actor Critic with TD-error

- `controllers.py`: PID and Relay classes for benchmarking
- `utils.py`: functions for testing and plotting results

