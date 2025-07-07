"""
Advanced Example for FA customization (SB3)
- https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
- https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/distributions.html#CategoricalDistribution.proba_distribution_net

Custom NN architecture (Pytorch): 
- https://stackoverflow.com/questions/70269663/how-to-efficiently-implement-a-non-fully-connected-linear-layer-in-pytorch 
- https://discuss.pytorch.org/t/parallel-execution-of-modules-in-nn-modulelist/43940/7 

"""

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
from torch import nn
import numpy as np

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.preprocessing import get_action_dim


class MultiNN(nn.Module):
    def __init__(self, controllers_number) -> None:
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(2, 1, bias= False) for i in range(controllers_number)])

    def forward(self, x : th.Tensor) -> th.Tensor:
        y= th.Tensor()
        for i,l in enumerate(self.linears):
            y= th.cat([y, l(x[0][2*i:2*(i+1)]) ])
        return y


class CustomDistribution(DiagGaussianDistribution):
    def __init__(self, action_dim: int):
        super().__init__(action_dim)
    
    def proba_distribution_net(self, latent_dim: int, log_std_init: float = 0.0) -> tuple[nn.Module, nn.Parameter]:
        """
        Create the layers and parameter that represent the distribution:
        one output will be the mean of the Gaussian, the other parameter will be the
        standard deviation (log std in fact to allow negative values)

        :param latent_dim: Dimension of the last layer of the policy (before the action layer)
        :param log_std_init: Initial value for the log standard deviation
        :return:
        """
        mean_actions = nn.Identity((latent_dim, self.action_dim))
        log_std = nn.Parameter(th.ones(self.action_dim) * log_std_init, requires_grad=True)
        return mean_actions, log_std


class CustomExtractor(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param observation_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param action_dim: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        value_dim: int = 1,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = action_dim
        self.latent_dim_vf = value_dim
        self.obs_dim= observation_dim

        # Policy network
        self.policy_net =  nn.Linear(in_features= observation_dim,
                      out_features= action_dim,
                      bias= False)

        # self.policy_net= MultiNN(controllers_number= self.obs_dim//2)
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(self.obs_dim, 4), nn.ReLU(),
            nn.Linear(4,2), nn.ReLU(),
            nn.Linear(2,value_dim)
        )

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features) 

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        
        self.obs_dim= observation_space.shape[0]      # type: ignore
        self.act_dim= action_space.shape[0]      # type: ignore
        
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            *args,
            **kwargs,
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomExtractor(self.obs_dim,
                                             self.act_dim)


    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        self.action_dist = CustomDistribution(get_action_dim(self.action_space))

        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init)
        
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]




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