"""
https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html
Advanced Example

https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/common/distributions.html#CategoricalDistribution.proba_distribution_net
Distributions
"""

from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gymnasium import spaces
import torch as th
from torch import nn
import numpy as np
from functools import partial

from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.preprocessing import get_action_dim


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
        mean_actions = nn.Linear(latent_dim, self.action_dim, bias=False)
        log_std = nn.Parameter(th.ones(self.action_dim) * log_std_init, requires_grad=True)
        return mean_actions, log_std


class CustomExtractor(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int = 2,
        last_layer_dim_pi: int = 1,
        last_layer_dim_vf: int = 1,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, 1, bias=False)
            )
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, 4), nn.ReLU(),
            nn.Linear(4,2), nn.ReLU(),
            nn.Linear(2,1)
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
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomExtractor(self.features_dim)

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