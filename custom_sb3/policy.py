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
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.preprocessing import get_action_dim

from custom_sb3 import FA


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
        # self.policy_net =  FA.NN(input_dim= observation_dim, output_dim= action_dim)
        self.policy_net= FA.MultiPI(controllers_number= self.obs_dim//2)
        
        # Value network
        # self.value_net = FA.NN(input_dim= observation_dim, output_dim= value_dim)
        self.value_net= nn.Linear(self.obs_dim, value_dim, bias= False)


    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
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
        use_expln = False,
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
            use_expln= use_expln,
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

        self.action_dist = FA.CustomDistribution(get_action_dim(self.action_space))
        # self.action_dist = FA.CustomgSDEDistribution(get_action_dim(self.action_space),
        #                                          use_expln=True)

        self.action_net, self.log_std = self.action_dist.proba_distribution_net(
                latent_dim=latent_dim_pi, log_std_init=self.log_std_init)
        
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1, bias= False)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr= lr_schedule(1), **self.optimizer_kwargs) # type: ignore
