from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import torch as th
from torch import nn

from stable_baselines3.common.distributions import (DiagGaussianDistribution,
                                                    StateDependentNoiseDistribution)


class MultiPI(nn.Module):
    def __init__(self, controllers_number) -> None:
        super().__init__()
        self.linears = nn.ModuleList([nn.Linear(2, 1, bias= False) for i in range(controllers_number)])

    def forward(self, x : th.Tensor) -> th.Tensor:
        y= th.Tensor()
        for i,l in enumerate(self.linears):
            y= th.cat([y, l(x[0][2*i:2*(i+1)]) ])
        return y

class MultiPIpositive(nn.Module):
    def __init__(self, controllers_number) -> None:
        super().__init__()
        self.params = nn.Parameter(th.Tensor([[0,0]]* controllers_number))

    def forward(self, x : th.Tensor) -> th.Tensor:
        y= th.Tensor()
        for i,p in enumerate(self.params):
            y= th.cat([y, th.matmul( th.exp(p), x[2*i:2*(i+1)].T)]) 
        return y

class NN(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.nonlinear = nn.Sequential(
            nn.Linear(input_dim, 4), nn.ReLU(),
            nn.Linear(4,2), nn.ReLU(),
            nn.Linear(2,output_dim))

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.nonlinear(x)

class Quadratic(nn.Module):
    def __init__(self, input_dim, output_dim) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias= False)

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.linear(th.square(x))
    

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

class CustomgSDEDistribution(StateDependentNoiseDistribution):
    def __init__(self, action_dim: int, use_expln: bool):
        super().__init__(action_dim= action_dim,
                         use_expln= use_expln)
    
    def proba_distribution_net(
            self, latent_dim: int, log_std_init: float = 0.0, latent_sde_dim: Optional[int] = None
        ) -> tuple[nn.Module, nn.Parameter]:
            """
            Create the layers and parameter that represent the distribution:
            one output will be the deterministic action, the other parameter will be the
            standard deviation of the distribution that control the weights of the noise matrix.

            :param latent_dim: Dimension of the last layer of the policy (before the action layer)
            :param log_std_init: Initial value for the log standard deviation
            :param latent_sde_dim: Dimension of the last layer of the features extractor
                for gSDE. By default, it is shared with the policy network.
            :return:
            """
            mean_actions_net = nn.Identity((latent_dim, self.action_dim))
            self.latent_sde_dim = latent_dim if latent_sde_dim is None else latent_sde_dim
            log_std = nn.Parameter(th.ones(self.latent_sde_dim,self.action_dim) * log_std_init, requires_grad=True)
            self.sample_weights(log_std)
            return mean_actions_net, log_std