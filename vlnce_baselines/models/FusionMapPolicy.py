"""
Design a policy net to combine different maps then decide a waypoint
"""
import torch.nn as nn

from habitat_baselines.rl.ppo import Policy
from habitat_baselines.common.baseline_registry import baseline_registry


@baseline_registry.register_policy
class FusionMapPolicy(Policy):
    def __init__(self, net, dim_actions):
        super().__init__(net, dim_actions)
        

class FusionMap(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass
    
    def forward(self,):
        pass