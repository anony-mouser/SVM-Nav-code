from typing import Any, Dict, Union

import habitat
from habitat import Config, Dataset
from habitat.core.simulator import Observations
from habitat_baselines.common.baseline_registry import baseline_registry


@baseline_registry.register_env(name="VLNCEZeroShotEnv")
class VLNCEZeroShotEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Union[Dataset,None]=None) -> None:
        super().__init__(config.TASK_CONFIG, dataset)
    
    def reset(self) -> Observations:
        return super().reset()
    
    def step(self, action: Union[int, str, Dict[str, Any]], **kwargs) -> Observations:
        obs = super().step(action, **kwargs)
        done = self.get_done(obs)
        info = self.get_info(obs)
        
        return obs, done, info
        
    def get_info(self, observations: Observations) -> Dict[Any, Any]:
        return self.habitat_env.get_metrics()
    
    def get_done(self, observations):
        return self._env.episode_over
    
    def get_reward_range(self):
        return (0.0, 0.0)
    
    def get_reward(self, observations: Observations) -> Any:
        return 0.0
    
    