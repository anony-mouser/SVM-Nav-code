"""
Implement other sensors if needed.
"""

import numpy as np
from gym import spaces
from typing import Any

from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, Simulator, SensorTypes

from habitat_extensions.pose_utils import get_pose_change, get_start_sim_location, get_sim_location


@registry.register_sensor
class SensorPoseSensor(Sensor):
    """It is a senor to get sensor's pose
    """
    def __init__(self, sim: Simulator, config: Config,  *args: Any, **kwargs: Any) -> None:
        super().__init__(config=config)
        self._sim = sim
        self.episode_start = False
        self.last_sim_location = get_sim_location(self._sim)
        self.sensor_pose = [0., 0., 0.] # initialize last sensor pose as [0,0,0]
    
    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "sensor_pose"
    
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.TACTILE
    
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-100., high=100, shape=(3,), dtype=np.float)

    def get_observation(self, observations, *args: Any, episode, **kwargs: Any):
        if not self.episode_start:
            start_position = episode.start_position
            start_rotation = np.quaternion(episode.start_rotation[-1], *episode.start_rotation[:-1])
            self.last_sim_location = get_start_sim_location(start_position, start_rotation)
            self.episode_start = True
        dx, dy, do, self.last_sim_location = get_pose_change(self._sim, self.last_sim_location)
        self.sensor_pose = [dx, dy, do]
        
        return self.sensor_pose