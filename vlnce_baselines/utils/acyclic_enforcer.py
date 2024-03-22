# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any, Set

import numpy as np


class StateAction:
    def __init__(self, position: np.ndarray, waypoint: Any):
        self.position = position
        self.waypoint = waypoint

    def __hash__(self) -> int:
        string_repr = f"{self.position}_{self.waypoint}"
        return hash(string_repr)


class AcyclicEnforcer:
    history: Set[StateAction] = set()

    def check_cyclic(self, position: np.ndarray, waypoint: Any) -> bool:
        state_action = StateAction(position, waypoint)
        cyclic = state_action in self.history
        return cyclic

    def add_state_action(self, position: np.ndarray, waypoint: Any) -> None:
        state_action = StateAction(position, waypoint)
        self.history.add(state_action)