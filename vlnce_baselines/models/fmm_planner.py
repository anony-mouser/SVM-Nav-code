import cv2
import copy
import skfmm
import numpy as np
from numpy import ma
from typing import Tuple


def get_mask(sx, sy, scale, step_size):
    size = int(step_size // scale) * 2 + 1 # size=11
    mask = np.zeros((size, size)) # (11,11)
    for i in range(size):
        for j in range(size):
            if ((i + 0.5) - (size // 2 + sx)) ** 2 + \
               ((j + 0.5) - (size // 2 + sy)) ** 2 <= \
                    step_size ** 2 \
               and ((i + 0.5) - (size // 2 + sx)) ** 2 + \
               ((j + 0.5) - (size // 2 + sy)) ** 2 > \
                    (step_size - 1) ** 2:
                mask[i, j] = 1

    mask[size // 2, size // 2] = 1
    return mask


def get_dist(sx, sy, scale, step_size):
    size = int(step_size // scale) * 2 + 1
    mask = np.zeros((size, size)) + 1e-10
    for i in range(size):
        for j in range(size):
            if ((i + 0.5) - (size // 2 + sx)) ** 2 + \
               ((j + 0.5) - (size // 2 + sy)) ** 2 <= \
                    step_size  ** 2:
                mask[i, j] = max(5,
                                 (((i + 0.5) - (size // 2 + sx)) ** 2 +
                                  ((j + 0.5) - (size // 2 + sy)) ** 2) ** 0.5)
    return mask


class FMMPlanner:
    def __init__(self, traversible: np.ndarray, scale: int=1, step_size: int=5, visualize: bool=False) -> None:
        self.scale = scale
        self.step_size = step_size
        self.visualize = visualize
        if scale != 1.:
            self.traversible = cv2.resize(traversible,
                                          (traversible.shape[1] // scale,
                                           traversible.shape[0] // scale),
                                          interpolation=cv2.INTER_NEAREST)
            self.traversible = np.rint(self.traversible)
        else:
            self.traversible = traversible

        self.du = int(self.step_size / (self.scale * 1.)) # du=5
        self.fmm_dist = None
        
    def set_goal(self, goal: np.ndarray) -> None:
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        goal_x, goal_y = goal

        traversible_ma[goal_x, goal_y] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd = ma.filled(dd, np.max(dd) + 1)
        self.fmm_dist = dd
    
    def get_short_term_goal(self, agent_position: np.ndarray) -> Tuple:
        dist = copy.deepcopy(self.fmm_dist)
        x, y = int(agent_position[0]), int(agent_position[1])
        dx, dy = agent_position[0] - x, agent_position[1] - y
        mask = get_mask(dx, dy, scale=1, step_size=5)
        dist_mask = get_dist(dx, dy, scale=1, step_size=5)
        subset = dist[x - 5 : x + 6, y - 5: y + 6]
        subset *= mask
        subset += (1 - mask) * 1e5
        print("subset: ", subset[5,5])
        if subset[5, 5] < 0.25 * 100 / 5:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TRUE !!!!!!!!!!!!!!!!!")
            stop = True
        else:
            stop = False
        subset -= subset[5, 5]
        ratio = subset / dist_mask
        subset[ratio < -1.5] = 1
        
        if self.visualize:
            subset_vis = ((subset - np.min(subset)) / (np.max(subset) - np.min(subset)) * 255).astype(np.uint8)
            cv2.imshow("subset", np.flipud(subset_vis))
            cv2.waitKey(1)
        
        (stg_x, stg_y) = np.unravel_index(np.argmin(subset), subset.shape)
        offset_x, offset_y = stg_x - 5, stg_y - 5
        goal_x = x + offset_x
        goal_y = y + offset_y
        
        return goal_x, goal_y, stop