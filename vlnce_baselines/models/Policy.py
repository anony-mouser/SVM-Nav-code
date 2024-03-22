"""
Design a policy net to combine different maps then decide a waypoint
"""

import cv2
import torch
import numpy as np
import torch.nn as nn
from typing import List
from collections import Sequence
from skimage.morphology import remove_small_objects

from habitat import Config

from vlnce_baselines.utils.map_utils import *
from vlnce_baselines.models.fmm_planner import FMMPlanner
from vlnce_baselines.utils.acyclic_enforcer import AcyclicEnforcer


class FusionMapPolicy(nn.Module):
    def __init__(self, config: Config, map_shape: float=480) -> None:
        super().__init__()
        self.config = config
        self.map_shape = map_shape
        self.visualize = config.MAP.VISUALIZE
        self.print_images = config.MAP.PRINT_IMAGES
        self.resolution = config.MAP.MAP_RESOLUTION
        self.turn_angle = config.TASK_CONFIG.SIMULATOR.TURN_ANGLE
        
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        
    def reset(self) -> None:
        self._acyclic_enforcer = AcyclicEnforcer()
        self._last_value = float("-inf")
        self._last_waypoint = np.zeros(2)
        self.waypoint = []
        self.vis_image = np.ones((self.map_shape, self.map_shape, 3)).astype(np.uint8) * 255
        
    def _get_waypoint(self, value_map: np.ndarray) -> List:
        ret, thresh = cv2.threshold(value_map, 0.1, 1.0, cv2.THRESH_BINARY)
        thresh = remove_small_objects(thresh.astype(bool), min_size=180).astype(np.uint8)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        
        avg_values = []
        for i in range(1, nb_components):
            tmp = value_map * (output == i)
            avg_value = tmp.sum() / (tmp != 0).sum()
            avg_values.append((i, avg_value))
        avg_values = sorted(avg_values, key=lambda x: x[1], reverse=True)
        
        if len(avg_values) >= 3000:
            top_idx = [item[0] for item in avg_values[:3]]
        else:
            top_idx = [avg_values[0][0]]
        top_centroids = centroids[top_idx]
        waypoints = [(int(item[1]), int(item[0])) for item in top_centroids]
        
        return waypoints
    
    def _sort_waypoints_by_value(self, frontiers: np.ndarray, value_map: np.ndarray) -> List:
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(frontiers)
        centroids = centroids[1:]
        waypoints = [[int(item[1]), int(item[0])] for item in centroids]
        waypoints_value = [[waypoint, value_map[waypoint[0], waypoint[1]]] for waypoint in waypoints]
        waypoints_value = sorted(waypoints_value, key=lambda x: x[1], reverse=True)
        sorted_waypoints = np.concatenate([[np.array(item[0])] for item in waypoints_value], axis=0)
        sorted_values = [item[1] for item in waypoints_value]
        
        return sorted_waypoints, sorted_values
    
    def _get_best_waypoint(self, frontiers: np.ndarray, value_map: np.ndarray, position: np.ndarray):
        sorted_waypoints, sorted_values = self._sort_waypoints_by_value(frontiers, value_map)
        best_waypoint_idx = None
        
        if not np.array_equal(self._last_waypoint, np.zeros(2)):
            curr_index = None
            
            for idx, waypoint in enumerate(sorted_waypoints):
                if np.array_equal(waypoint, self._last_waypoint):
                    curr_index = idx
                    break
            
            if curr_index is None:
                closest_index = closest_point_within_threshold(sorted_waypoints, self._last_waypoint, threshold=0.5)
                if closest_index != -1:
                    curr_index = closest_index
            else:
                curr_value = sorted_values[curr_index]
                if curr_value + 0.01 > self._last_value:
                    best_waypoint_idx = curr_index
        
        if best_waypoint_idx is None:
            for idx, waypoint in enumerate(sorted_waypoints):
                cyclic = self._acyclic_enforcer.check_cyclic(position, waypoint)
                if cyclic:
                    continue
                best_waypoint_idx = idx
                break
        
        if best_waypoint_idx is None:
            print("All waypoints are cyclic! Choosing the closest one.")
            best_waypoint_idx = max(range(len(sorted_waypoints)), 
                                    key=lambda i: np.linalg.norm(sorted_waypoints[i] - position))
        
        best_waypoint = sorted_waypoints[best_waypoint_idx]
        best_value = sorted_values[best_waypoint_idx]
        self._acyclic_enforcer.add_state_action(position, best_waypoint)
        self._last_value = best_value
        self._last_waypoint = best_waypoint
        
        return best_waypoint, best_value, sorted_waypoints
    
    def _get_action(self, full_pose: Sequence, waypoint: np.ndarray, map: np.ndarray) -> int:
        """
        The coordinates among agent's pose in full_pose, agent's position in full_map, 
        agent's position in visualization are ignoring. And there're many np.flipud which
        could be confusing.
        
        PAY ATTENTION:
        
        1. full pose: [x, y, heading] -> standard cartesian coordinates
           agent's initial pose is [12, 12, 0]. (12, 12) is the center of the whole range
           heading = 0 in cartesian is pointing in the right direction.
           
           Now let's take an example: agent's full_pose=[7, 21, 0].
           ^ y
           | 
           | * (7, 21) => (7*100/5, 21*100/5)=(140, 420)
           |
           |
           |
            -------------> x
        
           
        2. what's the agent's position in full map?
           full_map.shape=[480, 480], agent's initial index is [240, 240]
           since full_map is a 2D ndarray so the x axis points downward and y axis points rightward
            -------------> y
           |
           | * (60, 140)
           |
           |
           |
           V x
           
           when we want to convert agent's position from cartesian coordinate to ndarray coordinate
           x_ndarray = 480 - y_cartesian
           y_ndarray = x_cartesian
           so the index in full_map is [60, 140]
        
           NOTICE: the agent didn't move when you convert coordinate from cartesian to ndarray, which
           means you should not just rotate the coordinate 90 degrees
           
        3. Does that finish?
           No! You should be extreamly careful that (60, 140) is the position we want to see in visualization
           but before visualization we will flip upside-down and this means (60, 140) is the position after
           flip upside-down. So, what's the index before flip upside-down?
           x_ndarray_raw = 480 - x_ndarray = y_cartesian
           y_ndarray_raw = y_ndarray = x_cartesian
           so the index in full_map before flip should be (420, 140)
           
        Till now, we have convert full_pose from cartesian coordinate to ndarray coordinate
        we designed a function: "angle_and_direction" to calculate wheather agent should turn 
        left or right to face the goal. this function takse in everything in ndarray coordinate. 
        
        We design it in this way because ndarray coordinate is the most commonly used.
        
        """
        x, y, heading = full_pose
        x, y = x * (100 / self.resolution), y * (100 / self.resolution)
        position = np.array([y, x])
        heading = -1 * full_pose[-1]
        rotation_matrix = np.array([[0, -1], 
                                    [1, 0]])
        
        objects = get_objects(map)
        obstacles = get_obstacle(map)
        traversible = 1 - (objects + obstacles)
        planner = FMMPlanner(traversible, visualize=self.visualize)
        
        goal_x = waypoint[0]
        goal_y = waypoint[1]
        goal = np.array([goal_x, goal_y])
        planner.set_goal(goal)
        stg_x, stg_y, stop = planner.get_short_term_goal(position)
        sub_waypoint = (stg_x, stg_y)
        heading_vector = angle_to_vector(heading)
        heading_vector = np.dot(rotation_matrix, heading_vector)
        waypoint_vector = sub_waypoint - position
        
        if stop:
            action = 0
            print("stop")
        else:
            relative_angle, action = angle_and_direction(heading_vector, waypoint_vector, self.turn_angle + 1)
        
        if self.visualize:
            normalized_data = ((planner.fmm_dist - np.min(planner.fmm_dist)) / 
                            (np.max(planner.fmm_dist) - np.min(planner.fmm_dist)) * 255).astype(np.uint8)
            normalized_data = np.stack((normalized_data,) * 3, axis=-1)
            normalized_data = cv2.circle(normalized_data, (int(x), int(y)), radius=5, color=(255,0,0), thickness=2)
            normalized_data = cv2.circle(normalized_data, (206, 302), radius=5, color=(0,0,255), thickness=2)
            cv2.imshow("fmm distance field", np.flipud(normalized_data))
            cv2.waitKey(1)
        
        return action

    def forward(self, 
                value_map: np.ndarray, 
                full_map: np.ndarray, 
                full_pose: Sequence, 
                frontiers: np.ndarray, 
                step: int):
        
        x, y, heading = full_pose
        x, y = x * (100 / self.resolution), y * (100 / self.resolution)
        position = np.array([x, y])
        best_waypoint, best_value, sorted_waypoints = self._get_best_waypoint(frontiers, value_map, position)
        print("===best waypoint: ", best_waypoint)
        best_waypoint = np.array([302, 206])
        action = self._get_action(full_pose, best_waypoint, full_map)
        
        if self.visualize:
            self._visualization(value_map, sorted_waypoints, best_waypoint, step)
        
        return {"action": action}
    
    def _visualization(self, 
                       value_map: np.ndarray, 
                       waypoints: np.ndarray, 
                       best_waypoint: np.ndarray, 
                       step: int):
        
        min_val = np.min(value_map)
        max_val = np.max(value_map)
        normalized_values = (value_map - min_val) / (max_val - min_val)
        normalized_values[value_map == 0] = 1
        map_vis = cv2.applyColorMap((normalized_values* 255).astype(np.uint8), cv2.COLORMAP_HOT)

        for i, waypoint in enumerate(waypoints):
            cx, cy = waypoint
            if i == 0:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)
            map_vis = cv2.circle(map_vis, (cy, cx), radius=3, color=color, thickness=1)
        map_vis = cv2.circle(map_vis, (best_waypoint[1], best_waypoint[0]), radius=5, color=(0,255,0), thickness=1)
        map_vis = np.flipud(map_vis)
        self.vis_image[:, :] = map_vis
        cv2.imshow("waypoints", self.vis_image)
        cv2.waitKey(1)
        
        if self.print_images:
            fn = "{}/step-{}.png".format("data/logs/eval_results/exp1/waypoints", step)
            cv2.imwrite(fn, self.vis_image)