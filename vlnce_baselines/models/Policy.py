"""
Design a policy to combine different maps then decide action
"""
import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from typing import List
import supervision as sv
from collections import Sequence

from habitat import Config

from vlnce_baselines.utils.map_utils import *
from vlnce_baselines.utils.data_utils import OrderedSet
from vlnce_baselines.models.fmm_planner import FMMPlanner
from vlnce_baselines.models.frontier_policy import FrontierPolicy


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
        
        self.frontier_policy = FrontierPolicy()
        
    def reset(self) -> None:
        self.frontier_policy.reset()
        self.max_destination_confidence = -1.
        self.vis_image = np.ones((self.map_shape, self.map_shape, 3)).astype(np.uint8) * 255
    
    def _get_action(self, 
                    full_pose: Sequence, 
                    waypoint: np.ndarray, 
                    map: np.ndarray, 
                    step: int,
                    current_episode_id: int) -> int:
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
            normalized_data = cv2.circle(normalized_data, (waypoint[1], waypoint[0]), 
                                         radius=5, color=(0,0,255), thickness=2)
            cv2.imshow("fmm distance field", np.flipud(normalized_data))
            cv2.waitKey(1)
        if self.print_images:
            save_dir = os.path.join(self.config.RESULTS_DIR, "fmm_fields/eps_%d"%current_episode_id)
            os.makedirs(save_dir, exist_ok=True)
            fn = "{}/step-{}.png".format(save_dir, step)
            cv2.imwrite(fn, np.flipud(normalized_data))
        
        return action
    
    def _search_destination(self, 
                            destination: str, 
                            current_value: float,
                            max_value: float,
                            classes: List,
                            detected_classes: OrderedSet, 
                            one_step_full_map: np.ndarray, 
                            full_map: np.ndarray, 
                            current_detection: sv.Detections, step: int):
        if destination not in detected_classes:
            """ 
            havn't detected destination
            """
            return None
        
        map_idx = detected_classes.index(destination)
        destination_map = one_step_full_map[4 + map_idx]
        class_idx = classes.index(destination)
        class_ids = current_detection.class_id
        confidences = current_detection.confidence
        # masks = current_detection.mask
        
        if class_idx not in class_ids:
            """ 
            Agent have already seen the destination in the past but not detected it in current step
            """
            return None
        
        destination_ids = np.argwhere(class_ids == class_idx)
        destination_confidences = confidences[destination_ids]
        max_confidence_idx = np.argmax(destination_confidences)
        max_idx = destination_ids[max_confidence_idx].item()
        # destination_mask = masks[max_idx]
        destination_confidence = confidences[max_idx]
        if destination_confidence > self.max_destination_confidence:
            self.max_destination_confidence = destination_confidence
            
        # plt.imshow(destination_mask)
        # plt.savefig("/data/ckh/Zero-Shot-VLN-FusionMap/tests/destination_mask/mask%d.png"%step)
        # np.save("/data/ckh/Zero-Shot-VLN-FusionMap/tests/destination_mask/mask%d.npy"%step, destination_mask)
        
        destination_waypoint = process_destination(destination_map, full_map)
        confidence_part = destination_confidence / self.max_destination_confidence
        value_part = current_value / max_value
        score = (confidence_part + value_part) / 2.0
        
        if score >= 0.5 and current_value >= 0.25 and destination_waypoint is not None:
            return destination_waypoint
        else:
            return None
            
    def forward(self, 
                value_map: np.ndarray, 
                full_map: np.ndarray, 
                full_pose: Sequence, 
                frontiers: np.ndarray, 
                destination: str, 
                classes: List,
                detected_classes: OrderedSet, 
                one_step_full_map: np.ndarray, 
                current_detection: sv.Detections, 
                current_episode_id: int,
                step: int):
        
        x, y, heading = full_pose
        x, y = x * (100 / self.resolution), y * (100 / self.resolution)
        position = np.array([y, x])
        best_waypoint, best_value, sorted_waypoints = self.frontier_policy(frontiers, value_map, position)
        print("===best waypoint: ", best_waypoint)
        print("current_position's value: ", value_map[int(y), int(x)])
        print("current pose: ", full_pose)
        current_value = value_map[int(y), int(x)]
        max_value = np.max(value_map)
        destination_waypoint = self._search_destination(destination, current_value, max_value,
                                 classes, detected_classes, 
                                 one_step_full_map, full_map, current_detection, step)
        if destination_waypoint is not None:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!destination waypoint: ", destination_waypoint)
            best_waypoint = destination_waypoint
        # best_waypoint = np.array([302, 206])
        action = self._get_action(full_pose, best_waypoint, full_map, step, current_episode_id)
        
        if self.visualize:
            self._visualization(value_map, sorted_waypoints, best_waypoint, step, current_episode_id)
        
        return {"action": action}
    
    def _visualization(self, 
                       value_map: np.ndarray, 
                       waypoints: np.ndarray, 
                       best_waypoint: np.ndarray, 
                       step: int,
                       current_episode_id: int):
        
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
            save_dir = os.path.join(self.config.RESULTS_DIR, "waypoints/eps_%d"%current_episode_id)
            os.makedirs(save_dir, exist_ok=True)
            fn = "{}/step-{}.png".format(save_dir, step)
            cv2.imwrite(fn, self.vis_image)