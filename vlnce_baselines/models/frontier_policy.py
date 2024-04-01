import cv2
import numpy as np
import torch.nn as nn
from typing import List
from collections import Sequence
from scipy.spatial.distance import cdist
from skimage.morphology import remove_small_objects
from vlnce_baselines.models.waypoint_policy import WaypointSelector


class FrontierPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.waypoint_selector = WaypointSelector()
        
    def reset(self) -> None:
        self.waypoint_selector.reset()
    
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
    
    def _sort_waypoints_by_value(self, frontiers: np.ndarray, value_map: np.ndarray, position: np.ndarray) -> List:
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(frontiers)
        centroids = centroids[1:]
        tmp_waypoints = [[int(item[1]), int(item[0])] for item in centroids]
        waypoints = []
        for waypoint in tmp_waypoints:
            value = value_map[waypoint[0], waypoint[1]]
            if value == 0:
                print("find a waypoint whose value is 0: ", waypoint)
                nearest_waypoint = self._get_nearest_nonzero_waypoint(value_map.astype(bool), waypoint)
                print("nearest not zero waypoint: ", nearest_waypoint)
                waypoints.append(nearest_waypoint)
            else:
                waypoints.append(waypoint)
        waypoints_value = [[waypoint, value_map[waypoint[0], waypoint[1]]] for waypoint in waypoints]
        waypoints_value = sorted(waypoints_value, key=lambda x: x[1], reverse=True)
        if len(waypoints_value) > 0:
            sorted_waypoints = np.concatenate([[np.array(item[0])] for item in waypoints_value], axis=0)
            sorted_values = [item[1] for item in waypoints_value]
        else:
            sorted_waypoints = np.expand_dims(position.astype(int), axis=0)
            sorted_values = [value_map[int(position[0]), int(position[1])]]
        
        return sorted_waypoints, sorted_values
    
    def _get_nearest_nonzero_waypoint(self, arr: np.ndarray, start: Sequence) -> np.ndarray:
        nonzero_indices = np.argwhere(arr != 0)
        distances = cdist([start], nonzero_indices)
        nearest_index = np.argmin(distances)
        
        return np.array(nonzero_indices[nearest_index])
    
    def forward(self, frontiers: np.ndarray, value_map: np.ndarray, position: np.ndarray):
        sorted_waypoints, sorted_values = self._sort_waypoints_by_value(frontiers, value_map, position)
        best_waypoint, best_value, sorted_waypoints = \
            self.waypoint_selector(sorted_waypoints, sorted_values, position)
        
        return best_waypoint, best_value, sorted_waypoints