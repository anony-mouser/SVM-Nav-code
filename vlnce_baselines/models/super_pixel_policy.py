import cv2
import numpy as np
import torch.nn as nn
from typing import List
from collections import Sequence
from vlnce_baselines.utils.map_utils import *
from vlnce_baselines.models.waypoint_policy import WaypointSelector


class SuperPixelPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.waypoint_selector = WaypointSelector()
    
    def reset(self) -> None:
        self.waypoint_selector.reset()
    
    def _get_sorted_regions(self, full_map: np.ndarray, value_map: np.ndarray):
        floor = process_floor(full_map)
        traversible = get_traversible_area(full_map)
        
        valid_mask = value_map.astype(bool)
        min_val = np.min(value_map)
        max_val = np.max(value_map)
        normalized_values = (value_map - min_val) / (max_val - min_val)
        normalized_values[value_map == 0] = 1
        img = cv2.applyColorMap((normalized_values* 255).astype(np.uint8), cv2.COLORMAP_HOT)
        slic = cv2.ximgproc.createSuperpixelSLIC(img, region_size=20, ruler=20.0) 
        slic.iterate(10)
        mask_slic = slic.getLabelContourMask()
        mask_slic *= valid_mask
        label_slic = slic.getLabels()
        label_slic *= valid_mask
        valid_labels = np.unique(label_slic)[1:]
        value_regions = []
        for label in valid_labels:
            mask = np.zeros_like(mask_slic)
            mask[label_slic == label] = 1
            mask_value = mask * value_map
            nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
            waypoint = np.array([int(centroids[1][1]), int(centroids[1][0])])
            waypoint = get_nearest_nonzero_waypoint(np.logical_and(floor, traversible), waypoint)
            value_regions.append((mask, np.average(mask_value), waypoint))
        sorted_regions = sorted(value_regions, key=lambda x: x[1], reverse=True)
        
        return sorted_regions

    def _sorted_waypoints(self, sorted_regions: List, top_k: int=3):
        waypoints, values = [], []
        for item in sorted_regions[:top_k]:
            waypoints.append([item[2]])
            values.append(item[1])
        waypoints = np.concatenate(waypoints, axis=0)
        
        return waypoints, values
    
    def forward(self, full_map: np.ndarray, value_map: np.ndarray, position: Sequence):
        if np.sum(value_map.astype(bool)) < 20 * 20:
            best_waypoint = position
            best_value = 0
            sorted_waypoints = [position]
            
            return best_waypoint, best_value, sorted_waypoints
        else:
            sorted_regions = self._get_sorted_regions(full_map, value_map)
            sorted_waypoints, sorted_values = self._sorted_waypoints(sorted_regions)
            best_waypoint, best_value, sorted_waypoints = \
                self.waypoint_selector(sorted_waypoints, sorted_values, position)
                
            return best_waypoint, best_value, sorted_waypoints