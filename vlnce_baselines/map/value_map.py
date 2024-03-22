"""
Value map moudle aims to calcluate cosine similarity
between current observation and destination description
"""

import cv2
import torch
import torch.nn as nn

import numpy as np
from PIL import Image
from habitat import Config
from collections import Sequence
from typing import Union, Tuple, List
from lavis.models import load_model_and_preprocess

from vlnce_baselines.utils.map_utils import *
from vlnce_baselines.utils.visualization import *


class ValueMap(nn.Module):
    def __init__(self, 
                 config: Config, 
                 full_value_map_shape: Union[Tuple, List, np.ndarray]) -> None:
        super(ValueMap, self).__init__()
        self.config = config
        self.shape = full_value_map_shape
        self.visualize = config.MAP.VISUALIZE
        self.print_images = config.MAP.PRINT_IMAGES
        
        # two channels for value map: 
        # channel 0 is confidence map;
        # channel 1 is blip value map;
        self.value_map = np.zeros((2, *self.shape))
        self.resolution = config.MAP.MAP_RESOLUTION
        self.hfov = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV
        self.radius = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        self.device = (torch.device("cuda", self.config.TORCH_GPU_ID) if 
                       torch.cuda.is_available() else torch.device("cpu"))
        self.vis_image = np.ones((580, 480 * 3 + 20 * 4, 3)).astype(np.uint8) * 255
        self._load_model_from_disk()
    
    def _create_model(self):
        self.model, vis_processors, text_processors = \
            load_model_and_preprocess(
                "blip2_image_text_matching", 
                "pretrain", 
                device=self.device,
                is_eval=True)
        self.vis_processors = vis_processors["eval"]
        self.text_processors = text_processors["eval"]
        
    def _load_model_from_disk(self):
        self.model = torch.load(self.config.BLIP2_MODEL_DIR).to(self.device)
        self.vis_processors = torch.load(self.config.BLIP2_VIS_PROCESSORS_DIR)["eval"]
        self.text_processors = torch.load(self.config.BLIP2_TEXT_PROCESSORS_DIR)["eval"]
            
    def get_blip_value(self, image: Image, caption: str) -> torch.Tensor:
        img = self.vis_processors(image).unsqueeze(0).to(self.device)
        txt = self.text_processors(caption)
        itc_score = self.model({"image": img, "text_input": txt}, match_head='itc')
        
        return itc_score
    
    def _calculate_confidence(self, theta: np.ndarray) -> np.float64:
        return (np.cos(0.5 * np.pi * theta / (self.hfov / 2)))**2

    def _angle_to_vector(self, angle: np.ndarray) -> np.ndarray:
        angle_rad = np.radians(angle)
        x = np.cos(angle_rad)
        y = np.sin(angle_rad)
        
        return np.array([x, y])

    def _angle_between_vectors(self, vector1: np.ndarray, vector2: np.ndarray) -> np.ndarray:
        dot_product = np.sum(vector1 * vector2, axis=0)
        vector1_length = np.linalg.norm(vector1, axis=0)
        vector2_length = np.linalg.norm(vector2, axis=0)
        angle = np.arccos(dot_product / (vector1_length * vector2_length))
        
        return np.degrees(angle)

    def _create_sector_mask(self, position: Sequence, heading: float):
        """ 
        arg "position" came from full pose, full pose use standard Cartesian coordinate.
        """
        mask = np.zeros(self.shape)
        confidence_mask = np.zeros(self.shape)
        heading = (360 - heading) % 360
        angle_high = (heading + 79.0 / 2) % 360
        angle_low = (heading - 79.0 / 2) % 360
        heading = np.ones(self.shape) * heading
        heading_vector = self._angle_to_vector(heading)

        y, x = np.meshgrid(np.arange(self.shape[0]) - position[0], np.arange(self.shape[1]) - position[1])
        # x = np.flipud(x)
        distance = np.sqrt(x**2 + y**2)
        angle = np.arctan2(x, y) * 180 / np.pi
        angle = (360 - angle) % 360

        angle_vector = self._angle_to_vector(angle) # (2, 480, 480)
        theta = self._angle_between_vectors(heading_vector, angle_vector)

        confidence = self._calculate_confidence(theta)

        valid_distance = distance <= 100
        if angle_high > angle_low:
            valid_angle = (angle_low <= angle) & (angle <= angle_high)
        else:
            valid_angle = (angle_low <= angle) | (angle <= angle_high)
        mask[valid_distance & valid_angle] = 1
        confidence_mask[valid_distance & valid_angle] = confidence[valid_distance & valid_angle]

        return mask, confidence_mask

    def _update_value_map(self, 
                          prev_value: np.ndarray, 
                          curr_value: np.ndarray, 
                          prev_confidence: np.ndarray, 
                          curr_confidence: np.ndarray,
                          mask: np.ndarray) -> np.ndarray:
        # import pdb;pdb.set_trace()
        new_map_mask = np.logical_and(curr_confidence < 0.35, curr_confidence < prev_confidence)
        curr_confidence[new_map_mask] = 0.0
        new_value = curr_confidence * curr_value * self.current_floor + prev_confidence * prev_value
        new_confidence = (self.current_floor * curr_confidence)**2 + prev_confidence**2
        partion = curr_confidence * self.current_floor + prev_confidence
        partion[partion == 0] = np.inf
        new_value /= partion
        new_confidence /= partion
        # new_value[new_map_mask] = 0.0
        self.value_map[0] = new_confidence
        self.value_map[1] = new_value
        # update_value = new_value * self.current_floor
        # update_mask = prev_value < update_value
        # self.value_map[1, update_mask] = update_value[update_mask]
    
    def forward(self,
                step: int,
                one_step_full_map: np.ndarray, 
                blip_value: np.ndarray,
                full_pose: Sequence):
        """project cosine similarity to floor

        Args:
            local_map (np.array): one step local map, current observation's 
                                  2D Top-down semantic map. shape: [c,h,w] 
                                  no batch dimension
            value (torch.Tensor): torch.size([1,1]) on device
        """
        self.current_floor = process_floor(one_step_full_map, kernel_size=3)
        position = full_pose[:2] * (100 / self.resolution)
        heading = full_pose[-1]
        mask, confidence_mask = self._create_sector_mask(position, heading)
        current_confidence = confidence_mask
        previous_confidence = self.value_map[0]
        current_value = blip_value
        previous_value = self.value_map[1]
        self._update_value_map(previous_value, current_value, previous_confidence, current_confidence, mask)
        if self.visualize:
            self._visualize(step)
        
    
    def _visualize(self, step):
        confidence_mask_vis = cv2.convertScaleAbs(self.value_map[0] * 255)
        confidence_mask_vis = np.stack((confidence_mask_vis,) * 3, axis=-1)
        # value_map_vis = cv2.convertScaleAbs(self.value_map[1] * 255)
        value_map_vis = self.value_map[1]
        
        min_val = np.min(value_map_vis)
        max_val = np.max(value_map_vis)
        normalized_values = (value_map_vis - min_val) / (max_val - min_val)
        normalized_values[value_map_vis == 0] = 1
        value_map_vis = cv2.applyColorMap((normalized_values* 255).astype(np.uint8), cv2.COLORMAP_HOT)
        floor_vis = cv2.convertScaleAbs(self.current_floor * 255)
        floor_vis = np.stack((floor_vis,) * 3, axis=-1)
        self.vis_image[80 : 560, 20 : 500] = np.flipud(floor_vis)
        self.vis_image[80: 560, 520 : 1000] = np.flipud(confidence_mask_vis)
        self.vis_image[80: 560, 1020: 1500] = np.flipud(value_map_vis)
        
        self.vis_image = add_text(self.vis_image, "Floor", (20, 50))
        self.vis_image = add_text(self.vis_image, "Confidence Mask", (520, 50))
        self.vis_image = add_text(self.vis_image, "Value Map", (1020, 50))
        
        cv2.imshow("info", self.vis_image)
        cv2.waitKey(1)
        
        if self.print_images:
            fn = "{}/step-{}.png".format("data/logs/eval_results/exp1/floor_confidence_value", step)
            cv2.imwrite(fn, self.vis_image)