import cv2
import torch
import torch.nn.functional as F

import numpy as np
from skimage.morphology import remove_small_objects, closing, disk


def get_grid(pose, grid_size, device):
    """
    Input:
        `pose` FloatTensor(bs, 3)
        `grid_size` 4-tuple (bs, _, grid_h, grid_w)
        `device` torch.device (cpu or gpu)
    Output:
        `rot_grid` FloatTensor(bs, grid_h, grid_w, 2)
        `trans_grid` FloatTensor(bs, grid_h, grid_w, 2)

    """
    pose = pose.float()
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    bs = x.size(0)
    t = t * np.pi / 180. # t.shape=[bs]
    cos_t = t.cos()
    sin_t = t.sin()

    theta11 = torch.stack([cos_t, -sin_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1) # [bs, 3]
    theta12 = torch.stack([sin_t, cos_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta1 = torch.stack([theta11, theta12], 1) # [bs, 2, 3] rotation matrix

    theta21 = torch.stack([torch.ones(x.shape).to(device),
                           -torch.zeros(x.shape).to(device), x], 1)
    theta22 = torch.stack([torch.zeros(x.shape).to(device),
                           torch.ones(x.shape).to(device), y], 1)
    theta2 = torch.stack([theta21, theta22], 1) # [bs, 2, 3] translation matrix

    rot_grid = F.affine_grid(theta1, torch.Size(grid_size))
    trans_grid = F.affine_grid(theta2, torch.Size(grid_size))

    return rot_grid, trans_grid


def get_obstacle(map: np.ndarray, kernel_size: int=3) -> np.ndarray:
    floor = map[4, ...] # channel is floor area
    floor_mask = 1 - (floor != 0)
    obstacle = map[1, ::-1] * floor_mask # channel 1 is explored area
    obstacle = remove_small_objects(
        obstacle.astype(bool), 
        min_size=50, # you can try different minimum object size
        connectivity=5)
    selem = disk(kernel_size)
    obstacle = closing(obstacle, selem)
    
    return obstacle.astype(bool)


def get_objects(map: np.ndarray, kernel_size: int=3) -> np.ndarray:
    objects = np.zeros(map.shape[-2:])
    for obj in map[5:, ...]:
        obj = remove_small_objects(
            obj.astype(bool), min_size=50)
        selem = disk(kernel_size)
        obj = closing(obj, selem)
        objects += obj
    
    return objects.astype(bool)


def get_floor(map: np.ndarray, kernel_size: int=3) -> np.ndarray:
    floor = map[4, ...]
    floor = remove_small_objects(
        floor.astype(bool), min_size=50)
    selem = disk(kernel_size)
    floor = closing(floor, selem)
    
    return floor.astype(bool)


def process_floor(map: np.ndarray, kernel_size: int=3) -> np.ndarray:
    floor = get_floor(map, kernel_size)
    obstacles = get_obstacle(map, kernel_size)
    objects = get_objects(map, kernel_size)
    
    obstacles_mask = (obstacles == False)
    objects_mask = (objects == False)
    floor = floor * objects_mask * obstacles_mask
    floor = remove_small_objects(
        floor.astype(bool), min_size=50)
    floor = np.flipud(floor)
    
    return floor